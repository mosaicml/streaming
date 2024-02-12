# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A mid-epoch-resumable streaming/caching pytorch IterableDataset."""

import json
import logging
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures._base import Future
from enum import IntEnum
from math import ceil
from multiprocessing import Pool
from tempfile import gettempdir
from threading import Event, Lock
from time import sleep, time_ns
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray
from torch import distributed as dist
from torch.utils.data import IterableDataset

from streaming.array import Array
from streaming.batching import generate_work
from streaming.constant import (BARRIER, BARRIER_FILELOCK, CACHE_FILELOCK, CACHE_USAGE, EPOCH_DATA,
                                EPOCH_SHAPE, NEXT_EPOCH, RESUME, SHARD_ACCESS_TIMES, SHARD_STATES,
                                TICK)
from streaming.distributed import maybe_init_dist
from streaming.format.base.phaser import Phaser
from streaming.sampling import get_sampling
from streaming.shared import (SharedArray, SharedBarrier, SharedMemory, SharedScalar, _get_path,
                              get_shm_prefix)
from streaming.spanner import Spanner
from streaming.stream.base import Stream
from streaming.util.shorthand import normalize_bytes, normalize_count
from streaming.world import World

# An arbitrary time in the future, used for cold shard eviction.
NEVER = np.iinfo(np.uint64).max

logger = logging.getLogger(__name__)


class _ShardState(IntEnum):
    """The download status of a shard.

    Restrictions:
    - The initial state of INVALID must be zero.
    - State transitions: REMOTE -> FETCHING -> LOCAL -> REMOTE.
    """
    INVALID = 0  # The state is allocated (e.g., in an array), but not initialized yet.
    REMOTE = 1  # The shard exists only at the remote source.
    FETCHING = 2  # The shard is currently being worked on: (a) downloading from remote to local,
    # (b) decompressing zip-only, etc.
    LOCAL = 3  # Some form of the shard (raw or zip) exists locally (as well as remotely).


class _IterState(IntEnum):
    """The iter status of an _Iterator.

    Restrictions:
    - State transitions: ITERATING -> EXITING -> EXITED.
    """
    ITERATING = 0  # We are currently iterating through an epoch.
    EXITING = 1  # We have been signalled to end the epoch (either we hit end of __iter__, or
    # someone else started a new epoch, of which only one can be valid at a time).
    EXITED = 2  # All threads have noticed the exit signal and exited.


class _Iterator:
    """State of StreamingDataset __iter__, used to track and coordinate its threads.

    Has methods to implement early exit when a new epoch is started before the last one is done.

    Order of threads: 0 <= yield loop <= ready thread <= download thread <= total.

    Three indices:
    * Download index: points to the sample we are presently downloading, skipping other workers'
      downloads in progress.
    * Ready index: points to the farthest contiguously downloaded sample by any worker on this
      node.
    * Yield index: points to the (downloaded) sample that we are currently yielding.

    Args:
        sample_ids (NDArray[np.int64]): This worker's samples to download and yield.
    """

    # The number of threads (`download`, `ready`, `yield``) to wait on the exits of before
    # returning. The `yield` main thread exits at the end of epoch(s).
    _num_threads_to_exit = 2

    def __init__(self, sample_ids: NDArray[np.int64]) -> None:
        self.sample_ids = sample_ids

        self.total = len(sample_ids)
        self.fetch_index = 0
        self.ready_index = 0
        self.yield_index = 0

        self._lock = Lock()
        self._state = 0
        self._num_exited = 0

        # python will attempt to join all threads on shutdown.
        # Here, we register a call to self.non_blocking_exit to run
        # at shutdown to prevent a deadlock.
        # In python version >=3.9 this can be accomplished via
        # threading._register_atexit but not with the atexit module.
        # In older python versions, the atexit module can be used, and
        # threading._register_atexit does not exist.
        if sys.version_info[1] <= 8:  # check if python version <=3.8
            import atexit
            atexit.register(self.non_blocking_exit)
        else:
            from threading import _register_atexit  # pyright: ignore
            _register_atexit(self.non_blocking_exit)

    def non_blocking_exit(self) -> None:
        """Signal threads to exit without blocking.

        This will be called at process exit.
        """
        with self._lock:
            if self._state == _IterState.ITERATING:
                self._state = _IterState.EXITING

    def exit(self) -> None:
        """Signal threads to exit, wait until they have all exited, then return.

        This is called when the user starts a new epoch without the threads from the previous epoch
        having exited yet.
        """
        # Signal threads to exit.
        with self._lock:
            if self._state == _IterState.ITERATING:
                self._state = _IterState.EXITING
            elif self._state == _IterState.EXITING:
                pass
            elif self._state == _IterState.EXITED:
                return
            else:
                raise RuntimeError(f'Invalid _IterState: {self._state}')

        # Block until they have all exited, updating _state to done.
        while True:
            with self._lock:
                if self._num_exited >= self._num_threads_to_exit:
                    self._state = _IterState.EXITED
                    break
            sleep(TICK)

    def should_exit(self) -> bool:
        """Check if the calling thread should exit.

        Returns:
            bool: Whether to exit.
        """
        with self._lock:
            return self._state in {_IterState.EXITING, _IterState.EXITED}

    def on_exit(self) -> None:
        """Note that a thread has exited."""
        with self._lock:
            self._num_exited += 1


def download_index(stream: Stream) -> None:
    """A Stream index download process.

    If the file download fails for any reason, writes an empty file to the target location, causing
    all ranks to raise an error when they attempt to load and parse the index file. Otherwise, they
    would have to wait for a timeout, unless we switch to ProcessPoolExecutor.

    Args:
        stream (Stream): The stream.
    """
    try:
        stream.download_index()
    except:
        with open(stream.local_index_path, 'wb') as out:
            out.write(b'')


class StreamingDataset(Array, IterableDataset):
    """A mid-epoch-resumable streaming/caching pytorch IterableDataset.

    Features elastically deterministic shuffling, which enables fast mid-epoch resumption.

    Checkpoints are represented in JSON as follows:
    >>> from typing import TypedDict
    >>>
    >>> class StateDict(TypedDict):
    >>>     epoch: int
    >>>     sample_in_epoch: int
    >>>     shuffle_seed: int
    >>>     num_canonical_nodes: int
    >>>     num_physical_nodes: int

    Organization of arguments:
      * What to iterate:
          * The streams (either provide streams, or remote and/or local for an implicit stream):
              * ``epoch_size``
              * ``streams``
              * ``remote``
              * ``local``
          * Stream arguments (used as defaults if explicit streams, or as args if implicit stream):
              * Filesystem:
                  * ``split``
              * Schema:
                  * ``allow_schema_mismatch``
                  * ``allow_unsafe_types``
                  * ``allow_unchecked_resumption``
              * Downloads:
                  * ``download_retry``
                  * ``download_timeout``
                  * ``download_max_size``
                  * ``validate_hash``
                  * ``keep_phases``
      * How to iterate:
          * Init:
              * ``index_download_procs``
              * ``index_download_procs_per_cpu``
              * ``index_download_max_procs``
          * Shard lifecycle:
              * ``predownload``
              * ``cache_limit``
          * Reproducibility:
              * ``shuffle_seed``
          * Sampling:
              * ``sampling_method``
              * ``sampling_granularity``
          * Partitioning:
              * ``partition_algo``
              * ``num_canonical_nodes``
              * ``batch_size``
          * Shuffling:
              * ``shuffle``
              * ``shuffle_algo``
              * ``shuffle_seed``
              * ``shuffle_block_size``
          * Batching:
              * ``batching_method``

    Args:
        epoch_size (Union[str, int], optional): Number of samples to draw per epoch balanced
            across all streams. If ``None``, takes its value from the total number of underlying
            samples. Provide this field if you are weighting streams relatively to target a larger
            or smaller epoch size. Defaults to ``None``. Can also take in human-readable number
            abbreviations (e.g., ``"100k"``, ``"64M"``, ``"77b"``, etc). Defaults to ``None``.
        streams (Sequence[Stream], optional): One or more streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path to stream the dataset from. If ``None``, dataset must
            be complete locally. Defaults to ``None``.
        local (str, optional): Local working directory to stream the dataset to. Uses a
            deterministically-calculated temp directory if not set. Defaults to ``None``.
        split (str, optional): Which dataset split sub-path to use, if any. Defaults to ``None``.
        allow_schema_mismatch (bool): If ``True``, continue if schemas mismatch across
            shards, streams, or the whole dataset. If ``False``, raises if schemas mismatch.
            Defaults to ``False``.
        allow_unsafe_types (bool): If ``True``, continue if unsafe type(s) are encountered
            in shard(s). If ``False``, raises if unsafe type(s) encountered. Defaults to ``False``.
        allow_unchecked_resumption (bool): If ``True``, upon resume, accept and use shard
            file phases that we are unable to check the size/hash(es) of. If ``False``, upon
            resume, drop such files, to regenerate on the fly when needed. Defaults to ``True``.
        download_retry (str | int): Number of download re-attempts before raising an error.
            Defaults to ``2``.
        download_timeout (str | float, optional): Time to wait for a file download to complete
            before raising an error, if any. Set to ``None`` for no limit. Streaming duration
            shorthand (e.g., ``1m23s``) is also accepted. Defaults to ``2m``.
        download_max_size (str | int, optional): Maximum size of an individual download, if any.
            This is used to prevent over-large shard files from cripping Streaming performance. Set
            to ``None`` for no size limit. Set to ``str`` to specify the limit using Streaming
            bytes shorthand. Set to ``int`` to specify bytes. Defaults to ``200mb``.
        validate_hash (str | Sequence[str], optional): Ranked list of hashing algorithms to
            apply if expected digest is available. Defaults to ``None``.
        keep_phases (str | Sequence[str] | Dict[str, Optional[bool]] | Phaser): Which phases
            to keep and to drop upon conversion, given either by intended use case or literally.
            Specified as a single use or phase to keep, a sequence of uses or phases to keep, a
            mapping of uses or phases to whether to keep or drop, or a ``Phaser`` (which performs
            the same keeping or dropping). Defaults to ``None``.
        index_download_procs (int, optional): Size of the process pool. You may set either this
            arg or ``index_download_procs_per_cpu``, but not both. Defaults to ``None``.
        index_download_procs_per_cpu (float | int, optional): Size of the process pool as the
            number of processes per CPU. You may set either this arg or ``index_download_procs``,
            but not both. If its value is negative, it is taken to be the reciprocal of that value
            as a positive number, e.g. ``-4`` means ``num_cpus // 4``. Defaults to ``None``.
        index_download_max_procs (int, optional): Optional ceiling on the number of index
            downlaod processes. Used to not overwhelm remote storage with concurrency connectinos.
            Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[str, int], optional): Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s)
            may be evicted (deleted from the local cache) in order to stay under the limit.
            Set to ``None`` to disable shard eviction. Supports integer bytes as well as string
            human-readable bytes (e.g., ``100b``, ``64kb``, ``77mb``, and so on). Defaults to
            ``None``.
        shuffle_seed (int): Seed for deterministic data shuffling. Defaults to ``9176``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``relaxed``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. The sample space is divided evenly according to the number of canonical
            nodes. The higher the value, the more independent non-overlapping paths the
            StreamingDataset replicas take through the shards per model replica (increasing data
            source diversity). If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.

            .. note::

                For sequential sample ordering, set ``shuffle`` to ``False`` and
                ``num_canonical_nodes`` to the number of physical nodes of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
        kwargs (Any): Any unsupported (for forward compat) or deprecated args.
    """

    def __init__(
        self,
        *,
        epoch_size: Optional[Union[str, int]] = None,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        allow_schema_mismatch: bool = False,
        allow_unsafe_types: bool = False,
        allow_unchecked_resumption: bool = True,
        download_retry: Union[str, int] = 2,
        download_timeout: Optional[Union[str, float]] = '2m',
        download_max_size: Optional[Union[str, int]] = '200mb',
        validate_hash: Union[None, str, Sequence[str]] = None,
        keep_phases: Union[None, str, Sequence[str], Dict[str, Optional[bool]], Phaser] = None,
        index_download_procs: Optional[int] = None,
        index_download_procs_per_cpu: Optional[Union[float, int]] = 4,
        index_download_max_procs: Optional[int] = 64,
        predownload: Optional[int] = None,
        cache_limit: Optional[Union[str, int]] = None,
        shuffle_seed: int = 9176,
        sampling_method: str = 'balanced',
        sampling_granularity: int = 1,
        partition_algo: str = 'relaxed',
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = 'py1e',
        shuffle_block_size: Optional[int] = None,
        batching_method: str = 'random',
        **kwargs: Any,
    ) -> None:
        # Global arguments (which do not live in Streams).
        self.predownload = predownload
        self.shuffle_seed = shuffle_seed
        self.sampling_method = sampling_method
        self.sampling_granularity = sampling_granularity
        self.partition_algo = partition_algo
        self.num_canonical_nodes = num_canonical_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_algo = shuffle_algo
        self.shuffle_block_size = shuffle_block_size
        self.batching_method = batching_method
        self.allow_unsafe_types = allow_unsafe_types

        # Initialize initial_physical_nodes to None. If we are resuming, then we will set it to the
        # number of physical nodes of the initial run in the _resume function.
        self.initial_physical_nodes = None

        # Check streams vs remote/local.
        if bool(streams) == (bool(remote) or bool(local)):
            raise ValueError(
                'You must provide either `streams` or `remote`/`local`, but not both.')

        # Check sampling method is one of "balanced" or "fixed".
        if self.sampling_method not in ['balanced', 'fixed']:
            raise ValueError(
                f'Invalid sampling method: {sampling_method}. ' + \
                f'Must be one of `balanced` or `fixed`.'
            )

        # Check sampling granularity.
        if self.sampling_granularity <= 0:
            raise ValueError(f'`sampling_granularity` must be a positive integer, but got: ' +
                             f'{self.sampling_granularity}.')

        # Check batching method is one of "random", "stratified", or "per_stream".
        if self.batching_method not in ['random', 'stratified', 'per_stream']:
            raise ValueError(
                f'Invalid batching method: {batching_method}. ' + \
                f'Must be one of `random`, `stratified`, or `per_stream.'
            )

        # issue deprecation warning for py1b shuffle algorithm.
        if self.shuffle_algo == 'py1b':
            warnings.warn('The \'py1b\' shuffle algorithm will soon be deprecated. \
                Please use the more performant \'py1br\' algorithm instead.',
                          DeprecationWarning,
                          stacklevel=2)

        # Check shuffle seed.
        if self.shuffle_seed < 0:
            raise ValueError(f'`shuffle_seed` must be a non-negative integer, but got: ' +
                             f'{self.shuffle_seed}.')

        # Check that predownload is at least per device batch size, and set it if currently `None`.
        if self.predownload is not None and self.batch_size is not None and \
            self.predownload < self.batch_size:
            warnings.warn(f'predownload < batch_size ({self.predownload} < {self.batch_size}).' +
                          f'This may result in slower batch time. Recommendation is to set ' +
                          f'predownload to at-least batch_size.')
        elif self.predownload is None:
            logger.warning(f'Because `predownload` was not specified, it will default to ' +
                           f'8*batch_size if batch_size is not None, otherwise 64. Prior to ' +
                           f'Streaming v0.7.0, `predownload` defaulted to ' +
                           f'max(batch_size, 256 * batch_size // num_canonical_nodes).')
            self.predownload = 8 * self.batch_size if self.batch_size is not None else 64

        # Convert epoch size from string to int, if needed. Cannot be negative.
        epoch_size_value = normalize_count(epoch_size) if epoch_size else None

        # Initialize torch dist ourselves, if necessary.
        destroy_dist = maybe_init_dist()

        # Initialize the Stream defaults and normalize to a list of Streams.
        if streams:
            for stream in streams:
                stream.apply_defaults(
                    split=split,
                    allow_schema_mismatch=allow_schema_mismatch,
                    allow_unsafe_types=allow_unsafe_types,
                    allow_unchecked_resumption=allow_unchecked_resumption,
                    download_retry=download_retry,
                    download_timeout=download_timeout,
                    download_max_size=download_max_size,
                    validate_hash=validate_hash,
                    keep_phases=keep_phases,
                    **kwargs,
                )
        else:
            streams = Stream(
                remote=remote,
                local=local,
                split=split,
                allow_schema_mismatch=allow_schema_mismatch,
                allow_unsafe_types=allow_unsafe_types,
                allow_unchecked_resumption=allow_unchecked_resumption,
                download_retry=download_retry,
                download_timeout=download_timeout,
                download_max_size=download_max_size,
                validate_hash=validate_hash,
                keep_phases=keep_phases,
                **kwargs,
            ),

        # Validate the stream weighting scheme (relative or absolute) to catch errors before we go
        # to the trouble of loading them.
        Stream.validate_weights(streams)

        # Set streams.
        self.streams = streams
        self.num_streams = len(streams)

        # Initialize the World context.
        #
        # Beware: This information is for the per-rank process. DataLoader worker processes may see
        # different values for these fields. We are saving the rank World here because we cannot
        # instantiate a World inside the StreamingDataset destructor.
        self._rank_world = world = World()

        # Download each Stream's index in parallel from local rank 0.
        #
        # Parallelism is important because there could be a very large number of Streams, and we
        # expect equal performance as them having been concatenated into one Stream.
        if world.is_local_leader:
            num_procs = self._get_index_download_procs(
                index_download_procs,
                index_download_procs_per_cpu,
                index_download_max_procs,
            )
            pool = Pool(num_procs)
            pool.imap_unordered(download_index, self.streams)
            pool.close()
        else:
            pool = None

        # All ranks then walk all Streams, for which they (1) wait for its index to become
        # downloaded, (2) load its shards, and (3) map streams <-> shards <-> samples.
        self.num_samples = 0
        self.shards = []
        stream_per_shard = []
        self.shard_offset_per_stream = np.zeros(self.num_streams, np.int64)
        self.shards_per_stream = np.zeros(self.num_streams, np.int64)
        self.sample_offset_per_stream = np.zeros(self.num_streams, np.int64)
        self.samples_per_stream = np.zeros(self.num_streams, np.int64)
        for stream_id, stream in enumerate(self.streams):
            stream.await_index()
            stream_shards = stream.load_index()
            num_stream_samples = sum(map(len, stream_shards))
            if not num_stream_samples:
                raise RuntimeError(f'Stream contains no samples: {stream.local_index_path}.')
            stream_per_shard += [stream_id] * len(stream_shards)
            self.shard_offset_per_stream[stream_id] = len(self.shards)
            self.shards_per_stream[stream_id] = len(stream_shards)
            self.sample_offset_per_stream[stream_id] = self.num_samples
            self.samples_per_stream[stream_id] = num_stream_samples
            self.shards += stream_shards
            self.num_samples += num_stream_samples
        self.stream_per_shard = np.array(stream_per_shard, np.int64)
        self.num_shards = len(self.shards)

        # Wait for the pool workers (stream index download processes) to finish.
        if pool is not None:
            pool.join()

        # Check that cache limit is possible.
        if cache_limit:
            self.cache_limit = normalize_bytes(cache_limit)
            min_cache_usage = sum((stream.index_size for stream in streams))
            if self.cache_limit <= min_cache_usage:
                raise ValueError(f'Minimum cache usage ({min_cache_usage} bytes) is larger than ' +
                                 f'the cache limit ({self.cache_limit} bytes). Please raise ' +
                                 f'`cache_limit`. Recommendation is to provide a `cache_limit` ' +
                                 f'as high as possible to avoid thrashing.')

            max_phase_size = 0
            for stream in self.streams:
                for shard in stream.shards:
                    for file in shard.files:
                        for phase in file.phases:
                            if not phase:
                                continue
                            if not phase.expected_size:
                                continue
                            if max_phase_size < phase.expected_size:
                                max_phase_size = phase.expected_size

            if self.cache_limit < 4 * max_phase_size:
                raise ValueError(f'Cache limit ({self.cache_limit} bytes) is too low. ' +
                                 f'Increase the `cache_limit` to at-least four times the ' +
                                 f'largest shard size ({max_phase_size} ' +
                                 f'bytes) which includes raw (decompressed) and zip ' +
                                 f'(compressed) file size. Recommendation is to provide a ' +
                                 f'`cache_limit` as high as possible to avoid thrashing.')
        else:
            self.cache_limit = None

        # Build the shard index (for partitioning and mapping samples to shards).
        self.samples_per_shard = np.array([shard.samples for shard in self.shards], np.int64)
        self.sample_offset_per_shard = self.samples_per_shard.cumsum() - self.samples_per_shard
        self.spanner = Spanner(self.samples_per_shard)

        # Now that we know the number of underlying samples of each stream, derive each stream's
        # true proportion/repeat/choose, as well as the total epoch size.
        self.epoch_size = Stream.apply_weights(self.streams, self.samples_per_stream,
                                               epoch_size_value, self.shuffle_seed)

        # Length (__len__) is the resampled epoch size divided over the number of devices.
        self.length = ceil(self.epoch_size / world.num_ranks)

        # Register/lookup our shared memory prefix and filelock root directory.
        stream_locals = []
        for stream in streams:
            local = os.path.join(stream.local, stream.split or '')
            local = os.path.abspath(local)
            stream_locals.append(local)

        stream_remotes = []
        for stream in streams:
            if stream.remote is not None:
                remote = os.path.join(stream.remote, stream.split or '')
            else:
                remote = None
            stream_remotes.append(remote)

        self._shm_prefix_int, self._locals_shm = get_shm_prefix(stream_locals, stream_remotes,
                                                                world)
        self._filelock_root = os.path.join(gettempdir(), 'streaming')
        os.makedirs(self._filelock_root, exist_ok=True)

        # Create the shared memory-backed barrier, without its lock, which is unpickleable.
        self._shared_barrier = SharedBarrier(
            os.path.join(self._filelock_root, _get_path(self._shm_prefix_int, BARRIER_FILELOCK)),
            _get_path(self._shm_prefix_int, BARRIER))

        # Epoch counter.
        #
        # Note: we do not assume that the end of __iter__() will ever be reached, so we need to
        # increment the epoch counter at the start of __iter__() instead of at the end, so we need
        # to track what the next epoch is, not the current epoch.
        self._next_epoch = SharedScalar(np.int64, _get_path(self._shm_prefix_int, NEXT_EPOCH))

        # Cache filelock. Protects downloading and evicting shards.
        self._cache_filelock_path = os.path.join(self._filelock_root,
                                                 _get_path(self._shm_prefix_int, CACHE_FILELOCK))
        self._cache_filelock: FileLock

        # Cache usage in bytes.
        self._cache_usage = SharedScalar(np.int64, _get_path(self._shm_prefix_int, CACHE_USAGE))

        # Shard states array. Tells if a shard is missing, downloading, or present (eviction
        # happens under the lock).
        self._shard_states = SharedArray(self.num_shards, np.uint8,
                                         _get_path(self._shm_prefix_int, SHARD_STATES))

        # Time of last access per shard. This is used to decide which shard(s) to evict when we run
        # out of space.
        self._shard_access_times = SharedArray(self.num_shards, np.uint64,
                                               _get_path(self._shm_prefix_int, SHARD_ACCESS_TIMES))

        # Initialize shared memory objects.
        if world.is_local_leader:
            # Set initial epoch (before any resumption).
            self.next_epoch = 0

            # Get cache usage due to streams.
            self.cache_usage = 0
            for stream in self.streams:
                self.cache_usage += stream.index_size

            # Get cache usage due to shards.
            cache_usage_per_shard = np.zeros(self.num_shards, np.int64)
            for stream_id, stream in enumerate(self.streams):
                begin = self.shard_offset_per_stream[stream_id]
                end = begin + self.shards_per_stream[stream_id]
                stream.inventory_local(cache_usage_per_shard[begin:end])
            self.cache_usage += cache_usage_per_shard.sum()

            # If either raw or zip are present after local dir setup, the shard is considered
            # present for download/eviction logic purposes (may need to decompress upon use).
            for shard_id, size in enumerate(cache_usage_per_shard):
                self._shard_states[shard_id] = _ShardState.LOCAL if size else _ShardState.REMOTE
                self._shard_access_times[shard_id] = time_ns()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if destroy_dist:
            dist.destroy_process_group()

        # Placeholder for a shared memory object where load_state_dict() saves its data to be
        # picked up by __iter__().
        self._resume_shm: SharedMemory

        # Placeholder for an _Iterator which tracks state during __iter__().
        self._iterator: _Iterator

        # For exception handling in __iter__ threads.
        self._executor: ThreadPoolExecutor
        self._event: Event

        del self._shared_barrier.lock  # Remote the lock that makes it unpickleable.

    def __del__(self) -> None:
        """Destructor, which releases its local working directories."""
        if hasattr(self, '_locals_shm'):
            try:
                self._locals_shm.buf[:4] = np.int32(0).tobytes()
            except:
                pass

    @property
    def size(self) -> int:
        """Get the size of the dataset in samples.

        Returns:
            int: Number of samples.
        """
        return self.num_samples

    @property
    def next_epoch(self) -> int:
        """Get the next epoch.

        Returns:
            int: Next epoch.
        """
        return int(self._next_epoch.get())

    @next_epoch.setter
    def next_epoch(self, next_epoch: int) -> None:
        """Set the next epoch.

        Args:
            next_epoch (int): Next epoch.
        """
        self._next_epoch.set(next_epoch)

    @property
    def cache_usage(self) -> int:
        """Get the cache usage.

        Returns:
            int: Cache usage in bytes.
        """
        return int(self._cache_usage.get())

    @cache_usage.setter
    def cache_usage(self, cache_usage: int) -> None:
        """Set the cache usage.

        Args:
            cache_usage (int): Cache usage in bytes.
        """
        self._cache_usage.set(cache_usage)

    def __len__(self) -> int:
        """Get the length as a PyTorch IterableDataset.

        Returns:
            int: Dataset length.
        """
        return self.length

    def _get_index_download_procs(
        self,
        index_download_procs: Optional[int] = None,
        index_download_procs_per_cpu: Optional[Union[float, int]] = 4,
        index_download_max_procs: Optional[int] = 64,
    ) -> int:
        """Get the size of the process pool used by index downloading.

        Args:
            index_download_procs (int, optional): Size of the process pool. You may set either this
                arg or ``index_download_procs_per_cpu``, but not both. Defaults to ``None``.
            index_download_procs_per_cpu (float | int, optional): Size of the process pool as the
                number of processes per CPU. You may set either this arg or
                ``index_download_procs``, but not both. If its value is negative, it is taken to be
                the reciprocal of that value as a positive number, e.g. ``-4`` means
                ``num_cpus // 4``. Defaults to ``4``.
            index_download_max_procs (int, optional): Optional ceiling on the number of index
                downlaod processes. Used to not overwhelm remote storage with concurrency
                connectinos. Defaults to ``64``.
        """
        if index_download_procs is not None:
            if index_download_procs_per_cpu is not None:
                raise ValueError(
                    f'You may specify `index_download_procs`, `index_download_procs_per_cpu`, ' +
                    f'or neither, but got both: {index_download_procs} and ' +
                    f'{index_download_procs_per_cpu} respectively.')
            else:
                num_procs = index_download_procs
                if num_procs <= 0:
                    raise ValueError(f'`index_download_procs` must be a positive integer.')
        else:
            num_cpus = os.cpu_count() or 1
            if index_download_procs_per_cpu is not None:
                procs_per_cpu = index_download_procs_per_cpu
                if procs_per_cpu < 0:
                    num_procs = num_cpus / -procs_per_cpu
                elif not procs_per_cpu:
                    num_procs = num_cpus
                else:
                    num_procs = num_cpus * procs_per_cpu
                num_procs = int(np.ceil(num_procs))
                if num_procs <= 0:
                    raise ValueError(f'`index_download_procs_per_cpu` must result in a positive ' +
                                     f'number of index download processes after rounding up.')
            else:
                num_procs = num_cpus

        max_procs = index_download_max_procs
        if max_procs is not None:
            if max_procs <= 0:
                raise ValueError(f'`index_download_max_procs` must be a positive integer, but ' +
                                 f'got: {max_procs}.')
            if max_procs < num_procs:
                num_procs = max_procs

        return num_procs

    def _set_shuffle_block_size(self, world: World) -> None:
        """Set the shuffle block size value."""
        if self.shuffle_block_size is None:
            if not world.worker_of_rank:
                logger.warning(f'Because `shuffle_block_size` was not specified, it will ' +
                               f'default to max(4_000_000 // num_canonical_nodes, 1 << 18) if ' +
                               f'num_canonical_nodes is not None, otherwise 262144. Prior to ' +
                               f'Streaming v0.7.0, `shuffle_block_size` defaulted to 262144.')
            self.shuffle_block_size = max(4_000_000 // self.num_canonical_nodes, 1 << 18) \
                if self.num_canonical_nodes is not None else 1 << 18

    def _resume(self, world: World, epoch: int) -> Tuple[int, int]:
        """Either resume from checkpoint or start at the beginning.

        Args:
            world (World): World state.
            epoch (int): What epoch we think it is (pre-checkpoint).

        Returns:
            Tuple[int, int]: What epoch this is, and sample offset in that epoch.
        """
        # Get the resume state, if it exists.
        name = _get_path(self._shm_prefix_int, RESUME)
        try:
            shm = SharedMemory(name=name, create=False)
        except FileNotFoundError:
            # There is nothing to resume.
            if not self.num_canonical_nodes:
                if self.shuffle_algo in ['py1s', 'py2s']:
                    self.num_canonical_nodes = 64 * world.num_nodes
                else:
                    if not world.worker_of_rank:
                        logger.warning(
                            f'Because `num_canonical_nodes` was not specified, and ' +
                            f'`shuffle_algo` is {self.shuffle_algo}, it will default to ' +
                            f'be equal to physical nodes. Prior to Streaming ' +
                            f'v0.7.0, `num_canonical_nodes` defaulted to 64 * physical ' +
                            f'nodes.')
                    self.num_canonical_nodes = world.num_nodes
            self._set_shuffle_block_size(world)
            return epoch, 0

        # SharedMemory buffers may contain additional null bytes at the end.
        buf = bytes(shm.buf)
        index = buf.find(b'\0')
        buf = buf[:index] if index != -1 else buf
        obj = json.loads(buf.decode('utf-8'))

        # Check if the resume state is stale.
        if obj['epoch'] < epoch:
            if not self.num_canonical_nodes:
                if self.shuffle_algo in ['py1s', 'py2s']:
                    self.num_canonical_nodes = 64 * world.num_nodes
                else:
                    if not world.worker_of_rank:
                        logger.warning(
                            f'Because `num_canonical_nodes` was not specified, and ' +
                            f'`shuffle_algo` is {self.shuffle_algo}, it will default to ' +
                            f'be equal to physical nodes. Prior to Streaming ' +
                            f'v0.7.0, `num_canonical_nodes` defaulted to 64 * physical ' +
                            f'nodes.')
                    self.num_canonical_nodes = world.num_nodes
            self._set_shuffle_block_size(world)
            return epoch, 0

        # Load the correct resumption meta data.
        epoch = obj['epoch']
        sample_in_epoch = obj['sample_in_epoch']
        self.num_canonical_nodes = obj['num_canonical_nodes']
        self.shuffle_seed = obj['shuffle_seed']
        # Ensure that we are backwards compatible with old checkpoint dataset state, since the
        # 'initial_physical_nodes' key may not be present.
        self.initial_physical_nodes = obj.get('initial_physical_nodes', None)
        self._set_shuffle_block_size(world)

        return epoch, sample_in_epoch

    def _resume_incr_epoch(self, world: World) -> Tuple[int, int]:
        """Start or resume training, pre-incrementing the next epoch.

        Args:
            world (World): World state.

        Returns:
            Tuple[int, int]: What epoch this is, and sample offset in that epoch.
        """
        # Lazily create the shared barrier's FileLock, which contains a threading Lock, which is
        # unpickleable.
        if not hasattr(self._shared_barrier, 'lock'):
            self._shared_barrier.lock = FileLock(self._shared_barrier.filelock_path)

        # Either resume from checkpoint, or start from scratch.
        presumed_epoch = self.next_epoch
        epoch, sample_in_epoch = self._resume(world, presumed_epoch)

        # Wait for everyone to get the epoch above.
        self._shared_barrier(world.workers_per_node)

        # Set the new next epoch.
        if world.is_local_leader:
            self.next_epoch = epoch + 1

        return epoch, sample_in_epoch

    def state_dict(self, num_samples: int, from_beginning: bool) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Our stock StreamingDataLoader counts samples from start of training (from_beginning=false).
        However, if you are always counting from the start of the epoch, set from_beginning=true.

        Args:
            num_samples (int): The number of samples processed so far in the current epoch.
            from_beginning (int): Whether we are counting samples from the start of this epoch, or
                the start of just this potentially resumed training run this epoch.

        Returns:
            Dict[str, Any]: The state.
        """
        world = World()
        epoch = self.next_epoch - 1
        epoch, offset = self._resume(world, epoch)
        if from_beginning:
            sample_in_epoch = num_samples
        else:
            sample_in_epoch = offset + num_samples

        # If `self.initial_physical_nodes` is None, we are running for the first time, so we set
        # initial_physical_nodes to the current number of physical nodes. Otherwise, we persist
        # initial_physical_nodes as the value loaded and set from the resumption state.
        initial_physical_nodes = world.num_nodes if self.initial_physical_nodes is None \
            else self.initial_physical_nodes

        return {
            'epoch': epoch,
            'sample_in_epoch': sample_in_epoch,
            'num_canonical_nodes': self.num_canonical_nodes,
            'shuffle_seed': self.shuffle_seed,
            'initial_physical_nodes': initial_physical_nodes,
        }

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        We just save the state to shared memory for workers to pick up when __iter__ is next
        called. We use shm because changes to this copy of the dataset wouldn't be picked up by
        persistent workers.

        Args:
            obj (Dict[str, Any]): The state.
        """
        name = _get_path(self._shm_prefix_int, RESUME)
        data = json.dumps(obj, sort_keys=True).encode('utf-8')
        # Some platforms choose to allocate chunks of memory based upon that platform's memory page
        # size, hence the exact size of the shared memory block that was returned may be larger
        # than what was requested.
        self._resume_shm = SharedMemory(name=name, size=len(data))
        self._resume_shm.buf[:len(data)] = data

    def resample_streams(
        self,
        epoch: int,
        stream_id: Optional[int] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Perform the up/down-sampling needed to generate the weighted epoch.

        Args:
            epoch (int): What epoch this is for. Used in seeding the sampling RNG.
            stream_id (Optional[int]): Which stream to resample. If ``None``, resample all streams.
                Defaults to ``None``.

        Returns:
            Tuple[NDArray[np.int64], NDArray[np.int64]]: Sampled shard sizes and sample mapping.
        """
        # Initialize random number generator and arrays. If sampling_method is "fixed", the rng
        # seed does not change, resulting in the same samples from each stream each epoch.
        rng = np.random.default_rng(self.shuffle_seed + epoch) \
            if self.sampling_method == 'balanced' \
            else np.random.default_rng(self.shuffle_seed)
        shuffle_units = []
        sample_ids = []

        resampling_streams = range(self.num_streams) if stream_id is None else [stream_id]

        # Iterate over each stream.
        for stream_id in resampling_streams:
            # stream's shard offset in list of all shards from all streams
            stream_shard_offset = self.shard_offset_per_stream[stream_id]
            num_stream_shards = self.shards_per_stream[stream_id]
            stream_shard_ids = stream_shard_offset + np.arange(num_stream_shards)

            # Calculate choose per stream shard.
            samples_per_stream_shard = self.samples_per_shard[stream_shard_ids]
            # the number of items to choose from each stream, obtained during initialization
            stream_choose = self.streams[stream_id].choose
            use_epoch = self.sampling_method == 'balanced'
            choose_per_stream_shard = get_sampling(samples_per_stream_shard, stream_choose,
                                                   self.sampling_granularity, self.shuffle_seed,
                                                   epoch, use_epoch)

            # Iterate over each shard of this stream.
            for shard_id, shard_samples, shard_choose in zip(stream_shard_ids,
                                                             samples_per_stream_shard,
                                                             choose_per_stream_shard):
                # Calculate shuffle units for this shard.
                # shuffle units are lists where each entry is a number of samples to take
                # from the shard. If upsampling a shard with 4 samples by 2.5x,
                # shard_choose will be 10, and shard_shuffle_units will be [4, 4, 2]. If
                # downsampling that same shard by 0.5x, shard_choose will be 2 and
                # shard_shuffle_units will be just [2].
                shard_shuffle_units = [shard_samples] * (shard_choose // shard_samples)
                remainder = shard_choose % shard_samples
                if remainder:
                    shard_shuffle_units.append(remainder)
                shuffle_units.append(shard_shuffle_units)

                # Calculate sample IDs of any full repeats.
                shard_sample_offset = self.sample_offset_per_shard[shard_id]
                num_full_repeats = shard_choose // shard_samples
                if num_full_repeats:
                    full_repeat = shard_sample_offset + np.arange(shard_samples)
                    sample_ids += [full_repeat] * num_full_repeats

                # Calculate sample IDs of a possible partial repeat.
                # for fixed sampling this partial repeat chooses the same
                # samples since we have fixed the rng seed.
                shortfall = shard_choose % shard_samples
                if shortfall:
                    partial_repeat = shard_sample_offset + rng.choice(
                        shard_samples, shortfall, False)
                    partial_repeat.sort()
                    sample_ids.append(partial_repeat)

        shuffle_units = np.concatenate(shuffle_units).astype(np.int64)
        sample_ids = np.concatenate(sample_ids).astype(np.int64)
        return shuffle_units, sample_ids

    def _share_work(self, sample_ids: NDArray[np.int64]) -> Tuple[SharedMemory, SharedMemory]:
        """Put an epoch's sample ordering into shared memory.

        Args:
            sample_ids (NDArray[np.int64]): Sample IDs.

        Returns:
            Tuple[SharedMemory, SharedMemory]: Shared memory arrays containing shape and data.
        """
        ndim = 5

        # Validate shape.
        if sample_ids.ndim != ndim:
            raise ValueError(f'Sample IDs must be of {ndim}D shape (num physical nodes, ' +
                             f'ranks per node, workers per rank, batches per worker, ' +
                             f'batch size). Instead, found as {sample_ids.ndim}D shape.')

        # Save the generated epoch shape to shared memory.
        name = _get_path(self._shm_prefix_int, EPOCH_SHAPE)
        size = ndim * np.int64().nbytes
        shape_shm = SharedMemory(name=name, create=True, size=size, auto_cleanup=False)
        shape_shm.buf[:size] = np.array(sample_ids.shape, np.int64).tobytes()

        # Save the generated epoch data to shared memory.
        name = _get_path(self._shm_prefix_int, EPOCH_DATA)
        size = sample_ids.size * np.int64().nbytes
        data_shm = SharedMemory(name=name, create=True, size=size, auto_cleanup=False)
        data_shm.buf[:size] = sample_ids.tobytes()

        return shape_shm, data_shm

    def _attach_work(self) -> Tuple[NDArray[np.int64], SharedMemory, SharedMemory]:
        """Get an epoch's sample ordering from shared memory.

        Returns:
            NDArray[np.int64]: Sample IDs.
        """
        ndim = 5

        # Load the generated epoch shape from shared memory.
        name = _get_path(self._shm_prefix_int, EPOCH_SHAPE)
        size = ndim * np.int64().nbytes
        shape_shm = SharedMemory(name=name, create=False, size=size, auto_cleanup=False)
        shape = tuple(np.ndarray(5, buffer=shape_shm.buf, dtype=np.int64))

        # Attach to the generated epoch data in shared memory.
        name = _get_path(self._shm_prefix_int, EPOCH_DATA)
        size = int(np.prod(shape)) * np.int64().nbytes
        data_shm = SharedMemory(name=name, create=False, size=size, auto_cleanup=False)
        sample_ids = np.ndarray(shape, buffer=data_shm.buf, dtype=np.int64)

        return sample_ids, shape_shm, data_shm

    def _get_work(self, world: World, epoch: int, sample_in_epoch: int) -> NDArray[np.int64]:
        """Get this worker's partition of this epoch's sample space.

        Args:
            world (World): World state.
            epoch (int): Which epoch it is.
            sample_in_epoch (int): Where we are in the epoch.

        Returns:
            Optional[NDArray[np.int64]]: Our partition of the epoch.
        """
        # Lazily create the shared barrier's FileLock, which contains a threading Lock, which is
        # unpickleable.
        if not hasattr(self._shared_barrier, 'lock'):
            self._shared_barrier.lock = FileLock(self._shared_barrier.filelock_path)

        # Do expensive work that may use a lot of cores/memory just once, in the local leader.
        if world.is_local_leader:
            epoch_sample_ids = generate_work(self.batching_method, self, world, epoch,
                                             sample_in_epoch)
            shape_shm, data_shm = self._share_work(epoch_sample_ids)
            self._shared_barrier(world.workers_per_node)
        else:
            self._shared_barrier(world.workers_per_node)
            epoch_sample_ids, shape_shm, data_shm = self._attach_work()

        # Each worker gets their portion of the work.
        worker_sample_ids = epoch_sample_ids[world.node, world.rank_of_node,
                                             world.worker_of_rank].flatten()

        self._shared_barrier(world.workers_per_node)

        # Now clean up after ourselves.
        shape_shm.cleanup()
        data_shm.cleanup()

        return worker_sample_ids

    def _evict_shard(self, shard_id: int) -> None:
        """Evict the given shard.

        Assumes you hold ``_cache_filelock``, preventing anyone else from modifying the cache. We
        expect that shard deletions are very fast.

        This method is called internally by ``fetch_shard`` to clear space for more downloads.

        Args:
            shard_id (int): Shard to evict.
        """
        # Delete the shard's last access time, so that it is not searchable when finding the
        # coldest shard to evict. This is done by setting the time far into the future.
        self._shard_access_times[shard_id] = NEVER

        # Set the shard state to missing.
        self._shard_states[shard_id] = _ShardState.REMOTE

        # Perform the eviction, updating cache usage to account for the removal.
        shard = self.shards[shard_id]
        self.cache_usage -= shard.evict()
        if self.cache_usage < 0:
            raise RuntimeError(f'Negative cache usage: {self.cache_usage}.')

    def _evict_coldest_shard(self) -> None:
        """Evict the coldeset (i.e., least recently accessed) shard.

        Assumes you hold ``_cache_filelock``, preventing anyone else from modifying the cache. We
        expect that shard deletions are very fast.

        This method is called internally by ``fetch_shard`` to clear space for more downloads.
        """
        while True:
            # Find the shard with the oldest last access time.
            shard_id = int(self._shard_access_times.numpy().argmin())

            # Check the shard's last access time. If it is NEVER, there are no downloaded shards to
            # evict. If any shards are currently being downloaded, wait, else raise an error.
            if self._shard_access_times[shard_id] == NEVER:
                if (self._shard_states.numpy() == _ShardState.FETCHING).any():
                    sleep(TICK)
                    continue
                else:
                    raise ValueError(
                        f'Tried to evict a shard {shard_id}, but no shards are present to evict ' +
                        f'(cache usage {self.cache_usage} of {self.cache_limit})')

            # The shard has a valid timestamp. Now, verify that it is actually present. There is an
            # edge case where it may not be present (see the note in get_item()). If not present,
            # pick the next lowest shard.
            if self._shard_states[shard_id] != _ShardState.LOCAL:
                self._shard_access_times[shard_id] = NEVER
                continue

            # Break on success.
            break

        # Evict that shard.
        self._evict_shard(shard_id)

    def evict_shard(self, shard_id: int) -> None:
        """Evict the given shard.

        This method is multithread/multiprocess-safe.

        Args:
            shard_id (int): Shard to evict.
        """
        # Lock the cache. FileLocks contain threading Locks, which are not pickleable, which is
        # incompatible with spawn, so must be created lazily.
        if not hasattr(self, CACHE_FILELOCK):
            self._cache_filelock = FileLock(self._cache_filelock_path)

        with self._cache_filelock:
            self._evict_shard(shard_id)

    def evict_coldest_shard(self) -> None:
        """Evict the coldest (i.e., least recently accessed) shard.

        This method is multithread/multiprocess-safe.
        """
        # Lock the cache. FileLocks contain threading Locks, which are not pickleable, which is
        # incompatible with spawn, so must be created lazily.
        if not hasattr(self, CACHE_FILELOCK):
            self._cache_filelock = FileLock(self._cache_filelock_path)

        with self._cache_filelock:
            self._evict_coldest_shard()

    def fetch_shard(self, shard_id: int, blocking: bool = True) -> None:
        """Download a shard, either waiting or skipping if in progress by another worker.

        This method is multithread/multiprocess-safe.

        If cache limit is enabled, this method may delete one or more other shards to make space
        for this download.

        Args:
            shard_id (int): Shard to download.
            blocking (bool): Whether to wait or skip if the shard is currently being downloaded by
                someone else. Defaults to ``True``.
        """
        # Lock the cache. FileLocks contain threading Locks, which are not pickleable, which is
        # incompatible with spawn, so must be created lazily.
        if not hasattr(self, CACHE_FILELOCK):
            self._cache_filelock = FileLock(self._cache_filelock_path)
        lock = self._cache_filelock
        lock.acquire()

        # Get the state of the shard to download.
        state = self._shard_states[shard_id]

        # Which state is it in?
        if state == _ShardState.REMOTE or state == _ShardState.LOCAL:
            # If missing, transition state to preparing.
            self._shard_states[shard_id] = _ShardState.FETCHING

            # Get the shard.
            shard = self.shards[shard_id]

            # With the above preamble done, we can release the cache lock.
            lock.release()

            # Perform the download (shard will not be modified by others in FETCHING state).
            self.cache_usage += shard.fetch()

            lock.acquire()

            # If cache_limit is enabled, we first may have to make space for the new shard.
            if self.cache_limit:
                # Evict one shard at a time until our download stays under the cache limit.
                # This means all phases of the shard due to decompression, canonicalization, etc.
                while self.cache_limit < self.cache_usage:
                    self._evict_coldest_shard()

            # Download accounting complete, note the time and transition shard state to LOCAL.
            self._shard_access_times[shard_id] = time_ns()
            self._shard_states[shard_id] = _ShardState.LOCAL
            lock.release()
        elif state == _ShardState.FETCHING:
            # Someone else is currently downloading the shard. Release the lock for others to make
            # progress.
            lock.release()

            # Do we wait on them?
            if blocking:
                # Wait for the shard to transition out of FETCHING state (to LOCAL, although
                # it would be possible for it to become evicted again before a TICK has elapsed).
                while self._shard_states[shard_id] == _ShardState.FETCHING:
                    sleep(TICK)

            # There is no need to update the last access time, because that will be set by the
            # process that downloaded the shard.
        else:
            # Unknown state.
            lock.release()
            raise RuntimeError(f'Invalid shard state: {state}')

    def get_item(self, sample_id: int, retry: int = 7) -> Any:
        """Get sample by global index, blocking to download its shard if not present.

        Args:
            sample_id (int): Sample index.
            retry (int): Maximum number of times to download its shard before giving up. In the
                edge case of a shard being evicted before sample access, you will have to
                re-download it. Defaults to ``7``.

        Returns:
            Dict[str, Any]: Mapping of column name to column data.
        """
        # Background thread crashed, terminate the main process
        if hasattr(self, '_event') and self._event.is_set():
            raise RuntimeError('Background thread failed. Check other traceback.')
        # Locate the shard and sample offset within that shard where the sample lives.
        shard_id, shard_sample_id = self.spanner[sample_id]
        shard = self.shards[shard_id]

        sample = None
        errors = []
        for _ in range(1 + retry):
            try:
                # Shortcut path: just assume the shard is present. Using exceptions as control flow
                # is actually faster than checking that the shard is present because python.
                sample = shard[shard_sample_id]

                # Manually update the last access time afterward. This also happens at the end of
                # shard.fetch().
                #
                # Note: for performance reasons, we have not taken the lock here. This results in
                # an edge case where a shard has a last access time but is actually not LOCAL.
                # This impacts _evict_coldest_shard(), which we modify to handle this case.
                self._shard_access_times[shard_id] = time_ns()

                # On success, break out.
                break
            except FileNotFoundError as e:
                # Fallback: shard file is missing (generates `FileNotFoundError` exception),
                # ensure the shard file is downloaded, then try to access the sample again.
                # Loops because it may become evicted in the meantime.
                errors.append(str(e))
                self.fetch_shard(shard_id)
        else:
            # Main process failed. Let the threads know to terminate.
            if hasattr(self, '_event'):
                self._event.set()
            if self.cache_limit:
                raise RuntimeError(f'{errors[-1]}. StreamingDataset repeatedly failed to ' +
                                   f'download a shard. This may be due to thrashing caused by ' +
                                   f'`cache_limit` being set too low.')
            else:
                raise RuntimeError(f'{errors[-1]}. Check if the shard file exists in your ' +
                                   f'remote location or have you deleted the shard file from ' +
                                   f'the local directory?')

        return sample

    def on_exception(self, future: Future) -> None:
        """Raise an exception to the caller if an exception was generated by a thread.

        Also, set the thread event to let the other threads know about the exception.

        Args:
            future (Future): The status of the task.

        Raises:
            Exception: re-raises the exception.
        """
        exception = future.exception()
        if exception:
            # Set the event to let the other threadpool threads know about the exception.
            self._event.set()
            # Re-raise the exception.
            raise exception

    def _fetch_thread(self, it: _Iterator) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling exit() on its state (only one epoch is valid at a
        time).

        Each worker has its own download thread, which iterates ahead of the ready thread and yield
        loop.

        Args:
            it (_Iterator): State of __iter__.
        """
        # Download loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if it.should_exit():
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if it.fetch_index == it.total:
                break

            # Background thread or a main process crashed, terminate this thread.
            if self._event.is_set():
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = it.fetch_index - it.yield_index
                if self.predownload < samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = it.sample_ids[it.fetch_index]
            if sample_id == -1:
                it.fetch_index += 1
                continue

            # Download and decompress the shard for this sample, if not already done.
            shard_id, _ = self.spanner[sample_id]
            self.fetch_shard(shard_id, False)

            # Step forward one sample.
            it.fetch_index += 1

        # Note that we exited.
        it.on_exit()

    def _ready_thread(self, it: _Iterator) -> None:
        """Wait for the relevant shards to become downloaded while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling exit() on its state (only one epoch is valid at a
        time).

        Each worker has its own ready thread, which iterates behind the download thread and ahead
        of the yield loop.

        Args:
            it (_Iterator): State of __iter__.
        """
        # Ready loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if it.should_exit():
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if it.ready_index == it.total:
                break

            # Background thread or a main process crashed, terminate this thread.
            if self._event.is_set():
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = it.ready_index - it.yield_index
                if self.predownload < samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = it.sample_ids[it.ready_index]
            if sample_id == -1:
                it.ready_index += 1
                continue

            # Wait for the shard for this sample to be downloaded and decompressed, if not already.
            shard_id, _ = self.spanner[sample_id]
            # During cold shard eviction, shard state might go in the reverse direction. If a shard
            # is missing while fetching a sample, download it.
            if self._shard_states[shard_id] == _ShardState.REMOTE:
                self.fetch_shard(shard_id, False)
            # Wait for a shard file to download completely.
            while self._shard_states[shard_id] != _ShardState.LOCAL:
                # Background thread or a main process crashed, terminate this thread.
                if self._event.is_set():
                    break
                sleep(TICK)

            # Step forward one sample.
            it.ready_index += 1

        # Note that we exited.
        it.on_exit()

    def _each_sample_id(self, it: _Iterator) -> Iterator[int]:
        """Iterate over our samples while waiting for them to download first.

        This method is entered at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling exit() on its state (only one epoch is valid at a
        time).

        Each worker has its own yield loop, which iterates behind the download and ready threads.

        Args:
            it (_Iterator): State of __iter__.

        Returns:
            Iterator[int]: Each sample, having been downloaded.
        """
        # Yield loop.
        while True:
            # If we've started a new epoch before this one is finished, exit this thread.
            if it.should_exit():
                break

            # Have we yielded all our samples?
            if it.yield_index == it.total:
                break

            # Background thread crashed, terminate the main process
            if hasattr(self, '_event') and self._event.is_set():
                break

            # Is there a sample ready to yield?
            if it.ready_index <= it.yield_index:
                sleep(TICK)
                continue

            # Yield sample ID if not -1.
            sample_id = it.sample_ids[it.yield_index]
            if sample_id != -1:
                yield sample_id

            # Step forward one sample.
            it.yield_index += 1

        # Note that we exited.
        it.on_exit()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        # Exit the threads that are pre-downloading and iterating the shards for previous epoch, if
        # it exists.
        if hasattr(self, '_iterator'):
            self._iterator.exit()

        # For exception handling.
        if not hasattr(self, '_executor'):
            self._executor = ThreadPoolExecutor()
        if not hasattr(self, '_event'):
            self._event = Event()
        elif self._event.is_set():
            raise RuntimeError('Background thread failed. Check other traceback.')

        # Discover where we left off, if there is a checkpoint, or start at the next epoch.
        # Also pre-increment the epoch counter.
        world = World()
        epoch, sample_in_epoch = self._resume_incr_epoch(world)

        # Get this worker's partition of samples to process.
        sample_ids = self._get_work(world, epoch, sample_in_epoch)
        if not len(sample_ids):  # Resumed at end of epoch, out of samples.
            return

        # Iterate over the samples while downloading ahead.
        self._iterator = it = _Iterator(sample_ids)
        fetch_future = self._executor.submit(self._fetch_thread, it)
        fetch_future.add_done_callback(self.on_exception)
        ready_future = self._executor.submit(self._ready_thread, it)
        ready_future.add_done_callback(self.on_exception)
        yield from map(self.__getitem__, self._each_sample_id(it))
        wait([fetch_future, ready_future], return_when='FIRST_EXCEPTION')
        it.exit()
