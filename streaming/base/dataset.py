# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A mid-epoch-resumable streaming/caching pytorch IterableDataset."""

import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures._base import Future
from enum import IntEnum
from math import ceil
from threading import Event, Lock
from time import sleep, time_ns
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray
from torch import distributed as dist
from torch.utils.data import IterableDataset

from streaming.base.array import Array
from streaming.base.constant import (BARRIER, BARRIER_FILELOCK, CACHE_FILELOCK, CACHE_USAGE,
                                     EPOCH_DATA, EPOCH_SHAPE, NEXT_EPOCH, RESUME,
                                     SHARD_ACCESS_TIMES, SHARD_STATES, TICK)
from streaming.base.distributed import maybe_init_dist
from streaming.base.format import get_index_basename
from streaming.base.partition import get_partitions
from streaming.base.shared import (SharedArray, SharedBarrier, SharedMemory, SharedScalar,
                                   _get_path, get_shm_prefix)
from streaming.base.shuffle import get_shuffle
from streaming.base.spanner import Spanner
from streaming.base.stream import Stream
from streaming.base.util import bytes_to_int, number_abbrev_to_int
from streaming.base.world import World

# An arbitrary time in the future, used for cold shard eviction.
NEVER = np.iinfo(np.uint64).max

logger = logging.getLogger(__name__)


class _ShardState(IntEnum):
    """The download status of a shard.

    Restrictions:
    - The initial state of INVALID must be zero.
    - State transitions: REMOTE -> PREPARING -> LOCAL -> REMOTE.
    """
    INVALID = 0  # The state is allocated (e.g., in an array), but not initialized yet.
    REMOTE = 1  # The shard exists only at the remote source.
    PREPARING = 2  # The shard is currently being worked on: (a) downloading from remote to local,
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
        self.prepare_index = 0
        self.ready_index = 0
        self.yield_index = 0
        self.eviction_index = 0

        self._lock = Lock()
        self._state = 0
        self._num_exited = 0

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


class StreamingDataset(Array, IterableDataset):
    """A mid-epoch-resumable streaming/caching pytorch IterableDataset.

    Features elastically deterministic shuffling, which enables fast mid-epoch resumption.

    Checkpoints are represented in JSON as follows:

    .. code-block:: json

        {
            "epoch" :"int",
            "sample_in_epoch": "int",
            "shuffle_seed": "int",
            "num_canonical_nodes": "int"
        }

    StreamingDataset init takes two kinds of arguments:

    * What to iterate:

      * One or more streams (you must provide either ``streams`` or ``remote``/``local``):

        * ``streams``
        * ``remote``
        * ``local``

      * Knobs to control streaming behavior, which, if multiple streams are provided, become defaults
        applied to each of them:

        * ``split``
        * ``download_retry``
        * ``download_timeout``
        * ``validate_hash``
        * ``keep_zip``

      * Absolute dataset size, if streams were weighted relatively:

        * ``epoch_size``

    * How to iterate:

      * Shard lifecycle:

        * ``predownload``
        * ``cache_limit``

      * Determinism:

        * ``partition_algo``
        * ``num_canonical_nodes``
        * ``batch_size``

      * Shuffling:

        * ``shuffle``
        * ``shuffle_algo``
        * ``shuffle_seed``
        * ``shuffle_block_size``

      * Sampling:

        * ``sampling_method``


    Args:
        streams (Sequence[Stream], optional): One or more streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            ``False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``. Can also take in human-readable number
            abbreviations (e.g., ``"100k"``, ``"64M"``, ``"77b"``, and so on). Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value gets derived using per device batch size and number of
            canonical nodes ``max(batch_size, 256 * batch_size // num_canonical_nodes)``.
            Defaults to ``None``.
        cache_limit (Union[int, str], optional): Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s)
            may be evicted (deleted from the local cache) in order to stay under the limit.
            Set to ``None`` to disable shard eviction. Supports integer bytes as well as string
            human-readable bytes (e.g., ``100b``, ``64kb``, ``77mb``, and so on). Defaults to
            ``None``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. The sample space is divided evenly according to the number of canonical
            nodes. The higher the value, the more independent non-overlapping paths the
            StreamingDataset replicas take through the shards per model replica (increasing data
            source diversity). Defaults to ``None``, which is interpreted as 64 times the number
            of nodes of the initial run.

            .. note::

                For sequential sample ordering, set ``shuffle`` to ``False`` and
                ``num_canonical_nodes`` to the number of physical nodes of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. A canonical node's samples are split into blocks
            of this size, and samples within each block are shuffled. Defaults to ``1 << 18``.
        sampling_method (str): Which sampling method to use, ``balanced``, ``fixed``, or ``uniform_batch``.
            Defaults to ``balanced``.
    """

    def __init__(self,
                 *,
                 streams: Optional[Sequence[Stream]] = None,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[Union[int, str]] = None,
                 predownload: Optional[int] = None,
                 cache_limit: Optional[Union[int, str]] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 sampling_method: str = 'balanced') -> None:
        # Global arguments (which do not live in Streams).
        self.predownload = predownload
        self.cache_limit = cache_limit
        self.partition_algo = partition_algo
        self.num_canonical_nodes = num_canonical_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_algo = shuffle_algo
        self.shuffle_seed = shuffle_seed
        self.shuffle_block_size = shuffle_block_size
        self.sampling_method = sampling_method.lower().strip()

        # Check streams vs remote/local.
        if bool(streams) == (bool(remote) or bool(local)):
            raise ValueError(
                'You must provide either `streams` or `remote`/`local`, but not both.')

        # Check sampling method is one of "balanced" or "fixed".
        if self.sampling_method not in ['balanced', 'fixed', 'uniform_batch']:
            raise ValueError(
                f'Invalid sampling method: {sampling_method}. Must be one of `balanced`, `fixed`, or `uniform_batch`.'
            )

        # issue deprecation warning for py1b shuffle algorithm.
        if self.shuffle_algo == 'py1b':
            warnings.warn(
                'The \'py1b\' shuffle algorithm will soon be deprecated. Please use the more performant \'py1br\' algorithm instead.',
                DeprecationWarning,
                stacklevel=2)

        # Check that predownload is at least per device batch size.
        if self.predownload is not None and self.batch_size is not None and \
            self.predownload < self.batch_size:
            warnings.warn(f'predownload < batch_size ({self.predownload} < {self.batch_size}).' +
                          f'This may result in slower batch time. Recommendation is to set ' +
                          f'predownload to at-least batch_size.')
        # Convert epoch size from string to int, if needed. Cannot be negative.
        epoch_size_value = None
        if epoch_size:
            epoch_size_value = number_abbrev_to_int(epoch_size)
            if epoch_size_value < 0:
                raise ValueError(f'Epoch size cannot be negative. Received {epoch_size_value}.')

        # Initialize torch dist ourselves, if necessary.
        destroy_dist = maybe_init_dist()

        # Initialize the Stream defaults and normalize to a list of Streams.
        if streams:
            default = {
                'remote': remote,
                'local': local,
                'split': split,
                'download_retry': download_retry,
                'download_timeout': download_timeout,
                'validate_hash': validate_hash,
                'keep_zip': keep_zip,
            }
            for stream in streams:
                stream.apply_default(default)
        else:
            default = Stream(remote=remote,
                             local=local,
                             split=split,
                             download_retry=download_retry,
                             download_timeout=download_timeout,
                             validate_hash=validate_hash,
                             keep_zip=keep_zip)
            streams = [default]

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

        # Download each stream's index, load their shards, and map streams <-> shards.
        self.num_samples = 0
        self.shards = []
        stream_per_shard = []
        self.shard_offset_per_stream = np.zeros(self.num_streams, np.int64)
        self.shards_per_stream = np.zeros(self.num_streams, np.int64)
        self.sample_offset_per_stream = np.zeros(self.num_streams, np.int64)
        self.samples_per_stream = np.zeros(self.num_streams, np.int64)
        for stream_id, stream in enumerate(self.streams):
            stream_shards = stream.get_shards(world)
            num_stream_samples = sum(map(len, stream_shards))
            if not num_stream_samples:
                index_filename = os.path.join(stream.local, stream.split, get_index_basename())
                raise RuntimeError(f'Stream contains no samples: {index_filename}.')
            stream_per_shard += [stream_id] * len(stream_shards)
            self.shard_offset_per_stream[stream_id] = len(self.shards)
            self.shards_per_stream[stream_id] = len(stream_shards)
            self.sample_offset_per_stream[stream_id] = self.num_samples
            self.samples_per_stream[stream_id] = num_stream_samples
            self.shards += stream_shards
            self.num_samples += num_stream_samples
        self.stream_per_shard = np.array(stream_per_shard, np.int64)
        self.num_shards = len(self.shards)

        # Check that cache limit is possible.
        if self.cache_limit:
            if isinstance(self.cache_limit, str):
                self.cache_limit = bytes_to_int(self.cache_limit)
            min_cache_usage = sum((stream.get_index_size() for stream in streams))
            if self.cache_limit <= min_cache_usage:
                raise ValueError(f'Minimum cache usage ({min_cache_usage} bytes) is larger than ' +
                                 f'the cache limit ({self.cache_limit} bytes). Please raise ' +
                                 f'`cache_limit`.')

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
        my_locals = [os.path.abspath(os.path.join(x.local, x.split)) for x in streams]
        self._shm_prefix_int, self._locals_shm = get_shm_prefix(my_locals, world)
        self._filelock_root = os.path.join(os.path.sep, 'tmp', 'streaming')
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
                self.cache_usage += stream.get_index_size()

            # Get cache usage due to shards.
            cache_usage_per_shard = np.zeros(self.num_shards, np.int64)
            for stream_id, stream in enumerate(self.streams):
                begin = self.shard_offset_per_stream[stream_id]
                end = begin + self.shards_per_stream[stream_id]
                stream.set_up_local(self.shards[begin:end], cache_usage_per_shard[begin:end])
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

    def _set_predownload(self) -> None:
        """Set the predownload value which is per number of workers."""
        if self.predownload is None:
            self.predownload = max(
                self.batch_size, 256 * self.batch_size // self.num_canonical_nodes
            ) if self.batch_size is not None and self.num_canonical_nodes is not None else 512

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
                self.num_canonical_nodes = world.num_nodes * 64
            self._set_predownload()
            return epoch, 0

        # SharedMemory buffers may contain additional null bytes at the end.
        buf = bytes(shm.buf)
        index = buf.find(b'\0')
        buf = buf[:index] if index != -1 else buf
        obj = json.loads(buf.decode('utf-8'))

        # Check if the resume state is stale.
        if obj['epoch'] < epoch:
            if not self.num_canonical_nodes:
                self.num_canonical_nodes = world.num_nodes * 64
            self._set_predownload()
            return epoch, 0

        # Load the correct resumption meta data.
        epoch = obj['epoch']
        sample_in_epoch = obj['sample_in_epoch']
        self.num_canonical_nodes = obj['num_canonical_nodes']
        self.shuffle_seed = obj['shuffle_seed']
        self._set_predownload()

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
        return {
            'epoch': epoch,
            'sample_in_epoch': sample_in_epoch,
            'num_canonical_nodes': self.num_canonical_nodes,
            'shuffle_seed': self.shuffle_seed
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

    def _resample_streams(
            self,
            epoch: int,
            stream_id: Optional[int] = None) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Perform the up/down-sampling needed to generate the weighted epoch.

        Args:
            epoch (int): What epoch this is for. Used in seeding the sampling RNG.
            stream_id (Optional[int]): Which stream to resample. If ``None``, resample all streams. Defaults to ``None``.

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
            stream_samples = sum(samples_per_stream_shard)
            # the number of items to choose from each stream (calculated during dataset initialization)
            stream_choose = self.streams[stream_id].choose
            if stream_choose == stream_samples:
                choose_per_stream_shard = samples_per_stream_shard
            else:
                choose_per_stream_shard = \
                    samples_per_stream_shard * stream_choose // stream_samples
                shortfall = stream_choose - choose_per_stream_shard.sum()
                indices = rng.choice(num_stream_shards, shortfall, False)
                choose_per_stream_shard[indices] += 1

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

    def _generate_work(self, world: World, epoch: int, sample_in_epoch: int) -> NDArray[np.int64]:
        """Generate this epoch's arrangement of samples.

        This is only called in local rank zero.

        Args:
            world (World): World state.
            epoch (int): Which epoch it is.
            sample_in_epoch (int): Where we are in the epoch.

        Returns:
            NDArray[np.int64]: The epoch (num physical nodes, ranks per node, workers per rank,
                batches per worker, batch size).
        """
        # Ensure that num_canonical_nodes has been set.
        if self.num_canonical_nodes is None:
            raise RuntimeError(f'`num_canonical_nodes` can never be None. ' +
                               f'Provide a positive integer.')

        if self.sampling_method == 'uniform_batch':
            # For the `uniform_batch` sampling method, we ensure that each batch consists of samples
            # selected only from one particular stream. Otherwise, samples in a batch can come from
            # many streams.

            # First, for each stream, sample each shard of the stream according to proportions/repeats/samples.
            # We obtain the resampled size of each shard in the stream and a mapping from the training "big" sample ID
            # to the underlying shard "small" sample ID.
            # Then, we also partition each stream's samples over nodes/devices/workers.
            # We handle sample_in_epoch (for resumption) at the end.
            shuffle_units_per_stream = []
            small_per_big_per_stream = []
            partition_per_stream = []
            stream_sample_offset = 0

            batch_size = self.batch_size or 1

            for stream_id in range(self.num_streams):
                shuffle_units, small_per_big = self._resample_streams(epoch, stream_id)
                shuffle_units_per_stream.append(shuffle_units)
                small_per_big_per_stream.append(small_per_big)
                samples_in_stream = len(small_per_big)
                stream_partition = get_partitions(self.partition_algo, samples_in_stream,
                                                  self.num_canonical_nodes, world.num_nodes,
                                                  world.ranks_per_node, world.workers_per_rank,
                                                  batch_size, 0)
                if self.shuffle:
                    stream_shuffle = get_shuffle(self.shuffle_algo, shuffle_units,
                                                 self.num_canonical_nodes, self.shuffle_seed,
                                                 epoch, self.shuffle_block_size)
                    stream_partition = np.where(stream_partition != -1,
                                                stream_shuffle[stream_partition], -1)
                # The small_per_big array already corresponds to indices of samples per shard of each stream.
                # So each sample ID in the stream's partition already corresponds to the sample ID in the right shard.
                partition_per_stream.append(
                    np.where(stream_partition != -1, small_per_big[stream_partition], -1))
                stream_sample_offset += samples_in_stream

            # We now merge the partitions from each stream to get our final partition over all streams, where
            # each global batch has samples only from a single stream.
            # Partitions are arranged (physical nodes, ranks per node, workers per rank, batches per worker, batch size).
            batches_per_stream = []
            batches_from_partitions = []
            for i, partition in enumerate(partition_per_stream):
                # reshape the partition to be global batches in order of traversal, and count only batches without -1 in them.
                global_batches_inorder = partition.transpose(3, 2, 0, 1, 4).reshape(
                    -1, batch_size * world.ranks_per_node * world.num_nodes)
                num_full_batches = np.count_nonzero(np.min(global_batches_inorder, axis=1) >= 0)
                batches_per_stream.append(num_full_batches)
                if num_full_batches != global_batches_inorder.shape[0]:
                    logger.warning(
                        'Because of the `uniform_batch` sampling method, some batches with an inadequate number of samples '
                        + 'from stream with index ' + str(i) + ' are being dropped.')
                batches_from_partitions.append(global_batches_inorder[:num_full_batches])

            # Combine all global batches from all streams into one array.
            all_partition_batches = np.concatenate(batches_from_partitions)

            # Shuffle seed changes with every epoch so that the order of streams in our batches changes as well.
            epoch_rng = np.random.default_rng(self.shuffle_seed + epoch)

            # stream_origins is an array that tells us which stream each batch is using.
            stream_origins = np.concatenate(
                [np.full(n_batch, i) for i, n_batch in enumerate(batches_per_stream)])
            epoch_rng.shuffle(stream_origins)

            # Now, we want the batch_indices array to correctly index into the all_partition_batches
            # array according to stream_origins in order to get our final batch order.
            # For each stream, we want to traverse its batches in the same order as given in its partition.
            batch_indices = np.zeros(stream_origins.shape[0]).astype(np.int64)
            batch_offset = 0
            for i, n_batch in enumerate(batches_per_stream):
                # Update batch_indices for the one stream at a time.
                batch_indices[stream_origins == i] += batch_offset + np.arange(n_batch)
                batch_offset += n_batch

            # Rearrange all_partition_batches by the batch_indices we have obtained.
            all_partition_batches = all_partition_batches[batch_indices]

            # If applicable we resume right after the most recently used full global batch.
            global_batch_size = batch_size * world.num_nodes * world.ranks_per_node
            if sample_in_epoch % global_batch_size != 0:
                logger.warning(
                    'Because of the `uniform_batch` sampling method, resumption may only occur on a sample that '
                    + 'is a multiple of the current global batch size of ' +
                    str(global_batch_size) + '. Resuming training ' +
                    'after the most recently finished global batch.')

            # Discard previous batches that may have already finished
            resumption_batch = sample_in_epoch // global_batch_size
            all_partition_batches = all_partition_batches[resumption_batch:]

            # Reverse the transposition and reshape from earlier.
            # Final result is (physical nodes, ranks per node, workers per rank, batches per worker, batch size), as desired.
            return all_partition_batches.reshape(-1, world.workers_per_rank, world.num_nodes,
                                                 world.ranks_per_node,
                                                 batch_size).transpose(2, 3, 1, 0, 4)

        else:
            # Sample each shard of each stream according to their proportions/repeats/samples. This
            # gives us the resampled size of each underlying shard, and a mapping from each fake "big"
            # sample ID to its underlying "small" sample ID.
            shuffle_units, small_per_big = self._resample_streams(epoch)

            # Partition the global sample space (of resampled "big" sample IDs) into a tensor of shape
            # (num physical nodes, ranks per node, workers per rank, batches per worker, samples per
            # batch) such that we have an elastically deterministic sample order.
            big_ids = get_partitions(self.partition_algo, self.epoch_size,
                                     self.num_canonical_nodes, world.num_nodes,
                                     world.ranks_per_node, world.workers_per_rank, self.batch_size,
                                     sample_in_epoch)

            # If we need to shuffle, shuffle in a node-aware and *underlying* shard-aware way.
            if self.shuffle:
                shuffle = get_shuffle(self.shuffle_algo, shuffle_units, self.num_canonical_nodes,
                                      self.shuffle_seed, epoch, self.shuffle_block_size)
                big_ids = np.where(big_ids != -1, shuffle[big_ids], -1)

            # Now that we have partitioning and shuffled with hallucinated "big" sample IDs, we don't
            # need them anymore, and can convert back to underlying "small" sample IDs.
            return np.where(big_ids != -1, small_per_big[big_ids], -1)

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
            epoch_sample_ids = self._generate_work(world, epoch, sample_in_epoch)
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

        This method is called internally by ``prepare_shard`` to clear space for more downloads.

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

        Assumes you hold ``__cache_filelock``, preventing anyone else from modifying the cache. We
        expect that shard deletions are very fast.

        This method is called internally by ``prepare_shard`` to clear space for more downloads.
        """
        while True:
            # Find the shard with the oldest last access time.
            shard_id = int(self._shard_access_times.numpy().argmin())

            # Check the shard's last access time. If it is NEVER, there are no downloaded shards to
            # evict. If any shards are currently being downloaded, wait, else raise an error.
            if self._shard_access_times[shard_id] == NEVER:
                if (self._shard_states.numpy() == _ShardState.PREPARING).any():
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

    def prepare_shard(self, shard_id: int, blocking: bool = True) -> None:
        """Download a shard, either waiting or skipping if in progress by another worker.

        This method is multithread/multiprocess-safe.

        If cache limit is enabled, this method may delete one or more other shards to make space
        for this download.

        Args:
            shard_id (int): Shard to download.
            blocking (bool): Whether to wait or skip if the shard is currently being downloaded by
                someone else.
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
        if state == _ShardState.REMOTE:
            # If missing, transition state to preparing.
            self._shard_states[shard_id] = _ShardState.PREPARING

            # Get the stream and shard.
            stream_id = self.stream_per_shard[shard_id]
            stream = self.streams[stream_id]
            shard = self.shards[shard_id]

            # If cache_limit is enabled, we first may have to make space for the new shard.
            if self.cache_limit:
                # Evict one shard at a time until our download will stay under the cache limit.
                # This means both the raw and zip forms of the shard due to decompressing.
                shard_max_cache_usage = shard.get_max_size()
                while self.cache_limit < self.cache_usage + shard_max_cache_usage:
                    self._evict_coldest_shard()

            # With the above preamble done, we can release the cache lock.
            lock.release()

            # Perform the download (shard will not be modified by others in PREPARING state).
            delta = stream.prepare_shard(shard)

            # Download completed, so note the time and transition shard state to LOCAL.
            lock.acquire()
            self.cache_usage += delta
            self._shard_access_times[shard_id] = time_ns()
            self._shard_states[shard_id] = _ShardState.LOCAL
            lock.release()
        elif state == _ShardState.PREPARING:
            # Someone else is currently downloading the shard. Release the lock for others to make
            # progress.
            lock.release()

            # Do we wait on them?
            if blocking:
                # Wait for the shard to transition out of PREPARING state (to LOCAL, although it would
                # be possible for it to become evicted again before a TICK has elapsed).
                while self._shard_states[shard_id] == _ShardState.PREPARING:
                    sleep(TICK)

            # There is no need to update the last access time, because that will be set by the
            # process that downloaded the shard.
        elif state == _ShardState.LOCAL:
            # Get the stream and shard.
            stream_id = self.stream_per_shard[shard_id]
            stream = self.streams[stream_id]
            shard = self.shards[shard_id]

            # We may need to decompress the shard (if local dir just contains zips).
            raw_info, _ = shard.file_pairs[0]  # Each file pair is present in the same way.
            raw_filename = os.path.join(stream.local, stream.split, raw_info.basename)  # Find raw.
            if not os.path.isfile(raw_filename):  # Is raw missing?
                self._shard_states[shard_id] = _ShardState.PREPARING  # Lock the shard.
                lock.release()  # Unblock other workers.
                delta = stream.prepare_shard(shard)  # Decompress and remove zip.
                lock.acquire()  # Briefly take the lock back.
                self._shard_states[shard_id] = _ShardState.LOCAL  # Restore shard state.
                self.cache_usage += delta  # Update accounting.
            self._shard_access_times[shard_id] = time_ns()  # Touch the shard.
            lock.release()
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
                # prepare_shard().
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
                self.prepare_shard(shard_id)
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

    def _prepare_thread(self, it: _Iterator) -> None:
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
            if it.prepare_index == it.total:
                break

            # Background thread or a main process crashed, terminate this thread.
            if self._event.is_set():
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = it.prepare_index - it.yield_index
                if self.predownload < samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = it.sample_ids[it.prepare_index]
            if sample_id == -1:
                it.prepare_index += 1
                continue

            # Download and decompress the shard for this sample, if not already done.
            shard_id, _ = self.spanner[sample_id]
            self.prepare_shard(shard_id, False)

            # Step forward one sample.
            it.prepare_index += 1

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
                self.prepare_shard(shard_id, False)
            # Wait for a shard file to download completely.
            while self._shard_states[shard_id] != _ShardState.LOCAL:
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
        prepare_future = self._executor.submit(self._prepare_thread, it)
        prepare_future.add_done_callback(self.on_exception)
        ready_future = self._executor.submit(self._ready_thread, it)
        ready_future.add_done_callback(self.on_exception)
        yield from map(self.__getitem__, self._each_sample_id(it))
        wait([prepare_future, ready_future])
        it.exit()
