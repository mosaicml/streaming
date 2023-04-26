# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A mid-epoch-resumable streaming/caching pytorch IterableDataset."""

import json
import os
from enum import IntEnum
from math import ceil
from threading import Lock, Thread
from time import sleep, time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from streaming.base.format import get_index_basename
from streaming.base.partition import get_partitions
from streaming.base.shared import (SharedArray, SharedBarrier, SharedMemory, SharedScalar,
                                   get_shm_prefix)
from streaming.base.shuffle import get_shuffle
from streaming.base.spanner import Spanner
from streaming.base.stream import Stream
from streaming.base.util import TICK
from streaming.base.world import World

# An arbitrary time in the future, used for cold shard eviction.
NEVER = float(np.iinfo(np.uint64).max)


class _ShardState(IntEnum):
    """The download status of a shard.

    Restrictions:
    - The initial state of INVALID must be zero.
    - State transitions: MISSING -> DOWNLOADING -> PRESENT -> MISSING.
    """
    INVALID = 0
    MISSING = 1
    DOWNLOADING = 2
    PRESENT = 3


class _IterState:
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

    # The number of threads to wait on the exits of before returning (download, ready, yield).
    _num_threads_to_exit = 3

    def __init__(self, sample_ids: NDArray[np.int64]) -> None:
        self.sample_ids = sample_ids

        self.total = len(sample_ids)
        self.download_index = 0
        self.ready_index = 0
        self.yield_index = 0
        self.eviction_index = 0

        self._lock = Lock()
        self._is_exiting = False
        self._num_exited = 0

    def exit(self) -> None:
        """Signal threads to exit, wait until they have all exited, then return.

        This is called when the user starts a new epoch without the threads from the previous epoch
        having exited yet.
        """
        # Signal threads to exit.
        with self._lock:
            if self._is_exiting:
                raise ValueError('Called exit() on an _IterState that is already exiting.')
            self._is_exiting = True

        # Block until they have all exited, returning _is_exiting to False.
        while True:
            with self._lock:
                if self._num_exited == self._num_threads_to_exit:
                    self._is_exiting = False
                    break
            sleep(TICK)

    def is_exiting(self) -> bool:
        """Check if the calling thread should exit.

        Returns:
            bool: Whether to exit.
        """
        with self._lock:
            return self._is_exiting

    def on_exit(self) -> None:
        """Note that a thread has exited."""
        with self._lock:
            self._num_exited += 1


class StreamingDataset(IterableDataset):
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

        * ``choose``

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
        choose (int, optional): Number of samples to draw per epoch balanced across all streams.
            If ``None``, takes its value from the total number of underlying samples. Provide this
            field if you are weighting streams relatively to target a larger or smaller epoch size.
            Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        cache_limit (int, optional): Maximum size in bytes of this StreamingDataset's shard cache.
            Before downloading a shard, the least recently used resident shard(s) may be evicted
            (deleted from the local cache) in order to stay under the limit. Set to ``None`` to
            disable shard eviction. Defaults to ``None``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1b``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
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
                 choose: Optional[int] = None,
                 predownload: Optional[int] = 100_000,
                 cache_limit: Optional[int] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1b',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18) -> None:
        # Global arguments (which do not live in Streams).
        self.cache_limit = cache_limit
        self.predownload = predownload
        self.partition_algo = partition_algo
        self.num_canonical_nodes = num_canonical_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_algo = shuffle_algo
        self.shuffle_seed = shuffle_seed
        self.shuffle_block_size = shuffle_block_size

        # Check streams vs remote/local.
        if bool(streams) == (bool(remote) or bool(local)):
            raise ValueError(
                'You must provide either "streams" or "remote"/"local", but not both -- ' +
                'that would be confusing')

        # Initialize the Stream defaults.
        default = Stream(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip)

        # Normalize to a list of Streams.
        if streams:
            for stream in streams:
                stream.apply_default(default)
        else:
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
            samples = sum(map(len, stream_shards))
            if samples == 0:
                index_path = os.path.join(stream.local, stream.split, get_index_basename())
                raise RuntimeError(f'Empty `shards` in {index_path} file.')
            stream_per_shard += [stream_id] * len(stream_shards)
            self.shard_offset_per_stream[stream_id] = len(self.shards)
            self.shards_per_stream[stream_id] = len(stream_shards)
            self.sample_offset_per_stream[stream_id] = self.num_samples
            self.samples_per_stream[stream_id] = samples
            self.shards += stream_shards
            self.num_samples += samples
        self.stream_per_shard = np.array(stream_per_shard, np.int64)
        self.num_shards = len(self.shards)

        # Build the shard index (for partitioning and mapping samples to shards).
        self.samples_per_shard = np.array([shard.samples for shard in self.shards], np.int64)
        self.sample_offset_per_shard = self.samples_per_shard.cumsum() - self.samples_per_shard
        self.spanner = Spanner(self.samples_per_shard)

        # Now that we know the number of underlying samples of each stream, derive each stream's
        # true proportion/repeat/choose, as well as the total epoch size.
        self.choose = Stream.apply_weights(self.streams, self.samples_per_stream, choose,
                                           self.shuffle_seed)

        # Register/lookup our shared memory prefix and filelock root directory.
        my_locals = [
            os.path.abspath(os.path.join(stream.local, stream.split)) for stream in streams
        ]
        self._shm_prefix, self._locals_shm = get_shm_prefix(my_locals, world)
        self._filelock_root = os.path.join(os.path.sep, 'tmp', 'streaming', self._shm_prefix)

        # Create the shared memory-backed barrier, without its lock, which is unpickleable.
        shared_barrier_filelock_path = os.path.join(self._filelock_root, 'barrier_filelock')
        shared_barrier_shm_path = f'{self._shm_prefix}_barrier'
        self._shared_barrier = SharedBarrier(shared_barrier_filelock_path, shared_barrier_shm_path)

        # Epoch counter.
        #
        # Note: we do not assume that the end of __iter__() will ever be reached, so we need to
        # increment the epoch counter at the start of __iter__() instead of at the end, so we need
        # to track what the next epoch is, not the current epoch.
        self._next_epoch = SharedScalar(np.int64, f'{self._shm_prefix}_next_epoch')

        # Cache filelock. Proects downloading and evicting shards.
        self._cache_filelock_path = os.path.join(self._filelock_root, '_cache_filelock')
        self._cache_filelock: FileLock

        # Cache usage in bytes.
        self._cache_usage = SharedScalar(np.int64, f'{self._shm_prefix}_cache_usage')

        # Shard states array. Tells if a shard is missing, downloading, or present (eviction
        # happens under the lock).
        self._shard_states = SharedArray(self.num_shards, np.uint8,
                                         f'{self._shm_prefix}_shard_states')

        # Time of last access per shard. This is used to decide which shard(s) to evict when we run
        # out of space.
        self._shard_access_times = SharedArray(self.num_shards, np.float32,
                                               f'{self._shm_prefix}_shard_access_times')

        # Initialize shared memory objects.
        if world.is_local_leader:
            # Set initial epoch (before any resumption).
            self.next_epoch = 0

            # Normalize each stream's local dir, discovering which shards are present.
            are_shards_present = []
            for stream in self.streams:
                stream_shards = stream.get_shards(world)
                are_shards_present += stream.init_local_dir(stream_shards)

            # Calculate the initial cache usage using shard presence info.
            #
            # If we are above cache_limit, do nothing about it until the first download (which will
            # evict until happy).
            self.cache_usage = 0
            for stream in self.streams:
                self.cache_usage += stream.get_index_size()
            for shard_id, is_shard_present in enumerate(are_shards_present):
                if is_shard_present:
                    stream_id = self.stream_per_shard[shard_id]
                    stream = self.streams[stream_id]
                    shard = self.shards[shard_id]
                    self.cache_usage += shard.get_persistent_size(stream.safe_keep_zip)

            # Also use shard presence to initialize the shard states array and last access times.
            for shard_id, is_shard_present in enumerate(are_shards_present):
                if is_shard_present:
                    self._shard_states[shard_id] = _ShardState.PRESENT
                    self._shard_access_times[shard_id] = time()
                else:
                    self._shard_states[shard_id] = _ShardState.MISSING
                    self._shard_access_times[shard_id] = NEVER

        self._shared_barrier(world.workers_per_node)

        # Placeholder for a shared memory object where load_state_dict() saves its data to be
        # picked up by __iter__().
        self._resume_shm: SharedMemory

        # Placeholder for an _IterState which tracks state during __iter__().
        self._iter_state: _IterState

        del self._shared_barrier.lock  # Remove the lock that makes it unpickleable.

    def __del__(self) -> None:
        """Destructor, which releases its local working directories."""
        if hasattr(self, '_locals_shm'):
            try:
                self._locals_shm.buf[:4] = np.int32(0).tobytes()
            except:
                pass

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
        """Get the length as an IterableDataset.

        Returns:
            int: Dataset length.
        """
        return ceil(self.num_samples / World().num_ranks)

    def _resume(self, world: World, epoch: int) -> Tuple[int, int]:
        """Either resume from checkpoint or start at the beginning.

        Args:
            world (World): World state.
            epoch (int): What epoch we think it is (pre-checkpoint).

        Returns:
            Tuple[int, int]: What epoch this is, and sample offset in that epoch.
        """
        # Get the resume state, if it exists.
        name = f'{self._shm_prefix}_resume'
        try:
            shm = SharedMemory(name=name, create=False)
        except FileNotFoundError:
            # There is nothing to resume.
            if not self.num_canonical_nodes:
                self.num_canonical_nodes = world.num_nodes
            return epoch, 0

        # SharedMemory buffers may contain additional null bytes at the end.
        buf = bytes(shm.buf)
        index = buf.find(b'\0')
        buf = buf[:index] if index != -1 else buf
        obj = json.loads(buf.decode('utf-8'))

        # Check if the resume state is stale.
        if obj['epoch'] < epoch:
            if not self.num_canonical_nodes:
                self.num_canonical_nodes = world.num_nodes
            return epoch, 0

        # Load the correct resumption meta data.
        epoch = obj['epoch']
        sample_in_epoch = obj['sample_in_epoch']
        self.num_canonical_nodes = obj['num_canonical_nodes']
        self.shuffle_seed = obj['shuffle_seed']

        return epoch, sample_in_epoch

    def _resume_incr_epoch(self, world: World) -> Tuple[int, int]:
        """Start or resume training, pre-incrementing the next epoch.

        Args:
            world (World): World state.

        Returns:
            Tuple[int, int]: What epoch this is, and sample offset in that epoch.
        """
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
        name = f'{self._shm_prefix}_resume'
        data = json.dumps(obj, sort_keys=True).encode('utf-8')
        # some platforms choose to allocate chunks of memory based upon that platform’s memory
        # page size, hence, the exact size of the shared memory block may be larger or
        # equal to the size requested.
        self._resume_shm = SharedMemory(name=name, size=len(data))
        self._resume_shm.buf[:len(data)] = data

    def _resample_streams(self, epoch: int) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Perform the up/down-sampling needed to generate the weighted epoch.

        Args:
            epoch (int): What epoch this is for. Used in seeding the sampling RNG.

        Returns:
            Tuple[NDArray[np.int64], NDArray[np.int64]]: Sampled shard sizes and sample mapping.
        """
        # Initialize random number generator and arrays.
        rng = np.random.default_rng(self.shuffle_seed + epoch)
        shuffle_units = []
        sample_ids = []

        # Iterate over each stream.
        for stream_id in range(self.num_streams):
            stream_shard_offset = self.shard_offset_per_stream[stream_id]
            num_stream_shards = self.shards_per_stream[stream_id]
            stream_shard_ids = stream_shard_offset + np.arange(num_stream_shards)

            # Calculate choose per stream shard.
            samples_per_stream_shard = self.samples_per_shard[stream_shard_ids]
            stream_samples = sum(samples_per_stream_shard)
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
                # Calculate shuffle units.
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
                shortfall = shard_choose % shard_samples
                if shortfall:
                    partial_repeat = shard_sample_offset + rng.choice(
                        shard_samples, shortfall, False)
                    partial_repeat.sort()
                    sample_ids.append(partial_repeat)

        shuffle_units = np.concatenate(shuffle_units)
        sample_ids = np.concatenate(sample_ids)
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
            raise RuntimeError('Number of canonical nodes can never be None')

        # Sample each shard of each stream according to their proportions/repeats/samples. This
        # gives us the resampled size of each underlying shard, and a mapping from each fake "big"
        # sample ID to its underlying "small" sample ID.
        shuffle_units, small_per_big = self._resample_streams(epoch)

        # Partition the global sample space (of resampled "big" sample IDs) into a tensor of shape
        # (num physical nodes, ranks per node, workers per rank, batches per worker, samples per
        # batch) such that we have an elastically deterministic sample order.
        big_ids = get_partitions(self.partition_algo, self.choose, self.num_canonical_nodes,
                                 world.num_nodes, world.ranks_per_node, world.workers_per_rank,
                                 self.batch_size, sample_in_epoch)

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
            raise ValueError('Sample IDs must be of shape (num physical nodes, ranks per node, ' +
                             'workers per rank, batches per worker, batch size)')

        # Save the generated epoch shape to shared memory.
        name = f'{self._shm_prefix}_epoch_shape'
        size = ndim * np.int64().nbytes
        shape_shm = SharedMemory(name=name, create=True, size=size, auto_cleanup=False)
        shape_shm.buf[:size] = np.array(sample_ids.shape, np.int64).tobytes()

        # Save the generated epoch data to shared memory.
        name = f'{self._shm_prefix}_epoch_data'
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
        name = f'{self._shm_prefix}_epoch_shape'
        size = ndim * np.int64().nbytes
        shape_shm = SharedMemory(name=name, create=False, size=size, auto_cleanup=False)
        shape = tuple(np.ndarray(5, buffer=shape_shm.buf, dtype=np.int64))

        # Attach to the generated epoch data in shared memory.
        name = f'{self._shm_prefix}_epoch_data'
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

        self._shared_barrier(world.workers_per_node)

        return worker_sample_ids

    def _evict_shard(self, shard_id: int) -> None:
        """Evict a shard.

        Assumes you hold ``_cache_filelock``, preventing anyone else from modifying the cache. We
        expect that shard deletions are very fast.

        This method is called internally by ``_download_shard`` to clear space for more downloads.

        Args:
            shard_id (int): Shard to evict.
        """
        # Delete the shard's last access time, so that it is not searchable when finding the
        # coldest shard to evict. This is done by setting the time far into the future.
        self._shard_access_times[shard_id] = NEVER

        # Set the shard state to missing.
        self._shard_states[shard_id] = _ShardState.MISSING

        # Perform the eviction.
        shard = self.shards[shard_id]
        shard.evict()

        # Lastly, update cache usage to account for the removal.
        stream_id = self.stream_per_shard[shard_id]
        stream = self.streams[stream_id]
        self.cache_usage -= shard.get_persistent_size(stream.safe_keep_zip)

    def _evict_coldest_shard(self) -> None:
        """Evict the coldeset (i.e., least recently accessed) shard.

        Assumes you hold ``_cache_filelock``, preventing anyone else from modifying the cache. We
        expect that shard deletions are very fast.

        This method is called internally by ``_download_shard`` to clear space for more downloads.
        """
        # Get the shard with the oldest last access time.
        shard_id = int(self._shard_access_times.numpy().argmin())
        last_accessed = self._shard_access_times[shard_id]

        # Verify the access time is valid (the shard has not been evicted). This will only pose a
        # problem if there are no shards left to evict but you are trying to anyway.
        if np.allclose(last_accessed, NEVER):
            raise RuntimeError('Internal error: need to evict a shard to free up space, but ' +
                               'no shards are present to evict')

        # Evict that shard.
        self._evict_shard(shard_id)

    def _download_shard(self, shard_id: int, blocking: bool = True) -> None:
        """Download a shard, either waiting or skipping if in progress by another worker.

        If cache limit is enabled, this method may delete one or more other shards to make space
        for this download.

        Args:
            shard_id (int): Shard to download.
            blocking (bool): Whether to wait or skip if the shard is currently being downloaded by
                someone else.
        """
        # Lock the cache. FileLocks contain threading Locks, which are not pickleable, which is
        # incompatible with spawn, so must be created lazily.
        if not hasattr(self, '_cache_filelock'):
            self._cache_filelock = FileLock(self._cache_filelock_path)
        lock = self._cache_filelock
        lock.acquire()

        # Get the state of the shard to download.
        state = self._shard_states[shard_id]

        if state == _ShardState.MISSING:
            # If missing, transition state to downloading.
            self._shard_states[shard_id] = _ShardState.DOWNLOADING

            # Get the stream and shard.
            stream_id = self.stream_per_shard[shard_id]
            stream = self.streams[stream_id]
            shard = self.shards[shard_id]

            # If cache_limit is enabled, we first may have to make space for the new shard.
            if self.cache_limit:
                # Here, we give the shard a provisional access time so it doesn't get inadvertently
                # evicted if we need to evict shard(s).
                self._shard_access_times[shard_id] = time()

                # Then, we evict one shard at a time until our download will stay under the cache
                # limit. This means both the raw and zip forms of the shard due to decompressing.
                shard_full_size = shard.get_full_size()
                while self.cache_limit < self.cache_usage + shard_full_size:
                    self._evict_coldest_shard()

                # Calculate and apply the persistent change in cache usage, which depends on
                # whether compression was used and keep_zip.
                self.cache_usage += shard.get_persistent_size(stream.safe_keep_zip)

            # With the above preamble done, we can release the cache lock.
            lock.release()

            # Perform the download.
            stream.download_shard(shard)

            # Download completed, so transition shard state to present.
            with lock:
                self._shard_states[shard_id] = _ShardState.PRESENT
        elif state == _ShardState.DOWNLOADING:
            # Someone else is currently downloading the shard. Release the lock for them to make
            # progress.
            lock.release()

            # Do we wait on them?
            if blocking:
                # Wait for the shard to transition out of downloading state (to present).
                while self._shard_states[shard_id] == _ShardState.DOWNLOADING:
                    sleep(TICK)
        elif state == _ShardState.PRESENT:
            # Shard is already downloaded. There is nothing to do.
            lock.release()
        else:
            # Unknown state.
            lock.release()
            raise RuntimeError(f'Unknown shard state: {state}')

        # Finally, update the shard's last access time to now.
        self._shard_access_times[shard_id] = time()

    def __getitem__(self, sample_id: int) -> Any:
        """Get sample by global index, blocking to download its shard if not present.

        Args:
            sample_id (int): Sample index.

        Returns:
            Dict[str, Any]: Mapping of column name to column data.
        """
        # Locate the shard and sample offset within that shard where the sample lives.
        shard_id, shard_sample_id = self.spanner[sample_id]
        shard = self.shards[shard_id]

        try:
            # Shortcut path: just assume the shard is present. Using exceptions as control flow is
            # actually faster than doing it properly because python.
            sample = shard[shard_sample_id]

            # Manually update the last access time afterward. Normally this would have happened at
            # the end of _download_shard (see proper path below).
            self._shard_access_times[shard_id] = time()
        except:
            # Proper path: First ensure the shard is downloaded, then access the sample.
            self._download_shard(shard_id)
            sample = shard[shard_sample_id]

        return sample

    def _download_thread(self, it: _IterState) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling exit() on its state (only one epoch is valid at a
        time).

        Each worker has its own download thread, which iterates ahead of the ready thread and yield
        loop.

        Args:
            it (_IterState): State of __iter__.
        """
        # Download loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if it.is_exiting():
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if it.download_index == it.total:
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = it.download_index - it.yield_index
                if self.predownload <= samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = it.sample_ids[it.download_index]
            if sample_id == -1:
                it.download_index += 1
                continue

            # Download and decompress the shard for this sample, if not already done.
            shard_id, _ = self.spanner[sample_id]
            self._download_shard(shard_id, False)

            # Step forward one sample.
            it.download_index += 1

        # Note that we exited.
        it.on_exit()

    def _ready_thread(self, it: _IterState) -> None:
        """Wait for the relevant shards to become downloaded while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling exit() on its state (only one epoch is valid at a
        time).

        Each worker has its own ready thread, which iterates behind the download thread and ahead
        of the yield loop.

        Args:
            it (_IterState): State of __iter__.
        """
        # Ready loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if it.is_exiting():
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if it.ready_index == it.total:
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = it.ready_index - it.yield_index
                if self.predownload <= samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = it.sample_ids[it.ready_index]
            if sample_id == -1:
                it.ready_index += 1
                continue

            # Wait for the shard for this sample to be downloaded and decompressed, if not already.
            shard_id, _ = self.spanner[sample_id]
            while self._shard_states[shard_id] != _ShardState.PRESENT:
                sleep(TICK)

            # Step forward one sample.
            it.ready_index += 1

        # Note that we exited.
        it.on_exit()

    def _each_sample_id(self, it: _IterState) -> Iterator[int]:
        """Iterate over our samples while waiting for them to download first.

        This method is entered at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling exit() on its state (only one epoch is valid at a
        time).

        Each worker has its own yield loop, which iterates behind the download and ready threads.

        Args:
            it (_IterState): State of __iter__.

        Returns:
            Iterator[int]: Each sample, having been downloaded.
        """
        # Yield loop.
        while True:
            # If we've started a new epoch before this one is finished, exit this thread.
            if it.is_exiting():
                break

            # Have we yielded all our samples?
            if it.yield_index == it.total:
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
        # Lazily create the shared barrier's FileLock, which contains a threading Lock, which is
        # unpickleable.
        if not hasattr(self._shared_barrier, 'lock'):
            self._shared_barrier.lock = FileLock(self._shared_barrier.filelock_path)

        # Exit the threads that are pre-downloading and iterating the shards for previous epoch, if
        # it exists.
        if hasattr(self, '_iter_state'):
            self._iter_state.exit()

        # Discover where we left off, if there is a checkpoint, or start at the next epoch.
        # Also pre-increment the epoch counter.
        world = World()
        epoch, sample_in_epoch = self._resume_incr_epoch(world)

        # Get this worker's partition of samples to process.
        sample_ids = self._get_work(world, epoch, sample_in_epoch)
        if not len(sample_ids):  # Resumed at end of epoch, out of samples.
            return

        # Iterate over the samples while downloading ahead.
        self._iter_state = it = _IterState(sample_ids)
        Thread(target=self._download_thread, args=(it,), daemon=True).start()
        Thread(target=self._ready_thread, args=(it,), daemon=True).start()
        yield from map(self.__getitem__, self._each_sample_id(it))
