# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A mid-epoch-resumable streaming/caching pytorch IterableDataset."""

import json
import os
from enum import IntEnum
from threading import Thread
from time import sleep
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from streaming.base.index import Index
from streaming.base.partition import get_partitions
from streaming.base.shared import SharedArray, SharedBarrier, SharedMemory, get_shm_prefix
from streaming.base.shuffle import get_shuffle
from streaming.base.stream import Stream
from streaming.base.util import TICK
from streaming.base.world import World


class _ShardState(IntEnum):
    """The download status of a shard.

    Restrictions:
    - The initial state of INVALID must be zero.
    - State transitions: MISSING -> DOWNLOADING -> PRESENT -> EVICTING -> MISSING.
    """
    INVALID = 0
    MISSING = 1
    DOWNLOADING = 2
    PRESENT = 3
    EVICTING = 4


class _IterState:
    """The download status of a partition of samples.

    0 <= yield <= ready <= download <= total

    Cursors
    * The yield cursor points to the (downloaded) sample we are yielding.
    * The ready cursor points to the last contiguously downloaded sample.
    * The download cursor points to the sample we are downloading (skipping other workers'
      downloads in progress).

    Args:
        sample_ids (NDArray[np.int64]): This worker's partition of the sample space.
    """

    def __init__(self, sample_ids: NDArray[np.int64]) -> None:
        self.sample_ids = sample_ids
        self.total = len(sample_ids)
        self.yield_index = 0
        self.ready_index = 0
        self.download_index = 0
        self.is_stopped = False

    def stop(self) -> None:
        """Stop the thread and exit."""
        self.is_stopped = True

    def __iter__(self) -> Iterator[int]:
        """Iterate over our samples while waiting for them to download first.

        Returns:
            Iterator[int]: Each sample, having been downloaded.
        """
        while self.yield_index < self.total:
            if self.yield_index < self.ready_index:
                sample_id = self.sample_ids[self.yield_index]
                if sample_id != -1:  # If -1, we skip.
                    yield sample_id
                self.yield_index += 1
                continue
            if self.is_stopped:
                break
            sleep(TICK)


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

    StreamingDataset init takes three kinds of arguments:

    * One or more Streams (you must provide either ``streams`` or ``remote``/``local``):

      * ``streams``
      * ``remote``
      * ``local``

    * Knobs to control streaming behavior, which, if multiple Streams are provided, become defaults
      applied to them:

      * ``split``
      * ``download_retry``
      * ``download_timeout``
      * ``validate_hash``
      * ``keep_zip``

    * How to iterate (controlling shard downloading, shard eviction, partitioning, and shuffling):

      * Shard downloading:

        * ``predownload``

      * Shard eviction:

        * ``cache_limit``

      * Partitioning:

        * ``partition_algo``
        * ``num_canonical_nodes``
        * ``batch_size``

      * Shuffling:

        * ``shuffle``
        * ``shuffle_algo``
        * ``shuffle_seed``
        * ``shuffle_block_size``

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
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
        samples_per_epoch (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
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
                 samples_per_epoch: Optional[int] = None,
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

        # Validate sub-dataset weight scheme ("proportion", "repeat", "samples", or none).
        are_weights_relative = Stream.validate_weights(streams)

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
            stream_per_shard += [stream_id] * len(stream_shards)
            self.shard_offset_per_stream[stream_id] = len(self.shards)
            self.shards_per_stream[stream_id] = len(stream_shards)
            self.sample_offset_per_stream[stream_id] = self.num_samples
            self.samples_per_stream[stream_id] = samples
            self.shards += stream_shards
            self.num_samples += samples
        self.stream_per_shard = np.array(stream_per_shard, np.int64)
        self.num_shards = len(self.shards)

        # Build the Index (for partitioning and mapping samples to shards).
        self.samples_per_shard = np.array([x.samples for x in self.shards])
        self.index = Index(self.samples_per_shard)

        # Now that we have the true size of each sub-dataset, derive the proportions/repeats/picks.
        if are_weights_relative:
            # Relative.
            if not samples_per_epoch:
                samples_per_epoch = self.index.total_samples
            self.proportion_per_stream = np.array([stream.proportion for stream in self.streams],
                                                  np.float64)
            self.proportion_per_stream /= self.proportion_per_stream.sum()
            self.pick_per_stream = (samples_per_epoch * self.proportion_per_stream).astype(
                np.int64)
            short = samples_per_epoch - self.pick_per_stream.sum()
            rng = np.random.default_rng(shuffle_seed)
            indices = rng.choice(self.num_streams, short, False)
            self.pick_per_stream[indices] += 1
            self.repeat_per_stream = self.pick_per_stream / self.samples_per_stream
            self.samples_per_epoch = samples_per_epoch
        else:
            # Absolute.
            if samples_per_epoch:
                raise ValueError('Only provide samples_per_epoch when proportionally weighting ' +
                                 'sub-datasets.')
            self.pick_per_stream = np.zeros(self.num_streams, np.int64)
            for stream_id, stream in enumerate(self.streams):
                if hasattr(stream, 'repeat'):
                    samples = int(stream.repeat * self.samples_per_stream[stream_id])
                elif hasattr(stream, 'samples'):
                    samples = stream.samples
                else:
                    samples = self.samples_per_stream[stream_id]
                self.pick_per_stream[stream_id] = samples
            self.repeat_per_stream = self.pick_per_stream / self.samples_per_stream
            self.proportion_per_stream = self.pick_per_stream / self.pick_per_stream.sum()
            self.samples_per_epoch = sum(self.pick_per_stream)

        # Now that we know the true props/reps/picks, inject those back into the Streams,
        for stream, proportion, repeat, pick in zip(self.streams, self.proportion_per_stream,
                                                    self.repeat_per_stream, self.pick_per_stream):
            stream.proportion = proportion
            stream.repeat = repeat
            stream.samples = pick

        # Register/lookup our shared memory prefix and filelock root directory.
        my_locals = [os.path.abspath(stream.local) for stream in streams]
        self._shm_prefix, self._locals_shm = get_shm_prefix(my_locals, world)
        self._filelock_root = os.path.join(os.path.sep, 'tmp', 'streaming', self._shm_prefix)

        # Create the shared memory-backed barrier, without its lock, which is unpickleable.
        shared_barrier_filelock_path = os.path.join(self._filelock_root, 'barrier_filelock')
        shared_barrier_shm_path = f'{self._shm_prefix}_barrier'
        self._shared_barrier = SharedBarrier(shared_barrier_filelock_path, shared_barrier_shm_path)

        # Set up the epoch counter.
        #
        # Note: we do not assume that the end of __iter__() will ever be reached, so we need to
        # increment the epoch counter at the start of __iter__() instead of at the end, so we need
        # to track what the next epoch is, not the current epoch.
        self._next_epoch = SharedArray(f'{self._shm_prefix}_next_epoch', 1, np.int64)

        # Create or attach shard states array (tells if each shard is unknown, downloading, or
        # downloaded).
        self._shard_states_filelock_path = os.path.join(self._filelock_root,
                                                        '_shard_states_filelock')
        self._shard_states_lock: FileLock
        self._shard_states = SharedArray(f'{self._shm_prefix}_shard_states', self.num_shards,
                                         np.uint8)

        # Size of each shard in bytes (raw and zip).
        #
        # Notes:
        # - Used for accounting to keep us under cache_limit.
        # - If compression was not used, the shard will not have a zip version, and we use -1.
        # - A "shard" comprises either one or two files, depending on the format.
        self._shard_raw_sizes = SharedArray(f'{self._shm_prefix}_shard_raw_sizes', self.num_shards,
                                            np.int64)
        self._shard_zip_sizes = SharedArray(f'{self._shm_prefix}_shard_zip_sizes', self.num_shards,
                                            np.int64)

        # Initialize shared memory objects.
        if world.is_local_leader:
            # Set initial epoch.
            self._next_epoch[0] = 0

            # Set initial shard states according to local dirs.
            are_shards_present = []
            for stream in self.streams:
                stream_shards = stream.get_shards(world)
                are_shards_present += stream.init_local_dir(stream_shards)
            for shard_id, is_shard_present in enumerate(are_shards_present):
                self._shard_states[shard_id] = \
                    _ShardState.PRESENT if is_shard_present else _ShardState.MISSING

            # Collect shard raw/zip sizes.
            raw_sizes, zip_sizes = zip(
                *map(lambda shard: shard.get_raw_and_zip_sizes(), self.shards))
            self._shard_raw_sizes[:] = raw_sizes
            self._shard_zip_sizes[:] = zip_sizes

        self._shared_barrier(world.workers_per_node)

        # Placeholder for a shared memory object where load_state_dict() saves its data to be
        # picked up by __iter__().
        self._resume_shm = None

        # Placeholder for an _IterState which tracks state during __iter__().
        self._iter_state = None

        del self._shared_barrier.lock  # Remove the lock that makes it unpickleable.

    def __del__(self) -> None:
        """Destructor, which releases its local working directories."""
        if hasattr(self, '_locals_shm'):
            try:
                self._locals_shm.buf[:4] = np.int32(0).tobytes()
            except:
                pass

    def _get_next_epoch(self) -> int:
        """Get the next epoch.

        Returns:
            int: Next epoch.
        """
        return int(self._next_epoch[0])

    def _set_next_epoch(self, next_epoch: int) -> None:
        """Set the next epoch.

        Args:
            next_epoch (int): Next epoch.
        """
        self._next_epoch[0] = next_epoch

    def __len__(self) -> int:
        """Get the length as an IterableDataset.

        Returns:
            int: Dataset length.
        """
        return self.index.get_samples_per_device()

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
        presumed_epoch = self._get_next_epoch()
        epoch, sample_in_epoch = self._resume(world, presumed_epoch)

        # Wait for everyone to get the epoch above.
        self._shared_barrier(world.workers_per_node)

        # Set the new next epoch.
        if world.is_local_leader:
            self._set_next_epoch(epoch + 1)

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
        epoch = self._get_next_epoch() - 1
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

            # Calculate pick per stream shard.
            samples_per_stream_shard = self.samples_per_shard[stream_shard_ids]
            stream_samples = sum(samples_per_stream_shard)
            stream_picks = self.pick_per_stream[stream_id]
            if stream_picks == stream_samples:
                pick_per_stream_shard = samples_per_stream_shard
            else:
                pick_per_stream_shard = samples_per_stream_shard * stream_picks // stream_samples
                short = stream_picks - pick_per_stream_shard.sum()
                indices = rng.choice(num_stream_shards, short, False)
                pick_per_stream_shard[indices] += 1

            # Iterate over each shard of this stream.
            for shard_id, shard_samples, shard_picks in zip(stream_shard_ids,
                                                            samples_per_stream_shard,
                                                            pick_per_stream_shard):
                # Calculate shuffle units.
                shard_shuffle_units = [shard_samples] * (shard_picks // shard_samples)
                remainder = shard_picks % shard_samples
                if remainder:
                    shard_shuffle_units.append(remainder)
                shuffle_units.append(shard_shuffle_units)

                # Calculate sample IDs of any full repeats.
                shard_sample_offset = self.index.shard_offsets[shard_id]
                num_full_repeats = shard_picks // shard_samples
                if num_full_repeats:
                    full_repeat = shard_sample_offset + np.arange(shard_samples)
                    sample_ids += [full_repeat] * num_full_repeats

                # Calculate sample IDs of a possible partial repeat.
                short = shard_picks % shard_samples
                if short:
                    partial_repeat = shard_sample_offset + rng.choice(shard_samples, short, False)
                    partial_repeat.sort()
                    sample_ids.append(partial_repeat)

        shuffle_units = np.concatenate(shuffle_units)
        sample_ids = np.concatenate(sample_ids)
        return shuffle_units, sample_ids

    def _generate_sample_ids(self, world: World, epoch: int,
                             sample_in_epoch: int) -> NDArray[np.int64]:
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
        big_ids = get_partitions(self.partition_algo, self.samples_per_epoch,
                                 self.num_canonical_nodes, world.num_nodes, world.ranks_per_node,
                                 world.workers_per_rank, self.batch_size, sample_in_epoch)

        # If we need to shuffle, shuffle in a node-aware and *underlying* shard-aware way.
        if self.shuffle:
            shuffle = get_shuffle(self.shuffle_algo, shuffle_units, self.num_canonical_nodes,
                                  self.shuffle_seed, epoch, self.shuffle_block_size)
            big_ids = np.where(big_ids != -1, shuffle[big_ids], -1)

        # Now that we have partitioning and shuffled with hallucinated "big" sample IDs, we don't
        # need them anymore, and can convert back to underlying "small" sample IDs.
        return np.where(big_ids != -1, small_per_big[big_ids], -1)

    def _share_sample_ids(self, sample_ids: NDArray[np.int64]) -> \
            Tuple[SharedMemory, SharedMemory]:
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

    def _attach_sample_ids(self) -> Tuple[NDArray[np.int64], SharedMemory, SharedMemory]:
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
            epoch_sample_ids = self._generate_sample_ids(world, epoch, sample_in_epoch)
            shape_shm, data_shm = self._share_sample_ids(epoch_sample_ids)
            self._shared_barrier(world.workers_per_node)
        else:
            self._shared_barrier(world.workers_per_node)
            epoch_sample_ids, shape_shm, data_shm = self._attach_sample_ids()

        # Each worker gets their portion of the work.
        worker_sample_ids = epoch_sample_ids[world.node, world.rank_of_node,
                                             world.worker_of_rank].flatten()
        self._shared_barrier(world.workers_per_node)

        # Now clean up after ourselves.
        shape_shm.cleanup()
        data_shm.cleanup()

        self._shared_barrier(world.workers_per_node)

        return worker_sample_ids

    def _evict_shard(self, shard_id: int, blocking: bool = True) -> None:
        """Evict a shard, waiting if in progress by another worker.

        Args:
            shard_id (int): Shard ID.
            blocking (bool): Whether to wait or fire-and-forget if the shard is currently being
                downloaded by someone else.
        """
        # FileLocks contain threading locks, which are not pickleable, so must be created lazily.
        if not hasattr(self, '_shard_states_lock'):
            self._shard_states_lock = FileLock(self._shard_states_filelock_path)

        def do_evict_shard(shard_id: int) -> None:
            self._shard_states[shard_id] = _ShardState.EVICTING
            self._shard_states_lock.release()
            stream_id = self.stream_per_shard[shard_id]
            stream = self.streams[stream_id]
            shard = self.shards[shard_id]
            stream.evict_shard(shard)
            self._shard_states_lock.acquire()
            self._shard_states[shard_id] = _ShardState.MISSING
            self._shard_states_lock.release()

        self._shard_states_lock.acquire()
        state = self._shard_states[shard_id]
        if state == _ShardState.MISSING:
            self._shard_states_lock.release()
        elif state == _ShardState.DOWNLOADING:
            self._shard_states_lock.release()
            while self._shard_states[shard_id] == _ShardState.DOWNLOADING:
                sleep(TICK)
            self._shard_states_lock.acquire()
            do_evict_shard(shard_id)
        elif state == _ShardState.PRESENT:
            do_evict_shard(shard_id)
        elif state == _ShardState.EVICTING:
            self._shard_states_lock.release()
            if blocking:
                while self._shard_states[shard_id] == _ShardState.EVICTING:
                    sleep(TICK)
        else:
            self._shard_states_lock.release()
            raise RuntimeError(f'Unknown shard state: {state}')

    def _download_shard(self, shard_id: int, blocking: bool = True) -> None:
        """Download a shard, waiting or skipping if in progress by another worker.

        Args:
            shard_id (int): Shard ID.
            blocking (bool): Whether to wait or fire-and-forget if the shard is currently being
                downloaded by someone else.
        """
        # FileLocks contain threading locks, which are not pickleable, so must be created lazily.
        if not hasattr(self, '_shard_states_lock'):
            self._shard_states_lock = FileLock(self._shard_states_filelock_path)

        def do_download_shard(shard_id: int) -> None:
            self._shard_states[shard_id] = _ShardState.DOWNLOADING
            self._shard_states_lock.release()
            stream_id = self.stream_per_shard[shard_id]
            stream = self.streams[stream_id]
            shard = self.shards[shard_id]
            stream.download_shard(shard)
            self._shard_states_lock.acquire()
            self._shard_states[shard_id] = _ShardState.PRESENT
            self._shard_states_lock.release()

        self._shard_states_lock.acquire()
        state = self._shard_states[shard_id]
        if state == _ShardState.MISSING:
            do_download_shard(shard_id)
        elif state == _ShardState.DOWNLOADING:
            self._shard_states_lock.release()
            if blocking:
                while self._shard_states[shard_id] == _ShardState.DOWNLOADING:
                    sleep(TICK)
        elif state == _ShardState.PRESENT:
            self._shard_states_lock.release()
        elif state == _ShardState.EVICTING:
            self._shard_states_lock.release()
            while self._shard_states[shard_id] == _ShardState.EVICTING:
                sleep(TICK)
            self._shard_states_lock.acquire()
            do_download_shard(shard_id)
        else:
            self._shard_states_lock.release()
            raise RuntimeError(f'Unknown shard state: {state}')

    def __getitem__(self, sample_id: int) -> Any:
        """Get sample by global index, blocking to download its shard if not present.

        Args:
            sample_id (int): Sample index.

        Returns:
            Dict[str, Any]: Mapping of column name to column data.
        """
        shard_id, shard_sample_id = self.index.find_sample(sample_id)
        shard = self.shards[shard_id]
        try:
            sample = shard[shard_sample_id]
        except:
            self._download_shard(shard_id)
            sample = shard[shard_sample_id]
        return sample

    def _download_thread(self, state: _IterState) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling stop() on its state (only one epoch is valid at a
        time).

        Each worker has its own download thread, which iterates ahead of the main thread.

        Args:
            state (_IterState): The partition state.
        """
        # Download loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if state.is_stopped:
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if state.download_index == state.total:
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = state.download_index - state.yield_index
                if self.predownload <= samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = state.sample_ids[state.download_index]
            if sample_id == -1:
                state.download_index += 1
                continue

            # Download and decompress the shard for this sample, if not already done.
            shard_id, _ = self.index.find_sample(sample_id)
            self._download_shard(shard_id, False)
            state.download_index += 1

    def _ready_thread(self, state: _IterState) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling stop() on its state (only one epoch is valid at a
        time).

        Each worker has its own ready thread, which iterates ahead of the main thread.

        Args:
            state (_IterState): The partition state.
        """
        # Ready loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if state.is_stopped:
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if state.ready_index == state.total:
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = state.ready_index - state.yield_index
                if self.predownload <= samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = state.sample_ids[state.ready_index]
            if sample_id == -1:
                state.ready_index += 1
                continue

            # Wait for this shard to become downloaded and decompressed, if not already.
            shard_id, _ = self.index.find_sample(sample_id)
            while self._shard_states[shard_id] != _ShardState.PRESENT:
                sleep(TICK)
            state.ready_index += 1

    def _each_sample(self, sample_ids: NDArray[np.int64]) -> Iterator[int]:
        """Iterate over each sample ID, while downloading ahead in the background.

        Args:
            sample_ids (NDArray[np.int64]): The sample IDs to download and iterate.

        Returns:
            Iterator[int]: Each sample ID, having been downloaded.
        """
        self._iter_state = _IterState(sample_ids)
        Thread(target=self._download_thread, args=(self._iter_state,), daemon=True).start()
        Thread(target=self._ready_thread, args=(self._iter_state,), daemon=True).start()
        yield from self._iter_state

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        # Lazily create the shared barrier's FileLock, which contains a threading Lock, which is
        # unpickleable.
        if not hasattr(self._shared_barrier, 'lock'):
            self._shared_barrier.lock = FileLock(self._shared_barrier.filelock_path)

        # Exit the thread that is downloading the shards for last epoch, if it exists.
        if self._iter_state:
            self._iter_state.stop()

        # Discover where we left off, if there is a checkpoint, or start at the next epoch.
        # Also pre-increment the epoch counter.
        world = World()
        epoch, sample_in_epoch = self._resume_incr_epoch(world)

        # Get this worker's partition of samples to process.
        sample_ids = self._get_work(world, epoch, sample_in_epoch)
        if not len(sample_ids):  # Resumed at end of epoch, out of samples.
            return

        # Iterate over the samples while downloading ahead.
        for sample_id in self._each_sample(sample_ids):
            yield self[sample_id]
