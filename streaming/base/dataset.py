# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A mid-epoch-resumable streaming/caching pytorch IterableDataset."""

import json
import os
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from threading import Thread
from time import sleep, time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from streaming.base.distributed import barrier
from streaming.base.index import Index
from streaming.base.partitioning import get_partitions
from streaming.base.shared import SharedBarrier, create_shared_memory
from streaming.base.shuffle import get_shuffle
from streaming.base.stream import Stream
from streaming.base.util import TICK, wait_for_file_to_exist, wait_for_local_leader
from streaming.base.world import World


class _ShardState(IntEnum):
    """The download status of a shard.

    Restrictions:
    - The initial state of UNKNOWN must be zero.
    - The state will only ever change in the upward direction.
    """
    UNKNOWN = 0
    DOWNLOADING = 1
    DOWNLOADED = 2


class _PartitionState:
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
      * Knobs to control streaming behavior, which if multiple Streams are provided, become defaults
        applied to them:
          * ``split``
          * ``download_retry``
          * ``download_timeout``
          * ``validate_hash``
          * ``keep_zip``
          * ``keep_raw``
      * How to iterate (controlling prefetching, partitioning, and shuffling):
          * Prefetching:
              * ``predownload``
          * Partitioning:
              * ``partition_algo``
              * ``num_canonical_nodes``
              * ``batch_size``
          * Shuffling:
              * ``shuffle``
              * ``shuffle_algo``
              * ``shuffle_seed``

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
            `False``.
        keep_raw (bool): Whether to keep or delete the decompressed form (or only form)
            of shards after all their samples have been yielded this epoch. If ``False``, keep iff
            remote is local or no remote and no compression. Defaults to ``True``.
        samples_per_epoch (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
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
                 keep_raw: bool = True,
                 samples_per_epoch: Optional[int] = None,
                 predownload: Optional[int] = 100_000,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176) -> None:
        # Global arguments (which do not live in Streams).
        self.samples_per_epoch = samples_per_epoch
        self.predownload = predownload
        self.partition_algo = partition_algo
        self.num_canonical_nodes = num_canonical_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_algo = shuffle_algo
        self.shuffle_seed = shuffle_seed

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
                         keep_zip=keep_zip,
                         keep_raw=keep_raw)

        # Normalize to a list of Streams.
        if streams:
            for stream in streams:
                stream.apply_default(default)
        else:
            streams = [default]

        # Validate proportions or repeats. Streams must each set at most one of proportion, repeat,
        # or samples. Proportion (relative) and repeat/samples (absolute) are incompatible with
        # each other.
        arr = np.zeros((len(streams), 4), np.uint8)
        for idx, stream in enumerate(streams):
            arr[idx, 0] = hasattr(stream, 'proportion')
            arr[idx, 1] = hasattr(stream, 'repeat')
            arr[idx, 2] = hasattr(stream, 'samples')
            arr[idx, 3] = not arr[idx].any()
        if (arr.sum(1) != 1).any():
            raise ValueError('Streams must set at most one of "proportion", "repeat", or ' +
                             '"samples"')
        if 0 < arr[:, 0].sum() < len(streams):
            raise ValueError('"Weights" (relative) and "repeat"/samples" (absolute) are ' +
                             'incompatible with each other')
        proportional = bool(arr[0, 0])
        if proportional == (samples_per_epoch is None):
            raise ValueError('You must provide samples_per_epoch iff using sample proportions')

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
        self.shard_offset_per_stream = np.zeros(len(streams), np.int64)
        self.shards_per_stream = np.zeros(len(streams), np.int64)
        self.sample_offset_per_stream = np.zeros(len(streams), np.int64)
        self.samples_per_stream = np.zeros(len(streams), np.int64)
        for idx, stream in enumerate(streams):
            stream_shards = stream.get_shards(world)
            samples = sum(map(len, stream_shards))
            stream_per_shard += [idx] * len(stream_shards)
            self.shard_offset_per_stream[idx] = len(self.shards)
            self.shards_per_stream[idx] = len(stream_shards)
            self.sample_offset_per_stream[idx] = self.num_samples
            self.samples_per_stream[idx] = samples
            self.shards += stream_shards
            self.num_samples += samples
        self.stream_per_shard = np.array(stream_per_shard, np.int64)

        # Now that we have the true size of each sub-dataset, derive the proportions/repeats/picks.
        if proportional:
            # Relative.
            self.proportion_per_stream = np.array([stream.proportion for stream in streams],
                                                  np.float64)
            self.proportion_per_stream /= self.proportion_per_stream.sum()
            self.pick_per_stream = (samples_per_epoch * self.proportion_per_stream).astype(
                np.int64)
            short = samples_per_epoch - self.pick_per_stream.sum()
            rng = np.random.default_rng(shuffle_seed)
            indices = rng.choice(len(self.pick_per_stream), short, False)
            self.pick_per_stream[indices] += 1
            self.repeat_per_stream = self.pick_per_stream / self.samples_per_stream
        else:
            # Absolute.
            self.pick_per_stream = np.zeros(len(streams), np.int64)
            for idx, stream in enumerate(streams):
                if hasattr(stream, 'repeat'):
                    samples = int(stream.repeat * self.samples_per_stream[idx])
                elif hasattr(stream, 'samples'):
                    samples = stream.samples
                else:
                    samples = self.samples_per_stream[idx]
                self.pick_per_stream[idx] = samples
            self.repeat_per_stream = self.pick_per_stream / self.samples_per_stream
            self.proportion_per_stream = self.pick_per_stream / self.pick_per_stream.sum()
            self.samples_per_epoch = sum(self.pick_per_stream)

        # Now that we know the true props/reps/picks, inject those back into the Streams,
        for stream, proportion, repeat, pick in zip(streams, self.proportion_per_stream,
                                                    self.repeat_per_stream, self.pick_per_stream):
            stream.proportion = proportion
            stream.repeat = repeat
            stream.samples = pick
        self.streams = streams

        # Build the Index (for partitioning and mapping samples to shards).
        self.shard_sizes = np.array([x.samples for x in self.shards])
        self.index = Index(self.shard_sizes)

        # Determine and distribute shuffle seed and shm prefix.
        seed_rng = np.random.default_rng(shuffle_seed)
        self.shuffle_seed = int(seed_rng.integers(1 << 60))
        prefix_int = int(seed_rng.integers(1 << 24))
        self._prefix = f'{prefix_int:06x}'

        # Should be a unique shared directory per each StreamingDataset instantiation to avoid a
        # conflict between a different StreamingDataset instance on a same machine.
        start_time = time()
        while True:
            self._shared_dir = os.path.join(os.path.sep, 'tmp', 'streaming', self._prefix)
            if os.path.exists(self._shared_dir):
                prefix_int = int(seed_rng.integers(1 << 24))
                self._prefix = f'{prefix_int:06x}'
            else:
                break
            elapsed = time() - start_time
            # Raise an exception if not finding a unique shared directory in 60 secs
            if elapsed > 60:
                raise RuntimeError(''.join([
                    f'Could not find the unique shared directory, bailing out.',
                    'Please provide a different `shuffle_seed` value.'
                ]))

            sleep(TICK)

        # Initialize the distributed package and synchronize all the ranks
        is_dist_pg_initialized = False
        if self._rank_world.num_ranks > 1:
            if dist.is_available() and not dist.is_initialized():
                is_dist_pg_initialized = True
                dist.init_process_group(backend='nccl' if torch.cuda.is_available() and
                                        dist.is_nccl_available() else 'gloo',
                                        rank=world.rank,
                                        world_size=world.num_ranks)
            dist.barrier()

        # Create the shared memory-backed worker barrier, without its lock, which is unpickleable.
        worker_barrier_filelock_path = os.path.join(self._shared_dir, 'barrier_filelock')
        worker_barrier_shm_path = f'{self._prefix}_barrier'
        self._worker_barrier = SharedBarrier(worker_barrier_filelock_path, worker_barrier_shm_path,
                                             world.is_local_leader)

        # Remove the lock that makes it unpickleable
        del self._worker_barrier.lock

        # Set up the epoch counter.
        #
        # Note: we do not assume that the end of __iter__() will ever be reached, so we need to
        # increment the epoch counter at the start of __iter__() instead of at the end, so we need
        # to track what the next epoch is, not the current epoch.
        self._next_epoch_shm = create_shared_memory(name=f'{self._prefix}_next_epoch',
                                                    size=np.int64().nbytes)
        self._next_epoch_arr = np.ndarray(1, buffer=self._next_epoch_shm.buf, dtype=np.int64)
        self._next_epoch_arr[0] = 0

        # Get the filelock filename that protects shard_states shared memory array.
        self.shard_states_filename = os.path.join(self._shared_dir, '_shard_states_filelock')

        # Create or attach shard_states array (tells if each shard is unknown, downloading, or
        # downloaded).
        self._shard_states = create_shared_memory(name=f'{self._prefix}_shard_states',
                                                  size=len(self.shard_sizes) * np.uint8(0).nbytes)

        # Destroy process group, and de-initialize the distributed package
        barrier()
        if is_dist_pg_initialized:
            dist.destroy_process_group()

        # Placeholder for a shared memory object where load_state_dict() saves its data to be
        # picked up by __iter__().
        self._resume_shm = None

        # Placeholder for a _PartitionState which tracks state during __iter__().
        self._partition_state = None

    def _get_next_epoch(self) -> int:
        """Get the next epoch.

        Returns:
            int: Next epoch.
        """
        return int(self._next_epoch_arr[0])

    def _set_next_epoch(self, next_epoch: int) -> None:
        """Set the next epoch.

        Args:
            next_epoch (int): Next epoch.
        """
        self._next_epoch_arr[0] = next_epoch

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
        name = f'{self._prefix}_resume'
        try:
            shm = SharedMemory(name)
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
        # Reference the same shared memory object in a worker process
        self._next_epoch_arr = np.ndarray(1, buffer=self._next_epoch_shm.buf, dtype=np.int64)

        # Either resume from checkpoint, or start from scratch.
        presumed_epoch = self._get_next_epoch()
        epoch, sample_in_epoch = self._resume(world, presumed_epoch)

        # Wait for everyone to get the epoch above.
        self._worker_barrier(world.workers_per_node)

        # Set the new next epoch.
        if world.is_local_leader:
            self._set_next_epoch(epoch + 1)

        return epoch, sample_in_epoch

    def _get_partition(self,
                       world: World,
                       epoch: int,
                       sample_in_epoch: int,
                       timeout: float = 60) -> NDArray[np.int64]:
        """Get this worker's partition of this epoch's sample space.

        Args:
            world (World): World state.
            epoch (int): Which epoch it is.
            sample_in_epoch (int): Where we are in the epoch.
            timeout (float): Max seconds to wait for the partitioning/shuffle to be generated.

        Returns:
            Optional[NDArray[np.int64]]: Our partition of the epoch.
        """
        # Ensure the parameters are not None. The parameters are either in _resume() or
        # in constructor method.
        if self.num_canonical_nodes is None:
            raise RuntimeError('Number of canonical nodes can never be None')
        if self.shuffle_seed is None:
            raise RuntimeError('Shuffle seed can never be None')

        # Decide where to save shuffle data.
        tmp_filename = os.path.join(self._shared_dir, 'shuffle.npy.tmp')
        filename = os.path.join(self._shared_dir, 'shuffle.npy')

        # In the local leader, generate this epoch's global sample ordering, then save to file.
        # Tensor shape: (num nodes, ranks per node, workers per rank, samples per worker).
        # This operation is expensive.
        if world.is_local_leader:
            sample_ids = get_partitions(self.index.total_samples, self.num_canonical_nodes,
                                        world.num_nodes, world.ranks_per_node,
                                        world.workers_per_rank, self.batch_size, sample_in_epoch)
            if self.shuffle:
                mapping = get_shuffle(self.shuffle_algo, self.shard_sizes,
                                      self.num_canonical_nodes, self.shuffle_seed, epoch)
                sample_ids = np.where(sample_ids == -1, -1, mapping[sample_ids])
            sample_ids.tofile(tmp_filename)
            os.rename(tmp_filename, filename)

        # Everyone waits for the file to become populated.
        wait_for_file_to_exist(filename, TICK, timeout, 'Partitioning and shuffling took too long')

        # Each worker loads its slice of the sample ID tensor to iterate through.
        # Tensor shape: (num nodes, ranks per node, workers per rank, samples per worker).
        sample_id_nbytes = np.int64().nbytes
        num_bytes = os.path.getsize(filename)
        if num_bytes % sample_id_nbytes:
            raise ValueError(f'Generated shuffle is invalid: {filename} ({num_bytes} bytes).')
        num_samples = num_bytes // sample_id_nbytes
        num_workers = world.num_nodes * world.ranks_per_node * world.workers_per_rank
        if num_samples % num_workers:
            raise ValueError(f'Generated shuffle is invalid: {filename} ({num_bytes} bytes).')
        samples_per_worker = num_samples // num_workers
        offset_in_bytes = world.worker * samples_per_worker * sample_id_nbytes
        bytes_to_read = samples_per_worker * sample_id_nbytes
        with open(filename, 'rb', 0) as fp:
            fp.seek(offset_in_bytes)
            data = fp.read(bytes_to_read)
        sample_ids = np.frombuffer(data, np.int64)

        # Wait for everyone to read their part.
        self._worker_barrier(world.workers_per_node)

        # Now clean up after ourselves.
        if world.is_local_leader:
            os.remove(filename)

        return sample_ids

    def _download_or_skip_shard(self, lock: FileLock, shard_states: NDArray[np.uint8],
                                shard_id: int, wait_if_downloading: bool) -> None:
        """Download a shard, waiting or skipping if in progress by another worker.

        Args:
            lock (FileLock): The lock protecting ``shard_states``.
            shard_states (NDArray[np.uint8]): The download status of each shard, as an array in
                shared memory.
            shard_id (int): Shard ID.
            wait_if_downloading (bool): Whether to wait or skip if the shard is currently being
                downloaded by someone else.
        """
        # First, the fast path: check the shared memory shard state without taking the lock. The
        # shard states only ever go up, so if we're at the downloaded state, it's downloaded.
        state = shard_states[shard_id]
        if state == _ShardState.DOWNLOADED:
            return

        # Shard is not necessarily downloaded, so check and update state with the lock.
        lock.acquire()
        state = shard_states[shard_id]
        if state == _ShardState.UNKNOWN:
            shard_states[shard_id] = _ShardState.DOWNLOADING
            lock.release()
            stream_id = self.stream_per_shard[shard_id]
            stream = self.streams[stream_id]
            shard = self.shards[shard_id]
            stream.download_shard(shard)
            # A shard state that is DOWNLOADING will never be written to elsewhere, so we don't
            # need to take the lock here.
            shard_states[shard_id] = _ShardState.DOWNLOADED
        elif state == _ShardState.DOWNLOADING:
            lock.release()
            if wait_if_downloading:
                while shard_states[shard_id] != _ShardState.DOWNLOADED:
                    sleep(TICK)
        elif state == _ShardState.DOWNLOADED:
            lock.release()
        else:
            raise RuntimeError('Unknown shard state')

    def _get_shard_states(self) -> Tuple[FileLock, NDArray[np.uint8]]:
        """Get the shared shard states array and its protecting lock.

        Returns:
            Tuple[FileLock, NDArray[np.uint8]]: Lock, and array.
        """
        # Get the filelock that protects shard_states shared memory array.
        lock = FileLock(self.shard_states_filename)

        shard_states = np.ndarray(len(self.shard_sizes),
                                  buffer=self._shard_states.buf,
                                  dtype=np.uint8)

        return lock, shard_states

    def __getitem__(self, idx: int) -> Any:
        """Get sample by global index, blocking to download its shard if not present.

        Args:
            idx (int): Sample index.

        Returns:
            Dict[str, Any]: Mapping of column name to column data.
        """
        # Locate the shard and sample offset within that shard where the sample lives.
        shard_idx, idx_in_shard = self.index.find_sample(idx)
        shard = self.shards[shard_idx]

        try:
            # Attempt to directly access the sample for performance reasons.
            sample = shard[idx_in_shard]
        except:
            # Get handles to the shared shard states array and its protective file lock.
            lock, shard_states = self._get_shard_states()

            # Download the shard if not already being downloaded. Block if download in progress.
            self._download_or_skip_shard(lock, shard_states, shard_idx, True)

            # Finally, access the sample.
            sample = shard[idx_in_shard]

        # Return the retrieved sample.
        return sample

    def _download_thread(self, state: _PartitionState) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling stop() on its state (only one epoch is valid at a
        time).

        Each worker has its own download thread, which iterates ahead of the main thread.

        Args:
            state (_PartitionState): The partition state.
        """
        shard_states_lock, shard_states = self._get_shard_states()

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
            self._download_or_skip_shard(shard_states_lock, shard_states, shard_id, False)
            state.download_index += 1

    def _ready_thread(self, state: _PartitionState) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling stop() on its state (only one epoch is valid at a
        time).

        Each worker has its own ready thread, which iterates ahead of the main thread.

        Args:
            state (_PartitionState): The partition state.
        """
        _, shard_states = self._get_shard_states()

        # Download loop.
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

            # Download and decompress the shard for this sample, if not already done.
            shard_id, _ = self.index.find_sample(sample_id)
            while shard_states[shard_id] != _ShardState.DOWNLOADED:
                sleep(TICK)
            state.ready_index += 1

    def _each_sample(self, sample_ids: NDArray[np.int64]) -> Iterator[int]:
        """Iterate over each sample ID, while downloading ahead in the background.

        Args:
            sample_ids (NDArray[np.int64]): The sample IDs to download and iterate.

        Returns:
            Iterator[int]: Each sample ID, having been downloaded.
        """
        self._partition_state = _PartitionState(sample_ids)
        Thread(target=self._download_thread, args=(self._partition_state,), daemon=True).start()
        Thread(target=self._ready_thread, args=(self._partition_state,), daemon=True).start()
        yield from self._partition_state

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        # Lazily create the worker barrier's FileLock, which contains a threading Lock, which is
        # unpickleable.
        if not hasattr(self._worker_barrier, 'lock'):
            self._worker_barrier.lock = FileLock(self._worker_barrier.filelock_path)

        # Exit the thread that is downloading the shards for last epoch, if it exists.
        if self._partition_state:
            self._partition_state.stop()

        # Discover where we left off, if there is a checkpoint, or start at the next epoch.
        # Also pre-increment the epoch counter.
        world = World()
        epoch, sample_in_epoch = self._resume_incr_epoch(world)

        # Get this worker's partition of samples to process.
        sample_ids = self._get_partition(world, epoch, sample_in_epoch)
        if not len(sample_ids):  # Resumed at end of epoch, out of samples.
            return

        # Iterate over the samples while downloading ahead.
        for sample_id in self._each_sample(sample_ids):
            yield self[sample_id]

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
        name = f'{self._prefix}_resume'
        data = json.dumps(obj, sort_keys=True).encode('utf-8')
        try:
            # some platforms choose to allocate chunks of memory based upon that platform’s memory
            # page size, hence, the exact size of the shared memory block may be larger or
            # equal to the size requested.
            self._resume_shm = SharedMemory(name, True, len(data))
            self._resume_shm.buf[:len(data)] = data
        except FileExistsError:
            sleep(TICK)
            self._resume_shm = SharedMemory(name)
            assert len(self._resume_shm.buf) == len(data)

    def _cleanup_shared_memory(self, shm: Any, world: World) -> None:
        """Clean up the shared memory resources.

        Args:
            shm (Any): A SharedMemory object
            world (World): World state.
        """
        if shm is not None:
            # Close each SharedMemory instance
            shm.close()
            if world.is_local_leader:
                # Call unlink only once to release the shared memory
                shm.unlink()
            else:
                # Wait for local leader process to execute first
                sleep(1)

    def __del__(self):
        # Wait for the local rank 0 process
        world = self._rank_world
        wait_for_local_leader(world)

        # Clean up shared memory resources
        if hasattr(self, '_next_epoch_shm'):
            self._cleanup_shared_memory(self._next_epoch_shm, world)
        if hasattr(self, '_shard_states'):
            self._cleanup_shared_memory(self._shard_states, world)
        if hasattr(self, '_resume_shm'):
            self._cleanup_shared_memory(self._resume_shm, world)
