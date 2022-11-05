# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A streaming pytorch IterableDataset, resumable mid-epoch, whose shards reside locally."""

import json
import os
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from threading import Thread
from time import sleep
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from filelock import FileLock
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from streaming.base import distributed as dist
from streaming.base.compression import decompress
from streaming.base.download import download
from streaming.base.format import reader_from_json
from streaming.base.format.base.reader import FileInfo
from streaming.base.hashing import get_hash
from streaming.base.index import Index, get_index_basename
from streaming.base.shared import SharedBarrier
from streaming.base.shuffle import get_shuffle
from streaming.base.world import World

# Time to wait, in seconds.
TICK = 0.07


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

    Args:
        sample_ids (NDArray[np.int64]): This worker's partition of the sample space.
    """

    def __init__(self, sample_ids: NDArray[np.int64]) -> None:
        self.sample_ids = sample_ids
        self.total = len(sample_ids)
        self.iter_index = 0
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
        while self.iter_index < self.total:
            if self.iter_index < self.download_index:
                yield self.sample_ids[self.iter_index]
                self.iter_index += 1
                continue
            if self.is_stopped:
                break
            sleep(TICK)


class Dataset(IterableDataset):
    """A streaming pytorch IterableDataset that is also resumable mid-epoch.

    Checkpoints are represented in JSON as follows:

        {
            'epoch': int,
            'sample_in_epoch': int,
        }

    Args:
        local (str): Local dataset directory where shards are cached by split.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to
            ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        shuffle_world_size (int, optional): Canonical world size for shuffling. Defaults to
            ``None``, which is interpreted as the world size of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 predownload: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 shuffle_seed: Optional[int] = None,
                 shuffle_world_size: Optional[int] = None,
                 batch_size: Optional[int] = None):
        self.local = local
        self.remote = remote
        self.split = split or ''  # Empty string for os.path.join().
        self.shuffle = shuffle
        self.predownload = predownload
        self.keep_zip = keep_zip
        self.download_retry = download_retry
        self.download_timeout = download_timeout
        self.validate_hash = validate_hash or None
        # Seed is set below.
        world = World()
        if shuffle_world_size is None:
            shuffle_world_size = world.num_workers
        if shuffle_world_size < 1:
            raise ValueError('Interleave must be at least one.')
        self.shuffle_world_size = shuffle_world_size
        self.batch_size = batch_size

        # Load the index.json file.
        basename = get_index_basename()
        if world.is_local_leader:
            filename = self._download_file(basename)
        else:
            filename = os.path.join(local, split, basename)  # pyright: ignore
        dist.barrier()
        obj = json.load(open(filename))
        if obj['version'] != 2:
            raise ValueError('Unsupported version')

        # Initialize shard readers according to the loaded info.
        self.shards = []
        for info in obj['shards']:
            shard = reader_from_json(local, split, info)
            self.shards.append(shard)

        # Build the Index (for partitioning and mapping samples to shards).
        self.shard_sizes = np.array([x.samples for x in self.shards])
        self.index = Index(self.shard_sizes)

        # Setup for coordinating.
        device = torch.device(f'cuda:{world.rank_of_node}')
        tensor = torch.zeros(1, dtype=torch.int64, device=device)

        # Coordinate the shuffle seed across ranks.
        if world.is_leader:
            if shuffle_seed is None:
                shuffle_seed = np.random.randint(1 << 60)
            tensor[0] = shuffle_seed
        dist.broadcast(tensor, 0)
        self.shuffle_seed = int(tensor)

        # Add a coordinated random prefix to all shm names for uniqueness.
        if world.is_leader:
            tensor[0] = np.random.randint(1 << 60)
        dist.broadcast(tensor, 0)
        self._prefix = f'{int(tensor):016x}_{self.split}'

        # Set up the epoch counter.
        #
        # Note: we do not assume that the end of __iter__() will ever be reached, so we need to
        # increment the epoch counter at the start of __iter__() instead of at the end, so we need
        # to track what the next epoch is, not the current epoch.
        name = f'{self._prefix}_next_epoch'
        size = np.int64().nbytes
        try:
            self._next_epoch_shm = SharedMemory(name, True, size)
        except FileExistsError:
            self._next_epoch_shm = SharedMemory(name, False, size)
        self._next_epoch_arr = np.ndarray(1, buffer=self._next_epoch_shm.buf, dtype=np.int64)
        self._next_epoch_arr[0] = 0

        # Placeholder for _resume_shm, a shared memory object where load_state_dict() saves its
        # data to be picked up by __iter__().
        self._resume_shm = None

        # Create the barrier.
        self._worker_barrier_filelock_path = os.path.join(os.path.sep, 'tmp', 'streaming',
                                                          self._prefix, 'barrier_filelock')
        self._worker_barrier_shm_path = f'{self._prefix}_barrier'
        self._worker_barrier = SharedBarrier(self._worker_barrier_filelock_path,
                                             self._worker_barrier_shm_path)

        # Partition state.
        self._partition_state = None

    @property
    def next_epoch(self) -> int:
        """Get property next_epoch.

        Returns:
            int: Next epoch.
        """
        return int(self._next_epoch_arr[0])

    @next_epoch.setter
    def next_epoch(self, next_epoch: int) -> None:
        """Set property next_epoch.

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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get sample by global index.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard, index_in_shard = self.index.find_sample(index)
        reader = self.shards[shard]
        return reader[index_in_shard]

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
            return epoch, 0

        # Parse the existent resume state.
        obj = json.loads(bytes(shm.buf).decode('utf-8'))

        # Check if the resume state is stale.
        if obj['epoch'] < epoch:
            # Clean up stale state.
            if world.is_local_leader:
                shm.close()
                shm.unlink()
            return epoch, 0

        # Load the correct epoch and sample offset.
        epoch = obj['epoch']
        sample_in_epoch = obj['sample_in_epoch']
        return epoch, sample_in_epoch

    def _get_progress(self, world: World) -> Tuple[int, int]:
        """Start or resume training, pre-incrementing next_epoch.

        Args:
            world (World): World state.

        Returns:
            Tuple[int, int]: What epoch this is, and sample offset in that epoch.
        """
        # Either resume from checkpoint, or start from scratch.
        presumed_epoch = self.next_epoch
        epoch, sample_in_epoch = self._resume(world, presumed_epoch)

        # Wait for everyone to get the epoch above.
        self._worker_barrier(world.num_workers)

        # Set the new next epoch.
        if world.is_local_leader:
            self.next_epoch = epoch + 1

        return epoch, sample_in_epoch

    def _get_partition(self, world: World, epoch: int,
                       sample_in_epoch: int) -> Optional[NDArray[np.int64]]:
        """Get this worker's partition of this epoch's sample space.

        Args:
            world (World): World state.
            epoch (int): Which epoch it is.
            sample_in_epoch (int): Where we are in the epoch.

        Returns:
            Optional[NDArray[np.int64]]: Our partition of the epoch.
        """
        shm_name = f'{self._prefix}_ordering'

        # Local leader generates the global ordering of samples this epoch.
        if world.is_local_leader:
            ids = get_shuffle(self.shard_sizes, self.shuffle, self.shuffle_seed,
                              self.shuffle_world_size, epoch)
            ids = ids[sample_in_epoch:]
            if not len(ids):
                return None
            shm_size = len(ids.tobytes())
            leader_shm = SharedMemory(shm_name, True, shm_size)
            leader_shm.buf[:] = ids.tobytes()

        # Wait for local leader to populate the shared memory object.
        self._worker_barrier(world.num_workers)

        # Each worker extracts its partition from the global shuffle.
        shm = SharedMemory(shm_name)
        num_samples = len(shm.buf) // np.int64(0).nbytes
        ids = np.ndarray(num_samples, buffer=shm.buf, dtype=np.int64)
        ids = ids[world.worker::world.num_workers].copy()  # TODO: partition correctly.
        shm.close()

        # Wait for all workers to load from that shared memory.
        self._worker_barrier(world.num_workers)

        # Clean up shared memory.
        if world.is_local_leader:
            leader_shm.close()  # pyright: ignore
            leader_shm.unlink()  # pyright: ignore

        return ids

    def _download_file(self, basename: str) -> str:
        """Safely download a file from remote to local cache.

        Args:
            basename (str): Basename of file to download.

        Returns:
            str: Local cache filename.
        """
        if self.remote is None:
            remote = None
        else:
            remote = os.path.join(self.remote, self.split, basename)
        local = os.path.join(self.local, self.split, basename)
        for _ in range(1 + self.download_retry):
            try:
                download(remote, local, self.download_timeout)
            except:
                continue
            break
        return local

    def _decompress_shard_part(self, zip_info: FileInfo, zip_filename: str, raw_filename: str,
                               compression: Optional[str]) -> None:
        """Validate and decompress shard data.

        Args:
            zip_info (FileInfo): Compressed file info.
            zip_filename (str): Compressed filename.
            raw_filename (str): Decompressed filename.
            compression (str, optional): Compression algorithm.
        """
        # Load compressed.
        data = open(zip_filename, 'rb').read()

        # Validate what was downloaded.
        if self.validate_hash:
            if get_hash(self.validate_hash, data) != zip_info.hashes[self.validate_hash]:
                raise ValueError(f'Checksum failure: {zip_filename}')

        # Decompress and save that.
        data = decompress(compression, data)  # pyright: ignore
        tmp_filename = raw_filename + '.tmp'
        with open(tmp_filename, 'wb') as out:
            out.write(data)
        os.rename(tmp_filename, raw_filename)

        # Maybe remove compressed to save space.
        if not self.keep_zip:
            os.remove(zip_filename)

    def _download_shard_part(self,
                             raw_info: FileInfo,
                             zip_info: Optional[FileInfo] = None,
                             compression: Optional[str] = None) -> None:
        """Download shard data given metadata for the raw and compressed versions of it.

        MDS format uses joint shards (ie, one file per shard). Other formats supported by streaming
        use split shards (ie, shard data lives in two files per shard: the raw data itself and
        metadata in a separate file).

        Args:
            raw_info (FileInfo): Raw file info.
            zip_info (FileInfo, optional): Zip file info. Defaults to ``None``.
            compression (str, optional): Compression algorithm used for zip_info. Defaults to
                ``None``.
        """
        # If the local raw file already exists, this is a no-op.
        raw_filename = os.path.join(self.local, self.split, raw_info.basename)
        if os.path.isfile(raw_filename):
            return

        # Is compression used?
        if zip_info:
            # Download the compressed form if missing.
            zip_filename = os.path.join(self.local, self.split, zip_info.basename)
            if not os.path.isfile(zip_filename):
                self._download_file(zip_info.basename)

            # Validate and decompress.
            self._decompress_shard_part(zip_info, zip_filename, raw_filename, compression)
        else:
            # Download the raw form.
            self._download_file(raw_info.basename)

            # Validate if requested.
            if self.validate_hash:
                data = open(raw_filename, 'rb').read()
                if get_hash(self.validate_hash, data) != raw_info.hashes[self.validate_hash]:
                    raise ValueError(f'Checksum failure: {raw_filename}')

    def _download_shard(self, shard_id: int) -> None:
        """Download the given shard.

        Args:
            shard_id (int): Shard ID.
        """
        reader = self.shards[shard_id]
        for raw_info, zip_info in reader.file_pairs:
            self._download_shard_part(raw_info, zip_info, reader.compression)

    def _download_or_await_shard(self, lock: FileLock, shard_states: NDArray[np.uint8],
                                 shard_id: int) -> None:
        """Either download the given shard or wait on its download.

        Args:
            lock (FileLock): The lock protecting ``shard_states``.
            shard_states (NDArray[np.uint8]): The download status of each shard, as an array in
                shared memory.
            shard_id (int): Shard ID.
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
            self._download_shard(shard_id)
            # A shard state that is DOWNLOADING will never be written to elsewhere, so we don't
            # need to take the lock here.
            shard_states[shard_id] = _ShardState.DOWNLOADED
        elif state == _ShardState.DOWNLOADING:
            lock.release()
            while shard_states[shard_id] != _ShardState.DOWNLOADED:
                sleep(TICK)
        elif state == _ShardState.DOWNLOADED:
            lock.release()
        else:
            raise RuntimeError('Unknown shard state')

    def _download_thread(self, state: _PartitionState) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling stop() on its state (only one epoch is valid at a
        time).

        Each worker has its own download thread, which iterates ahead of the main thread.

        Args:
            state (_PartitionState): The partition state.
        """
        # Get the filelock that protects shard_states shared memory array.
        filename = os.path.join(os.path.sep, 'tmp', 'streaming', self._prefix,
                                '_shard_states_filelock')
        shard_states_lock = FileLock(filename)

        # Create or attach shard_states array (tells if each shard is unknown, downlaoding, or
        # downloaded).
        name = f'{self._prefix}_shard_states'
        size = len(self.shard_sizes) * np.uint8(0).nbytes
        try:
            shm = SharedMemory(name, True, size)
        except FileExistsError:
            shm = SharedMemory(name)
        shard_states = np.ndarray(len(self.shard_sizes), buffer=shm.buf, dtype=np.uint8)

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
                samples_ahead = state.download_index - state.iter_index
                if self.predownload <= samples_ahead:
                    sleep(TICK)
                    continue

            # Download and decompress the shard for this sample, if not already done.
            sample_id = state.sample_ids[state.download_index]
            shard_id, _ = self.index.find_sample(sample_id)
            self._download_or_await_shard(shard_states_lock, shard_states, shard_id)
            state.download_index += 1

    def _each_sample(self, sample_ids: NDArray[np.int64]) -> Iterator[int]:
        """Iterate over each sample ID, while downloading ahead in the background.

        Args:
            sample_ids (NDArray[np.int64]): The sample IDs to download and iterate.

        Returns:
            Iterator[int]: Each sample ID, having been downloaded.
        """
        self._partition_state = _PartitionState(sample_ids)
        Thread(target=self._download_thread, args=(self._partition_state,), daemon=True).start()
        yield from self._partition_state

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        # Exit the thread that is downloading the shards for last epoch, if it exists.
        if self._partition_state:
            self._partition_state.stop()

        # Discover where we left off, if there is a checkpoint, or start at the next epoch.
        # Also pre-increment the epoch counter.
        world = World()
        epoch, sample_in_epoch = self._get_progress(world)

        # Get this worker's partition of samples to process.
        sample_ids = self._get_partition(world, epoch, sample_in_epoch)
        if sample_ids is None:  # Hit end of epoch, out of samples.
            return

        # Iterate over the samples while downloading ahead.
        for sample_id in self._each_sample(sample_ids):
            yield self[sample_id]

    def state_dict(self, sample_in_epoch: int) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Args:
            sample_in_epoch (int): The number of samples processed so far in the current epoch.

        Returns:
            Dict[str, Any]: The state.
        """
        world = World()
        epoch = self.next_epoch - 1
        epoch, offset = self._resume(world, epoch)
        return {
            'epoch': epoch,
            'sample_in_epoch': offset + sample_in_epoch,
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
            self._resume_shm = SharedMemory(name, True, len(data))
            self._resume_shm.buf[:] = data
        except FileExistsError:
            self._resume_shm = SharedMemory(name)
            assert len(self._resume_shm.buf) == len(data)
