# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A streaming pytorch IterableDataset, resumable mid-epoch, whose shards reside locally."""

import json
import os
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from threading import Thread
from time import sleep
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from filelock import FileLock
from numpy.typing import NDArray
from torch import distributed as dist
from torch.utils.data import IterableDataset

from streaming.base.download import download
from streaming.base.format import reader_from_json
from streaming.base.format.base.reader import FileInfo
from streaming.base.hashing import get_hash
from streaming.base.index import Index, get_index_basename
from streaming.base.shared import SharedBarrier
from streaming.base.shuffle import get_epoch
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


class Dataset(IterableDataset):
    """A streaming pytorch IterableDataset, resumable mid-epoch, whose shards reside locally.

    Training is represented as sequence of one or more training sessions, which are cleared between
    epochs. A training session is an array of how many samples each worker has processed during
    this session.

    To restore from checkpoint, even while changing the number of worker partitions, we recreate
    the deterministic initial shuffle then replay the training history: splitting, truncating from
    front, and rejoining for each session in order.

    We communicate this state across all ranks and worker processes by putting it in shared memory
    objects which are updated during checkpointing and training.

    Checkpoints are represented in JSON as follows:

        {
            'epoch': int,
            'sessions': List[List[int]],
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
        seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        num_workers (int, optional): Number of workers of its DataLoader, which determines the size
            of the barrier to coordinate workers while iterating. Defaults to ``None``.
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
                 seed: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 num_workers: Optional[int] = None):
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
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load the index.json file.
        basename = get_index_basename()
        world = World()
        if world.is_local_leader:
            filename = self._download_file(basename)
        else:
            filename = os.path.join(local, split, basename)  # pyright: ignore
        dist.barrier()
        obj = json.load(open(filename))
        assert obj['version'] == 2

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

        # Coordinate the seed across ranks.
        if world.is_leader:
            if seed is None:
                seed = np.random.randint(1 << 60)
            tensor[0] = seed
        dist.broadcast(tensor, 0)
        self.seed = int(tensor)

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
            self._next_epoch_shm = SharedMemory(name)
        self._next_epoch_arr = np.ndarray(1, buffer=self._next_epoch_shm.buf, dtype=np.int64)
        self._next_epoch_arr[0] = 0

        # Placeholder for _resume_shm, a shared memory object where load_state_dict() saves its
        # data to be picked up by __iter__().
        self._resume_shm = None

        # Create the barrier.
        total_workers = world.ranks_per_node * (self.num_workers or 1)
        self._barrier_filelock_path = os.path.join('/tmp', 'mds', self._prefix, 'barrier_filelock')
        self._barrier_shm_path = f'{self._prefix}_barrier_shm'
        self._barrier = SharedBarrier(total_workers, self._barrier_filelock_path,
                                      self._barrier_shm_path)

        self._iter_index = 0
        self._download_index = 0

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

    def _resume(self, epoch: int) -> Tuple[int, List[NDArray[np.int64]]]:
        """Resume from checkpoint.

        Args:
            epoch (int): What epoch we think it is (pre-checkpoint).

        Returns:
            Tuple[int, List[NDArray[np.int64]]]: Pair of (resumed epoch, old sessions).
        """
        world = World()

        # Get the resume state, if it exists.
        name = f'{self._prefix}_resume'
        try:
            shm = SharedMemory(name)
        except:
            # There is nothing to resume.
            return epoch, []

        # Parse the existent resume state.
        obj = json.loads(bytes(shm.buf).decode('utf-8'))

        # Check if the resume state is stale.
        if obj['epoch'] < epoch:
            # Clean up stale state.
            if world.is_local_leader:
                shm.unlink()
            return epoch, []

        # Load the correct epoch and previous training sessions this epoch.
        epoch = obj['epoch']
        old_sessions = [np.array(x) for x in obj['sessions']]
        return epoch, old_sessions

    def _create_cur_session(self, epoch: int) -> Tuple[NDArray[np.int64], SharedMemory]:
        """Create the current session, as we have just started an epoch.

        Called by __iter__() in a worker process, or a per-rank process if workers aren't used.

        Note: this also returns the underlying shared memory object, because the returned array
        will become invalidated when it goes out of scope.

        Args:
            epoch (int): The current epoch.

        Returns:
            Tuple[NDArray[np.int64], SharedMemory]: Session and handle to shared memory.
        """
        world = World()
        shm_name = f'{self._prefix}_session_{epoch}'
        shm_bytes = world.num_workers * np.int64().nbytes
        try:
            shm = SharedMemory(shm_name, True, shm_bytes)
        except FileExistsError:
            shm = SharedMemory(shm_name)
        cur_session = np.ndarray(world.num_workers, buffer=shm.buf, dtype=np.int64)
        cur_session[:] = 0
        return cur_session, shm

    def _lookup_cur_session(
            self, epoch: int) -> Tuple[Optional[NDArray[np.int64]], Optional[SharedMemory]]:
        """Look up the current session, which exists if we are currently training.

        Called by state_dict() in a per-rank process.

        Note: this also returns the underlying shared memory object (if it exists), because the
        returned array will become invalidated when it goes out of scope.

        Args:
            epoch (int): The current epoch.

        Returns:
            Tuple[Optional[NDArray[np.int64]], Optional[SharedMemory]]: Maybe session, maybe shm.
        """
        shm_name = f'{self._prefix}_session_{epoch}'
        try:
            shm = SharedMemory(shm_name)
        except:
            return None, None
        num_workers = shm.size // np.int64().nbytes
        cur_session = np.ndarray(num_workers, buffer=shm.buf, dtype=np.int64)
        return cur_session, shm

    def _get_partition(self, epoch: int, sessions: List[NDArray[np.int64]],
                       world: World) -> Optional[NDArray[np.int64]]:
        # Local leader generates the partitions.
        if world.is_local_leader:
            sequences = get_epoch(self.shard_sizes, self.shuffle, self.seed, epoch, sessions)
            base = world.node * world.ranks_per_node * world.workers_per_rank
            for rank_of_node in range(world.ranks_per_node):
                for worker_of_rank in range(world.workers_per_rank):
                    worker = base + rank_of_node * world.workers_per_rank + worker_of_rank
                    name = f'{self._prefix}_part_{worker:03}'
                    sequence = sequences[worker]
                    size = len(sequence) * np.int64(0).nbytes
                    if not size:
                        continue
                    shm = SharedMemory(name, True, size)
                    shm.buf[:] = sequence.tobytes()

        self._barrier()

        # Load our partition.
        name = f'{self._prefix}_part_{world.worker:03}'
        try:
            shm = SharedMemory(name)
        except:
            return None
        todos = np.frombuffer(shm.buf, np.int64).copy()
        shm.unlink()

        return todos

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
            assert get_hash(self.validate_hash, data) == zip_info.hashes[self.validate_hash]

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
                             shard: int,
                             raw_info: FileInfo,
                             zip_info: Optional[FileInfo] = None,
                             compression: Optional[str] = None) -> None:
        """Download shard data given metadata for the raw and compressed versions of it.

        MDS format uses joint shards (ie, one file per shard). Other formats supported by streaming
        use split shards (ie, shard data lives in two files per shard: the raw data itself and
        metadata in a separate file).

        Args:
            shard (int): Shard ID.
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
                assert get_hash(self.validate_hash, data) == raw_info.hashes[self.validate_hash]

    def _download_shard(self, shard_id: int) -> None:
        """Download the given shard.

        Args:
            shard_id (int): Shard ID.
        """
        reader = self.shards[shard_id]
        for raw_info, zip_info in reader.file_pairs:
            self._download_shard_part(shard_id, raw_info, zip_info, reader.compression)

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
                sleep(0.07)
        elif state == _ShardState.DOWNLOADED:
            lock.release()
        else:
            raise RuntimeError('Unknown shard state')

    def _download_thread(self, epoch: int, sample_ids: NDArray[np.int64]) -> None:
        """Download the relevant shards in the background while we are being iterated.

        Args:
            epoch (int): Which epoch. On noticing that a new epoch has started, we exit this thread
                because a new one will soon be running, with different sample IDs.
            sample_ids (NDArray[np.int64]): The samples to download the shards of.
        """
        # Create or attach shard_states array.
        name = f'{self._prefix}_shard_states'
        size = len(self.shard_sizes) * np.uint8(0).nbytes
        try:
            shm = SharedMemory(name, True, size)
        except:
            shm = SharedMemory(name)
        shard_states = np.ndarray(len(self.shard_sizes), buffer=shm.buf, dtype=np.uint8)

        filename = os.path.join('/tmp', 'mds', self._prefix, '_shard_states_filelock')
        shard_states_lock = FileLock(filename)

        # Download loop.
        num_samples = len(sample_ids)
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if epoch != self.next_epoch - 1:
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if self._download_index == num_samples:
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = self._download_index - self._iter_index
                if self.predownload <= samples_ahead:
                    sleep(0.07)
                    continue

            # Download and decompress the shard for this sample, if not already done.
            sample_id = sample_ids[self._download_index]
            shard_id, _ = self.index.find_sample(sample_id)
            self._download_or_await_shard(shard_states_lock, shard_states, shard_id)
            self._download_index += 1

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        # Load resume state and create the new training session.
        presumed_epoch = self.next_epoch
        epoch, old_sessions = self._resume(presumed_epoch)
        cur_session, _ = self._create_cur_session(epoch)

        world = World()
        assert world.num_workers == self._barrier.count
        self._barrier()

        # Update the pre-incremented epoch counter.
        world = World()
        if world.is_local_leader:
            self.next_epoch = epoch + 1

        self._barrier()

        # Get the samples for this worker to process.
        sessions = old_sessions + [cur_session]
        sample_ids = self._get_partition(epoch, sessions, world)
        if sample_ids is None:
            return

        # Iterate while downloading.
        self._iter_index = 0
        self._download_index = 0
        Thread(target=self._download_thread, args=(epoch, sample_ids)).run()
        num_samples = len(sample_ids)
        while self._iter_index < num_samples:
            if self._iter_index < self._download_index:
                cur_session[world.worker] += 1
                sample_id = sample_ids[self._iter_index]
                yield self[sample_id]
                self._iter_index += 1
                continue
            sleep(0.07)

        # Any code after the yields will never be reached by the Composer trainer.

    def _all_gather_current_session(self, session: NDArray[np.int64]) -> None:
        """All-gather the current session data.

        This is done in order to checkpoint.

        Args:
            session (NDArray[np.int64]): The current session.
        """
        # Bail if we are not multi-node.
        world = World()
        if not world.is_multinode:
            return

        # Do the all_gather on the last session counts.
        device = torch.device(f'cuda:{world.rank}')
        source = torch.tensor(session, device=device)
        dests = [
            torch.empty(len(session), dtype=torch.int64, device=device)
            for _ in range(world.num_ranks)
        ]
        dist.all_gather(dests, source)

        # Each rank provides ground truth for its workers.
        if world.is_local_leader:
            dests = torch.stack(dests).cpu().numpy()  # Shape: (world size, total workers).
            for rank in range(world.num_ranks):
                rank_start = rank * world.workers_per_rank
                rank_end = (rank + 1) * world.workers_per_rank
                session[rank_start:rank_end] = dests[rank]

        # Wait for local leaders to load session state from the other nodes.
        dist.barrier()

    def state_dict(self, batches_in_epoch: int) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Args:
            batches_in_epoch (int): The number of batches processed so far in current epoch.
        Returns:
            Dict[str, Any]: The state.
        """
        # TODO: Use batches_in_epoch to compute status of workers instead of shm / cur_session

        # Attempt to load resume state, if it exists.
        epoch, old_sessions = self._resume(self.next_epoch - 1)

        # Get the current training session array, if we are currently training.
        cur_session, _ = self._lookup_cur_session(epoch)

        # Concatenate the sessions, synchronizing current session if we have one.
        if cur_session is not None:
            self._all_gather_current_session(cur_session)
            sessions = old_sessions + [cur_session]
        else:
            sessions = old_sessions

        return {
            'epoch': epoch,
            'sessions': [x.tolist() for x in sessions],
        }

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        Args:
            obj (Dict[str, Any]): The state.
        """
        # Set the number of the next epoch.
        self.next_epoch = obj['epoch']

        # Save the resume state (old sessions).
        name = f'{self._prefix}_resume'
        data = json.dumps(obj, sort_keys=True).encode('utf-8')
        try:
            self._resume_shm = SharedMemory(name, True, len(data))
            self._resume_shm.buf[:] = data
        except FileExistsError:
            self._resume_shm = SharedMemory(name)
            assert len(self._resume_shm.buf) == len(data)
