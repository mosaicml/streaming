# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A streaming pytorch IterableDataset, resumable mid-epoch, whose shards reside locally."""

import json
import os
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import distributed as dist
from torch.utils.data import IterableDataset

from streaming.base.format import reader_from_json
from streaming.base.index import Index
from streaming.base.shared import SharedBarrier
from streaming.base.shuffle import get_epoch
from streaming.base.world import World


class LocalResumableDataset(IterableDataset):
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
        local (str): Local dataset directory where the dataset is present.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to shuffle the samples while iterating. Defaults to ``True``.
        seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        workers_per_rank (int): Workers per rank. Defauls to ``8``.
    """

    def __init__(self,
                 local: str,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 workers_per_rank: int = 8):
        self.local = local
        self.split = split or ''
        self.shuffle = shuffle

        # Load the index.json file.
        filename = os.path.join(local, split, 'index.json')  # pyright: ignore
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
        world = World()
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
        except:
            self._next_epoch_shm = SharedMemory(name)
        self._next_epoch_arr = np.ndarray(1, buffer=self._next_epoch_shm.buf, dtype=np.int64)
        self._next_epoch_arr[0] = 0

        # Placeholder for _resume_shm, a shared memory object where load_state_dict() saves its
        # data to be picked up by __iter__().
        self._resume_shm = None

        # Create the barrier.
        count = world.ranks_per_node * workers_per_rank
        self._barrier_filelock_path = os.path.join('/tmp', self._prefix, 'barrier_filelock')
        self._barrier_shm_path = f'{self._prefix}_barrier_shm'
        self._barrier = SharedBarrier(count, self._barrier_filelock_path, self._barrier_shm_path)

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
        except:
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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        # Load resume state and create the new training session.
        presumed_epoch = self.next_epoch
        epoch, old_sessions = self._resume(presumed_epoch)
        cur_session, _ = self._create_cur_session(epoch)

        self._barrier()

        # Update the pre-incremented epoch counter.
        world = World()
        if world.is_local_leader:
            self.next_epoch = epoch + 1

        self._barrier()

        # Generate the partitions and get ours.
        sessions = old_sessions + [cur_session]
        sequences = get_epoch(self.shard_sizes, self.shuffle, self.seed, epoch, sessions)
        todos = sequences[world.worker]

        # Iterate over our partition's samples.
        for index in todos:
            cur_session[world.worker] += 1
            yield self[index]

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

    def state_dict(self) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Returns:
            Dict[str, Any]: The state.
        """
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
        except:
            self._resume_shm = SharedMemory(name)
            assert len(self._resume_shm.buf) == len(data)
