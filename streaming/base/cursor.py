# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Tracks distributed progress through a streaming dataset for checkpointing."""

from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict

import numpy as np
import torch
from torch import distributed as dist

from streaming.base.world import World


class Cursor:
    """Tracks distributed progress through a streaming dataset for checkpointing.

    Training is represented as sequence of one or more training sessions, which are cleared between
    epochs. A training session is an array of how many samples each worker has processed during
    this session.

    To restore from checkpoint, even while changing the number of worker partitions, we recreate
    the deterministic initial shuffle then replay the training history: splitting, truncating from
    front, and rejoining for each session in order.

    We communicate this state across all ranks and worker processes by putting it in an int64 array
    in shared memory as follows, which is synchronized in state_dict() and load_state_dict(). The
    array is organized as follows:

        - epoch
        - num sessions
        - for each session,
          - num workers
          - samples processed per worker

    Checkpoints are given in JSON as follows:

        {
            'epoch': int,
            'sessions': List[List[int]],
        }

    Args:
        split (str): Dataset split, used to uniquely name the shm resource.

    TODO: handle num_workers=0.
    """

    _epoch_slot = 0
    _num_sessions_slot = 1
    _sessions_slot = 2

    def __init__(self, split: str) -> None:
        name = f'cursor_{split}'
        size = 1 << 20
        try:
            self._shm = SharedMemory(name, True, size)
        except:
            self._shm = SharedMemory(name, False, size)

        self._arr = np.frombuffer(self._shm.buf, np.int64)

        self.sessions = []

    @property
    def epoch(self) -> int:
        """Get the current epoch.

        Returns:
            int: The epoch.
        """
        return int(self._arr[self._epoch_slot])

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        """Set the current epoch.

        Args:
            epoch (int): The epoch.
        """
        self._arr[self._epoch_slot] = epoch

    @property
    def _num_sessions(self) -> int:
        """Get the number of sessions.

        Returns:
            int: Number of sessions.
        """
        return int(self._arr[self._num_sessions_slot])

    @_num_sessions.setter
    def _num_sessions(self, num_sessions: int) -> None:
        """Set the number of sessions.

        Args:
            num_sessions (int): Number of sessions.
        """
        self._arr[self._num_sessions_slot] = num_sessions

    def _get_end_slot(self) -> int:
        """Get the slot that lies after the last session.

        Returns:
            int: The end slot.
        """
        index = self._sessions_slot
        for _ in range(self._num_sessions):
            num_workers = self._arr[index]
            index += 1 + num_workers
        return index

    def push_session(self, world: World) -> None:
        """Begin a new training session at the beginning of an epoch.

        Args:
            world (World): The world.
        """
        if world.is_local_leader:
            index = self._get_end_slot()
            self._arr[index] = world.num_workers
            index += 1
            session = self._arr[index:index + world.num_workers]
            session[:] = 0
            self._num_sessions += 1
            self.sessions.append(session)
        world.barrier()

    def step_sample(self, world: World) -> None:
        """Incremenet this worker's sample position by one.

        Args:
            world (World): The world.
        """
        session = self.sessions[-1]
        session[world.worker] += 1

    def pop_sessions(self, world: World) -> None:
        """Reset our sessions at the end of an epoch.

        Args:
            world (World): The world.
        """
        if world.is_local_leader:
            self._num_sessions = 0
        self.sessions = []
        world.barrier()

    def step_epoch(self, world: World):
        """Increment the epoch by one.

        Args:
            world (World): The world.
        """
        if world.is_local_leader:
            self.epoch += 1
        world.barrier()

    def _all_gather_current_session(self) -> None:
        """All-gather the current session data.

        This is done in order to checkpoint.
        """
        # Bail if we are not multi-node.
        world = World()
        if not world.is_multinode:
            return

        # Bail if we are not currently training. If we aren't training, there are no running
        # sample counts to synchonize.
        if not self.sessions:
            return

        # Do the all_gather on the last session counts.
        current_session = self.sessions[-1]
        device = torch.device(f'cuda:{world.rank}')
        source = torch.tensor(current_session, device=device)
        dests = [
            torch.empty(len(current_session), dtype=torch.int64, device=device)
            for _ in range(world.num_ranks)
        ]
        dist.all_gather(dests, source)

        # Each rank provides ground truth for its workers.
        if world.is_local_leader:
            dests = torch.stack(dests).cpu().numpy()  # Shape: (world size, total workers).
            for rank in range(world.num_ranks):
                rank_start = rank * world.workers_per_rank
                rank_end = (rank + 1) * world.workers_per_rank
                current_session[rank_start:rank_end] = dests[rank]
        world.barrier()

    def state_dict(self) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        Returns:
            Dict[str, Any]: The state.
        """
        self._all_gather_current_session()

        epoch = self.epoch
        sessions = [x.tolist() for x in self.sessions]
        return {
            'epoch': epoch,
            'sessions': sessions,
        }

    def _broadcast_state(self) -> None:
        """Broadcast the entire state.

        This is done following a restore from checkpoint.
        """
        # Bail if we are not multi-node.
        world = World()
        if not world.is_multinode:
            return

        # Do the broadcast on the entire state.
        device = torch.device(f'cuda:{world.rank}')
        tensor = torch.tensor(self._arr, device=device)
        dist.broadcast(tensor, 0)

        # Each other rank receives ground truth from rank zero.
        if world.is_local_leader and world.rank:
            self._arr[:] = tensor.cpu().numpy()
        world.barrier()

        # Collect sessions list.
        self.sessions = []
        index = self._sessions_slot
        for _ in range(self._num_sessions):
            num_workers = self._arr[index]
            index += 1
            session = self._arr[index:index + num_workers]
            self.sessions.append(session)
            index += num_workers

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        Args:
            obj (Dict[str, Any]): The state.
        """
        self.epoch = obj['epoch']
        self._num_sessions = len(obj['sessions'])
        index = self._sessions_slot
        for session in obj['sessions']:
            self._arr[index] = num_workers = len(session)
            index += 1
            self._arr[index:index + num_workers] = session
            index += num_workers

        self._broadcast_state()
