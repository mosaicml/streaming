# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Tracks distributed progress through a streaming dataset for checkpointing."""

from multiprocessing.shared_memory import SharedMemory
from time import sleep
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch import distributed

from streaming.base import distributed as dist


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

    def __init__(self, split: str) -> None:
        name = f'cursor_{split}'
        size = 1 << 20
        try:
            self._shm = SharedMemory(name, True, size)
        except:
            self._shm = SharedMemory(name, False, size)

        self._arr = np.frombuffer(self._shm.buf, np.int64)

        self._epoch_slot = 0
        self._num_sessions_slot = 1
        self._sessions_slot = 2

    def get_epoch(self) -> int:
        """Get the current epoch.

        Returns:
            int: The epoch.
        """
        return int(self._arr[self._epoch_slot])

    def step_epoch(self):
        """Increment the epoch by one."""
        if dist.is_local_leader():
            self._arr[self._epoch_slot] += 1
        else:
            sleep(0.07)

    def _get_current_session_slot(self) -> Optional[int]:
        """Find the starting slot of the current session in the array.

        Used for synchrnoization during state_dict()/load_state_dict().

        If training is not in progress (no sessions), returns None.

        Returns:
            Optional[int]: The slot, if training is in progress.
        """
        num_sessions = self._arr[self._num_sessions_slot]
        if not num_sessions:
            # Mitigate
            self._arr[self._sessions_slot] = dist.get_num_workers()
            return self._sessions_slot

        index = self._sessions_slot
        for _ in range(num_sessions - 1):
            num_workers = self._arr[index]
            index += 1 + num_workers
        return index

    def new_session(self) -> int:
        """Start a new training session, returning our sample slot.

        A sample slot is where this worker's count of samples seen this session is found in _arr.

        Note: don't precompute this in dataset __init__ because you might be in the wrong process
        and get a garbage result. Instead, call at the start of __iter__.

        Returns:
            int: Slot of _arr where this worker's samples seen this session are counted.
        """
        if dist.is_local_leader():
            self._arr[self._num_sessions_slot] += 1
        else:
            sleep(0.07)
        index = self._get_current_session_slot()
        assert index
        self._arr[index] = num_workers = dist.get_num_workers()
        index += 1
        self._arr[index:index + num_workers] = 0
        return index + dist.get_worker()

    def step_sample(self, sample_slot: int) -> None:
        """Incremenet this worker's sample position by one.

        Args:
            sample_slot (int): The slot in _arr where this count is stored.
        """
        self._arr[sample_slot] += 1

    def clear_sessions(self) -> None:
        """Reset our sessions at the end of an epoch."""
        if dist.is_local_leader():
            self._arr[self._num_sessions_slot] = 0
        else:
            sleep(0.07)

    def each_session(self) -> Iterator[NDArray[np.int64]]:
        """Iterate over our sessions.

        Returns:
            Iterator[NDArray[np.int64]]: The iterator.
        """
        num_sessions = self._arr[self._num_sessions_slot]
        index = self._sessions_slot
        for _ in range(num_sessions):
            num_workers = self._arr[index]
            index += 1
            yield self._arr[index:index + num_workers]
            index += num_workers

    def _all_gather_current_session(self) -> None:
        """All-gather the current session data.

        This is done in order to checkpoint.
        """
        # Bail if we are not multi-node.
        if not dist.is_multinode():
            return

        # Bail if we are not currently training. If we aren't training, there are no running
        # sample counts to synchonize.
        index = self._get_current_session_slot()
        if not index:
            return

        # Do the all_gather on the last session counts.
        num_workers = self._arr[index]
        index += 1
        samples_per_worker = self._arr[index:index + num_workers]
        rank = dist.get_rank()
        device = torch.device(f'cuda:{rank}')
        source = torch.tensor(samples_per_worker, device=device)
        world_size = dist.get_world_size()
        dests = [
            torch.empty(num_workers, dtype=torch.int64, device=device) for _ in range(world_size)
        ]
        distributed.all_gather(dests, source)

        # Each rank provides ground truth for its workers.
        if dist.is_local_leader():
            dests = torch.stack(dests).cpu().numpy()  # Shape: (world size, total workers).
            workers_per_rank = num_workers // world_size
            for rank in range(world_size):
                rank_start = rank * workers_per_rank
                rank_end = (rank + 1) * workers_per_rank
                self._arr[index + rank_start:index + rank_end] = dests[rank]
        else:
            sleep(0.07)

    def _broadcast_state(self) -> None:
        """Broadcast the entire state.

        This is done following a restore from checkpoint.
        """
        # Bail if we are not multi-node.
        if not dist.is_multinode():
            return

        # Do the broadcast on the entire state.
        rank = dist.get_rank()
        device = torch.device(f'cuda:{rank}')
        tensor = torch.tensor(self._arr, device=device)
        distributed.broadcast(tensor, 0)

        # Each other rank receives ground truth from rank zero.
        if dist.is_local_leader() and rank:
            self._arr[:] = tensor.cpu().numpy()
        else:
            sleep(0.07)

    def state_dict(self) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        Returns:
            Dict[str, Any]: The state.
        """
        self._all_gather_current_session()

        epoch = self.get_epoch()
        sessions = [x.tolist() for x in self.each_session()]
        return {
            'epoch': epoch,
            'sessions': sessions,
        }

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        Args:
            obj (Dict[str, Any]): The state.
        """
        self._arr[self._epoch_slot] = obj['epoch']
        sessions = obj['sessions']
        self._arr[self._num_sessions_slot] = len(sessions)
        index = self._sessions_slot
        for session in sessions:
            self._arr[index] = num_workers = len(session)
            index += 1
            self._arr[index:index + num_workers] = session
            index += num_workers

        self._broadcast_state()
