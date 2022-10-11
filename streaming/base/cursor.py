# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Tracks distributed progress through a streaming dataset for checkpointing."""

from multiprocessing.shared_memory import SharedMemory
from time import sleep
from typing import Any, Dict, Iterator, Optional

import numpy as np
from numpy.typing import NDArray

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
        size = 1 << 16
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
            return None

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
        slot = self._get_current_session_slot()
        assert slot
        self._arr[slot] = num_workers = dist.get_num_workers()
        return num_workers + 1 + dist.get_worker()

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

    def state_dict(self) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        Returns:
            Dict[str, Any]: The state.
        """
        # TODO: all_gather state across nodes.
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
        # TODO: broadcast state across nodes.
