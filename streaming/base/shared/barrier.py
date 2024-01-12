# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Barrier that lives in shared memory.

Implemented with shared array and a filelock.
"""

from time import sleep

import numpy as np
from filelock import FileLock

from streaming.base.constant import TICK
from streaming.base.shared.array import SharedArray

# Time out to wait before raising exception
TIMEOUT = 60


class SharedBarrier:
    """A barrier that works inter-process using a filelock and shared memory.

    We set the number of processes (and thereby initialize num_exit) on the first time this object
    is called. This is because the object is created in a per-rank process, and called by worker
    processes.

    Args:
        filelock_path (str): Path to lock file on local filesystem.
        shm_name (str): Shared memory object name in /dev/shm.
    """

    def __init__(self, filelock_path: str, shm_name: str) -> None:
        # Create lock.
        self.filelock_path = filelock_path
        self.lock = FileLock(self.filelock_path)

        # Create three int32 fields in shared memory: num_enter, num_exit, flag.
        self._arr = SharedArray(3, np.int32, shm_name)
        self.num_enter = 0
        self.num_exit = -1
        self.flag = True

    @property
    def num_enter(self) -> int:
        """Get property num_enter.

        Returns:
            int: Number of processes that have entered the barrier.
        """
        return self._arr[0]

    @num_enter.setter
    def num_enter(self, num_enter: int) -> None:
        """Set property num_enter.

        Args:
            num_enter (int): Number of processes that have entered the barrier.
        """
        self._arr[0] = num_enter

    @property
    def num_exit(self) -> int:
        """Get property num_exit.

        Returns:
            int: Number of processes that have exited the barrier.
        """
        return self._arr[1]

    @num_exit.setter
    def num_exit(self, num_exit: int) -> None:
        """Set property num_exit.

        Args:
            num_exit (int): Number of processes that have exited the barrier.
        """
        self._arr[1] = num_exit

    @property
    def flag(self) -> bool:
        """Get property flag.

        Returns:
            bool: The flag value.
        """
        return bool(self._arr[2])

    @flag.setter
    def flag(self, flag: bool) -> None:
        """Set property flag.

        Args:
            flag (bool): The flag value.
        """
        self._arr[2] = bool(flag)

    def __call__(self, num_procs: int) -> None:
        """A set number of processes enter, wait, and exit the barrier.

        Args:
            num_procs (int): How many processes are sharing this barrier.
        """
        # Initialize num_exit to the number of processes.
        with self.lock:
            if self.num_exit == -1:
                self.num_exit = num_procs

        # If we are the first to arrive, wait for everyone to exit, then set flag to "don't go".
        self.lock.acquire()
        if not self.num_enter:
            self.lock.release()
            while self.num_exit != num_procs:
                sleep(TICK)
            self.lock.acquire()
            self.flag = False

        # Note that we entered.
        self.num_enter += 1

        # If we are the last to arrive, reset `enter` and `exit`, and set flag to "go".
        if self.num_enter == num_procs:
            self.num_enter = 0
            self.num_exit = 0
            self.flag = True
        self.lock.release()

        # Everybody waits until the flag is set to "go".
        while not self.flag:
            sleep(TICK)

        # Note that we exited.
        with self.lock:
            self.num_exit += 1
            if self.num_exit == num_procs:
                self.num_exit = -1
