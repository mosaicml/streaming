# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Synchronization primitives that live in shared memory.

For when using `threading` or `multiprocessing` from the python standard library won't do, because
we are coordinating separately instantiated pytorch worker processes.
"""

import os
from multiprocessing.shared_memory import SharedMemory
from time import sleep

import numpy as np
from filelock import FileLock

TICK = 0.007


class SharedBarrier:
    """A barrier that works inter-process using a file lock and shared memory.

    Args:
        count (int): Number of processes to synchronize.
        filelock_path (str): Path to lock file on local filesystem.
        shm_path (str): Shared memory object name in /dev/shm.
    """

    def __init__(self, count: int, filelock_path: str, shm_path: str) -> None:
        self.count = count
        self.filelock_path = filelock_path
        self.shm_path = shm_path

        # Create filelock.
        dirname = os.path.dirname(filelock_path)
        os.makedirs(dirname, exist_ok=True)
        self.lock = FileLock(filelock_path)

        # Create three int32 fields in shared memory: num_enter, num_exit, flag.
        size = 3 * np.int32().nbytes
        try:
            self._shm = SharedMemory(shm_path, True, size)
        except FileExistsError:
            self._shm = SharedMemory(shm_path, False, size)
        self._arr = np.ndarray(3, buffer=self._shm.buf, dtype=np.int32)
        self.num_enter = 0
        self.num_exit = count
        self.flag = True

    def __del__(self):
        """Destructor clears array that references shm."""
        try:
            del self._arr
        except:
            pass

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

    def __call__(self) -> None:
        """A set number of processes enter, wait, and exit the barrier."""
        # If we are the first to arrive, wait for everyone to exit, then set flag to "don't go".
        self.lock.acquire()
        if not self.num_enter:
            self.lock.release()
            while self.num_exit != self.count:
                sleep(TICK)
            self.lock.acquire()
            self.flag = False

        # Note that we entered.
        self.num_enter += 1

        # If we are the last to arrive, reset `enter` and `exit`, and set flag to "go".
        if self.num_enter == self.count:
            self.num_enter = 0
            self.num_exit = 0
            self.flag = True
        self.lock.release()

        # Everybody waits until the flag is set to "go".
        while not self.flag:
            sleep(TICK)

        # Note that we exited.
        self.lock.acquire()
        self.num_exit += 1
        self.lock.release()
