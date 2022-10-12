# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Primitives for synchronizing processes on the same node."""

import os
from multiprocessing.shared_memory import SharedMemory
from time import sleep

import numpy as np


class Lock:
    """A lock that works inter-process using directory creation.

    Args:
        dirname (str): Dirname that backs all instances of this lock across all processes.
        poll_interval (float): A short unit of time for spinlocking. Default: ``0.007``.
    """

    def __init__(self, dirname: str, poll_interval: float = 0.007) -> None:
        self.dirname = dirname
        self.poll_interval = poll_interval

        # Initially not locked.
        assert not os.path.exists(dirname)

    def acquire(self) -> None:
        """Acquire the lock."""
        while True:
            try:
                os.makedirs(self.dirname)
                break
            except FileExistsError:
                sleep(self.poll_interval)

    def release(self) -> None:
        """Release the lock."""
        os.rmdir(self.dirname)


class Barrier:
    """A barrier that works inter-process using a file lock and shared memory.

    Args:
        num_procs (int): Number of processes to synchronize.
        name (str): Unique name for both the shared memory and file lock backing this object.
        lock_dir (str): Directory where file locks are stored. Default: `/tmp/`.
        poll_interval (float): A short unit of time for spinlocking. Default: ``0.007``.
    """

    # The fields of the shared memory array.
    _num_enter_slot = 0
    _num_exit_slot = 1
    _flag_slot = 2

    def __init__(self,
                 num_procs: int,
                 name: str,
                 lock_dir: str = '/tmp/',
                 poll_interval: float = 0.007) -> None:
        self.num_procs = num_procs
        self.name = name
        self.lock_dir = lock_dir
        self.poll_interval = poll_interval

        # File lock.
        self.lock = Lock(name, poll_interval)

        # Shared memory bytes.
        size = 3 * np.int32().nbytes
        try:
            self._shm = SharedMemory(name, True, size)
        except:
            self._shm = SharedMemory(name, False, size)

        # Array backed by shared memory: num_enter, num_exit, flag.
        self._arr = np.frombuffer(self._shm.buf, np.int32)
        self._arr[self._num_enter_slot] = 0
        self._arr[self._num_exit_slot] = num_procs
        self._arr[self._flag_slot] = True

    def __del__(self):
        """Destructor clears array that lives in shm."""
        del self._arr

    @property
    def num_enter(self) -> int:
        """Get property num_enter.

        Returns:
            int: Number of processes that have entered the barrier.
        """
        return int(self._arr[0])

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
        return int(self._arr[1])

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
            while self.num_exit != self.num_procs:
                sleep(self.poll_interval)
            self.lock.acquire()
            self.flag = False

        # Note that we entered.
        self.num_enter += 1

        # If we are the last to arrive, reset `enter` and `exit`, and set flag to "go".
        if self.num_enter == self.num_procs:
            self.num_enter = 0
            self.num_exit = 0
            self.flag = True
        self.lock.release()

        # Everybody waits until the flag is set to "go".
        while not self.flag:
            sleep(self.poll_interval)

        # Note that we exited.
        self.lock.acquire()
        self.num_exit += 1
        self.lock.release()
