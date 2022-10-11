# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Primitives for synchronizing processes on the same node."""

import os
from multiprocessing.shared_memory import SharedMemory
from time import sleep

import numpy as np

IOTA = 0.007


class Lock:
    """A lock that works inter-process using directory creation.

    Args:
        dirname (str): Dirname that backs all instances of this lock across all processes.
    """

    def __init__(self, dirname: str) -> None:
        self.dirname = dirname

    def acquire(self) -> None:
        """Acquire the lock."""
        while True:
            try:
                os.makedirs(self.dirname)
                break
            except FileExistsError:
                sleep(IOTA)

    def release(self) -> None:
        """Release the lock."""
        os.rmdir(self.dirname)


COUNT = 0
ENTER = 1
EXIT = 2
GO = 3


class Barrier:
    """A barrier that works inter-process using a file lock and shared memory.

    Args:
        count (int): Number of processes to synchronize.
        name (str): Unique name of the shared memory and file lock backing this object.
    """

    def __init__(self, count: int, name: str) -> None:
        self.lock = Lock(name)

        # Fields: count, enter, exit, flag.
        size = 4 * np.int32().nbytes
        try:
            self.shm = SharedMemory(name, True, size)
        except:
            self.shm = SharedMemory(name, False, size)
        self._arr = np.frombuffer(self.shm.buf, np.int32)
        self._arr[:] = count, 0, count, True  # Count, enter, exit, go.

    def __call__(self) -> None:
        """However many processes enter, wait, and exit the barrier."""
        # If we are the first to arrive, wait for everyone to exit, then set flag to "don't go".
        self.lock.acquire()
        if not self._arr[ENTER]:
            self.lock.release()
            while self._arr[EXIT] != self._arr[COUNT]:
                sleep(IOTA)
            self.lock.acquire()
            self._arr[GO] = False

        # Note that we entered.
        self._arr[ENTER] += 1

        # If we are the last to arrive, reset `enter` and `exit`, and set flag to "go".
        if self._arr[ENTER] == self._arr[COUNT]:
            self._arr[ENTER] = 0
            self._arr[EXIT] = 0
            self._arr[GO] = True
        self.lock.release()

        # Everybody waits until the flag is set to "go".
        while not self._arr[GO]:
            sleep(IOTA)

        # Note that we exited.
        self.lock.acquire()
        self._arr[EXIT] += 1
        self.lock.release()
