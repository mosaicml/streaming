# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share a barrier across processes using mmap()."""

import os
from time import sleep

import numpy as np
from filelock import FileLock
from typing_extensions import Self

from streaming.base.coord.mmap.array import MMapArray

__all__ = ['MMapBarrier']


class MMapBarrier:
    """Share a barrier across processes using mmap().

    Args:
        arr_filename (str): File backing the internal MMapArray.
        lock_filename (str): File backing the internal FileLock.
        tick (float): Polling interval in seconds. Defaults to ``0.007``.
    """

    def __init__(self, arr_filename: str, lock_filename: str, tick: float = 0.007) -> None:
        self._arr_filename = arr_filename
        self._lock_filename = lock_filename
        self._tick = tick

        self._arr = MMapArray(arr_filename, np.int32(), 3)

        self._num_enter = 0
        self._num_exit = -1
        self._flag = True

    @property
    def _num_enter(self) -> int:
        """Getter for _num_enter.

        Returns:
            int: Entered process count.
        """
        return int(self._arr[0])

    @_num_enter.setter
    def _num_enter(self, num_enter: int) -> None:
        """Setter for _num_enter.

        Args:
            num_enter (int): Entered process count.
        """
        self._arr[0] = np.int32(num_enter)

    @property
    def _num_exit(self) -> int:
        """Getter for _num_exit.

        Returns:
            int: Exited process count.
        """
        return int(self._arr[1])

    @_num_exit.setter
    def _num_exit(self, num_exit: int) -> None:
        """Setter for _num_exit.

        Args:
            num_exit (int): Exited process count.
        """
        self._arr[1] = np.int32(num_exit)

    @property
    def _flag(self) -> bool:
        """Getter for _flag.

        Returns:
            bool: Flag value.
        """
        return bool(self._arr[2])

    @_flag.setter
    def _flag(self, flag: bool) -> None:
        """Setter for _flag.

        Args:
            flag (bool): Flag value.
        """
        self._arr[2] = np.int32(flag)

    @classmethod
    def create(cls, arr_filename: str, lock_filename: str, tick: float = 0.007) -> Self:
        """Create and load an MMapBarrier from scratch.

        Args:
            arr_filename (str): File backing the MMapArray.
            lock_filename (str): File bcking the FileLock.
            tick (float): Polling interval in seconds. Defaults to ``0.007``.
        """
        if os.path.exists(arr_filename):
            raise ValueError('File already exists: {arr_filename}.')

        MMapArray._write(arr_filename, np.int32(), 3)
        return cls(arr_filename, lock_filename, tick)

    def __call__(self, total: int) -> None:
        lock = FileLock(self._lock_filename)

        # Initialize num_exit to the number of processes.
        with lock:
            if self._num_exit == -1:
                self._num_exit = total

        # If we are the first to arrive, wait for everyone to exit, then set flag to "don't go".
        lock.acquire()
        if not self._num_enter:
            lock.release()
            while self._num_exit != total:
                sleep(self._tick)
            lock.acquire()
            self._flag = False

        # Note that we entered.
        self._num_enter += 1

        # If we are the last to arrive, reset `enter` and `exit`, and set flag to "go".
        if self._num_enter == total:
            self._num_enter = 0
            self._num_exit = 0
            self._flag = True
        lock.release()

        # Everybody waits until the flag is set to "go".
        while not self._flag:
            sleep(self._tick)

        # Note that we exited.
        with lock:
            self._num_exit += 1
            if self._num_exit == total:
                self._num_exit = -1
