# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share a barrier across processes using mmap()."""

from enum import IntEnum
from typing import Optional

import numpy as np

from streaming.base.coord.file import SoftFileLock
from streaming.base.coord.mmap.ndarray import MemMapNDArray
from streaming.base.coord.waiting import wait

__all__ = ['MemMapBarrier', 'barrier']


class BarrierFlag(IntEnum):
    """Whether to stop or go, used internally by MemMapBarrier."""

    STOP = 0
    GO = 1


class MemMapBarrier:
    """A barrier backed by a memory-mapped file and a file lock.

    Args:
        create (bool): If ``True``, create. If ``False``, attach.
        mmap_filename (str): Path to memory-mapped file.
        lock_filename (str): Path to SoftFileLock file.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.
    """

    def __init__(
        self,
        create: bool,
        mmap_filename: str,
        lock_filename: str,
        timeout: Optional[float] = 30,
        tick: float = 0.007,
    ) -> None:
        value = 0 if create else None
        self._arr = MemMapNDArray(mmap_filename, 3, np.int32, value)
        self._num_enter = 0
        self._num_exit = -1
        self._flag = BarrierFlag.GO
        self._lock = SoftFileLock(lock_filename)
        self._timeout = timeout
        self._tick = tick

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
        self._arr[0] = num_enter

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
        self._arr[1] = num_exit

    @property
    def _flag(self) -> BarrierFlag:
        """Getter for _flag.

        Returns:
            BarrierFlat: Flag value.
        """
        return BarrierFlag(self._arr[2])

    @_flag.setter
    def _flag(self, flag: BarrierFlag) -> None:
        """Setter for _flag.

        Args:
            flag (BarrierFlag): Flag value.
        """
        self._arr[2] = flag

    def __call__(self, total: int) -> None:
        """A set number of processes enter, wait, and exit the barrier.

        Args:
            num_procs (int): How many processes are sharing this barrier.
        """
        # Initialize `_num_exit` to the number of processes.
        with self._lock:
            if self._num_exit == -1:
                self._num_exit = total

        # If we are the first to arrive, wait for everyone to exit, then set `_flag` to `STOP`.
        self._lock.acquire()
        if not self._num_enter:
            self._lock.release()
            wait(lambda: self._num_exit == total, self._timeout, self._tick, self._lock)
            self._lock.acquire()
            self._flag = BarrierFlag.STOP

        # Note that we entered.z
        self._num_enter += 1

        # If we are the last to arrive, reset `_enter` and `_exit`, and set `_flag` to `GO`.
        if self._num_enter == total:
            self._num_enter = 0
            self._num_exit = 0
            self._flag = BarrierFlag.GO
        self._lock.release()

        # Everybody waits until `_flag` is set to `GO`.
        wait(lambda: self._flag == BarrierFlag.GO, self._timeout, self._tick, self._lock)

        # Note that we exited.
        with self._lock:
            self._num_exit += 1
            if self._num_exit == total:
                self._num_exit = -1


def barrier(
    create: bool,
    mmap_filename: str,
    lock_filename: str,
    timeout: Optional[float] = 30,
    tick: float = 0.007,
) -> MemMapBarrier:
    """Get a barrier backed by a memory-mapped file and a file lock.

    Args:
        create (bool): If ``True``, create. If ``False``, attach.
        mmap_filename (str): Path to memory-mapped file.
        lock_filename (str): Path to SoftFileLock file.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.
    """
    return MemMapBarrier(create, mmap_filename, lock_filename, timeout, tick)
