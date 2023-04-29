# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A pthread lock shared between processes that lives in shared memory."""

from types import TracebackType
from typing import Optional

from streaming.base.shared.memory import SharedMemory
from streaming.cpp.shared import locking


class SharedLock:
    """A pthread lock shared between processes that lives in shared memory.

    Like a FileLock, but doesn't leave filesystem crud, and is 25x faster.

    Args:
        name (str): Shared memory path where this object is found.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.size = locking.size()
        self.shm = SharedMemory(name=name, size=self.size)
        locking.create(self.shm.buf)

    def acquire(self) -> None:
        """Acquire the lock."""
        locking.acquire(self.shm.buf)

    def release(self) -> None:
        """Release the lock."""
        locking.release(self.shm.buf)

    def __enter__(self) -> None:
        """Enter a with statement."""
        self.acquire()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit a with statement.

        Args:
            exc_type (type[BaseException], optional): exc_type.
            exc_val (BaseException, optional): exc_val.
            exc_tb (TracebackType, optional): exc_tb.
        """
        self.release()

    def __del__(self) -> None:
        """Destructor."""
        locking.destroy(self.shm.buf)
        # self.shm.cleanup()
