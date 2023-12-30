# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Soft file locking via file open mode 'x'."""

import os
from time import sleep, time
from types import TracebackType
from typing import Optional, Type

from typing_extensions import Self

from streaming.base.coord.process import get_live_processes

__all__ = ['SoftFileLock']


class SoftFileLock:
    """Soft file locking via file open mode 'x'.

    Args:
        filename (str): Path to lock.
        timeout (float, optional): How long to wait in seconds before raising an exception.
            Defaults to ``10``.
        tick (float): Polling interval in seconds. Defaults to ``0.007``.
    """

    def __init__(self, filename: str, timeout: Optional[float] = 10, tick: float = 0.007) -> None:
        if not filename:
            raise ValueError('Path to file lock is empty.')

        if timeout is not None:
            if timeout <= 0:
                raise ValueError(
                    f'Timeout must be positive float seconds, but got: {timeout} sec.')

        if tick <= 0:
            raise ValueError(f'Tick must be positive float seconds, but got: {tick} sec.')

        self.filename = filename
        self.timeout = timeout
        self.tick = tick

        self._garbage_collect(filename)

        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)

    @classmethod
    def _set_pid(cls, filename: str, pid: int) -> None:
        """Set the locking process's pid.

        Args:
            filename (str): Path to lock.
        """
        with open(filename, 'x') as file:
            file.write(str(pid))

    @classmethod
    def _get_pid(cls, filename: str) -> int:
        """Get the locking process's pid.

        Args:
            filename (str): Path to lock.
        """
        with open(filename, 'r') as file:
            text = file.read()
        return int(text)

    @classmethod
    def _try_remove(cls, filename: str) -> None:
        """Try to remove this lock.

        Args:
            filename (str): Path to lock.
        """
        try:
            os.remove(filename)
        except:
            pass

    @classmethod
    def _garbage_collect(cls, filename: str) -> None:
        """Release this lock if held by a dead process.

        Args:
            filename (str): Path to lock.
        """
        try:
            pid = cls._get_pid(filename)
        except:
            cls._try_remove(filename)
            return

        if pid not in get_live_processes():
            cls._try_remove(filename)

    def acquire(self) -> None:
        """Acquire this lock."""
        start = time()

        while True:
            try:
                self._set_pid(self.filename, os.getpid())
                break
            except:
                pass

            if self.timeout is not None and start + self.timeout < time():
                raise ValueError(f'Timed out while attempting to acquire file lock: file ' +
                                 f'{self.filename}, timeout {self.timeout} sec.')

            sleep(self.tick)

    def release(self) -> None:
        """Release this lock."""
        if os.path.isfile(self.filename):
            os.remove(self.filename)
        elif os.path.exists(self.filename):
            raise ValueError(f'Path exists, but is not a file: {self.filename}.')
        else:
            raise ValueError(f'Path does not exist: {self.filename}.')

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns:
            Self: This object.
        """
        self.acquire()
        return self

    def __exit__(self,
                 err_type: Optional[Type[BaseException]] = None,
                 err: Optional[BaseException] = None,
                 trace: Optional[TracebackType] = None) -> None:
        """Exit context manager.

        Args:
            err_type (Type[BaseException], optional): Exc type.
            err (BaseException, optional): Exc.
            trace (TracebackType, optional): Traceback.
        """
        self.release()
