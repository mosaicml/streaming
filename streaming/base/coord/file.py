# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Soft file locking via file open mode 'x'."""

import os
from time import sleep, time
from types import TracebackType
from typing import Optional, Type, Union

from typing_extensions import Self

from streaming.base.coord.process import get_live_processes

__all__ = ['SoftFileLock']


class SoftFileLock:
    """Soft file locking via file open mode 'x'.

    Args:
        filename (str): Path to lock.
        timeout (float, optional): How long to wait in seconds before raising an exception.
            Set to ``None`` to never time out. Defaults to ``30``.
        tick (float): Polling interval in seconds. Defaults to ``0.007``.
    """

    def __init__(self, filename: str, timeout: Optional[float] = 30, tick: float = 0.007) -> None:
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

        self._normalize(filename)

    @classmethod
    def _write(cls, filename: str, pid: int) -> None:
        """Write the locking process's pid.

        Args:
            filename (str): Path to lock.
        """
        with open(filename, 'x') as file:
            file.write(str(pid))

    @classmethod
    def _read(cls, filename: str) -> int:
        """Read the locking process's pid.

        Args:
            filename (str): Path to lock.
        """
        with open(filename, 'r') as file:
            return int(file.read())

    @classmethod
    def _normalize(cls, filename: str) -> None:
        """Ensure parent dirs exist and lock files held by dead processes do not exist.

        Args:
            filename (str): Path to lock.
        """
        # Ensure the file's parent directory exists so we can write it in one shot.
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # If no file, we don't need to do anything.
        if not os.path.exists(filename):
            return

        # If we fail to open the file and parse the pid, bail out while deleting it.
        try:
            pid = cls._read(filename)
        except:
            os.remove(filename)
            return

        # If the pid is not among the living, delete the file.
        if pid not in get_live_processes():
            os.remove(filename)

    @classmethod
    def _get_timeout(cls,
                     init_timeout: Optional[float],
                     timeout: Optional[Union[str, float]] = 'auto') -> Optional[float]:
        """Determine the timeout for a given acquire().

        Args:
            init_timeout (float, optional): Default timeout provided to init.
            timeout (str | float, optional): Override timeout for just this method call.

        Returns:
            float, optional: Normalized timeout as positive float seconds or ``None`` to disable.
        """
        if timeout is None:
            # No timeout.
            ret = timeout
        elif isinstance(timeout, float):
            # Override timeout.
            if timeout <= 0:
                raise ValueError(
                    f'Timeout must be positive float seconds, but got: {timeout} sec.')
            ret = timeout
        elif timeout == 'auto':
            # Default timeout.
            ret = init_timeout
        else:
            raise ValueError(f'Timeout must either be positive float seconds, ``None`` to ' +
                             f'disable timing out, or ``auto`` to use the default passed to ' +
                             f'init, but got: {timeout}.')
        return ret

    def acquire(self, timeout: Optional[Union[str, float]] = 'auto') -> None:
        """Acquire this lock.

        Args:
            timeout (str | float, optional): Override timeout for just this method call.
        """
        start = time()
        timeout = self._get_timeout(self.timeout, timeout)
        while True:
            try:
                self._write(self.filename, os.getpid())
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
            Self: This lock.
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
