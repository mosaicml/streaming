# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""File locking via file open mode 'x'."""

import os
from contextlib import contextmanager
from time import sleep, time
from typing import Iterator, Optional

__all__ = ['acquire_file_lock', 'release_file_lock', 'file_lock']


def acquire_file_lock(filename: str, timeout: Optional[float] = 60, tick: float = 0.007) -> float:
    """Acquire a file lock."""
    start = time()

    if timeout is not None:
        if timeout <= 0:
            raise ValueError(f'Timeout must be positive float seconds, but got: {timeout} sec.')

    if tick <= 0:
        raise ValueError(f'Tick must be positive float seconds, but got: {tick} sec.')

    while True:
        try:
            text = str(os.getpid())
            with open(filename, 'x') as file:
                file.write(text)
            break
        except:
            pass

        try:
            with open(filename, 'r') as file:
                text = file.read()
            pid = int(text)

            if pid != os.getpid():
                os.remove(filename)
                continue
        except:
            pass

        if timeout is not None and start + timeout < time():
            raise ValueError(f'Timed out while attempting to acquire file lock: file ' +
                             f'{filename}, timeout {timeout} sec.')

        sleep(tick)

    return time() - start


def release_file_lock(filename: str) -> None:
    """Release a file lock."""
    if os.path.exists(filename):
        if os.path.isfile(filename):
            os.remove(filename)
        else:
            raise ValueError(f'Path exists, but is not a file: {filename}.')
    else:
        raise ValueError(f'Path does not exist: {filename}.')


@contextmanager
def file_lock(filename: str, timeout: Optional[float] = 60, tick: float = 0.007) -> Iterator[None]:
    """A with statement protected by a file lock."""
    acquire_file_lock(filename, timeout, tick)

    yield

    release_file_lock(filename)
