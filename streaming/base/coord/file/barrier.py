# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for doing barrier-like work with files."""

import os
from contextlib import nullcontext
from time import sleep, time
from typing import Any, Optional

__all__ = ['wait_for_creation', 'wait_for_deletion', 'create_file']


def _say_duration(duration: float) -> str:
    """Pretty-print a duration.

    Args:
        duration (float): The duration as a float.

    Returns:
        str: The duration as a str.
    """
    return f'{duration:.3f}'.rstrip('0').rstrip('.')


def _wait_for_path(
    path: str,
    exists: bool,
    timeout: Optional[float] = 30,
    poll_interval: float = 0.007,
    lock: Optional[Any] = None,
) -> None:
    """Wait for the creation or deletion of a path on the local filesystem.

    Args:
        path (str): Local path to wait on.
        exists (bool): Wait for existence if ``True`` or non-existence if ``False``.
        timeout (float, optional): How long to wait before raising an error, in seconds. Defaults
            to ``60``.
        poll_interval (float): Check interval, in seconds. Defaults to ``0.007``.
        lock (Any, optional): Lock that must be held when probing for the path. Defaults to
            ``None``.
    """
    start = time()

    if timeout is not None and timeout <= 0:
        raise ValueError(f'Timeout must be positive if provided, but got: ' +
                         f'{_say_duration(timeout)} sec.')

    if poll_interval <= 0:
        raise ValueError(f'Poll interval must be positive if provided, but got: ' +
                         f'{_say_duration(poll_interval)} sec.')

    if lock is not None:
        if not hasattr(lock, '__enter__'):
            raise ValueError(f'Lock must support `__enter__`, but got: {lock}.')

        if not hasattr(lock, '__exit__'):
            raise ValueError(f'Lock must support `__exit__`, but got: {lock}.')

    while True:
        with lock or nullcontext():
            if os.path.exists(path) == exists:
                break

        if timeout is not None:
            now = time()
            if timeout <= now - start:
                raise RuntimeError(f'Timed out while waiting for path to exist: path {path}, ' +
                                   f'timeout {_say_duration(timeout)} sec, elapsed ' +
                                   f'{_say_duration(now - start)} sec.')

        sleep(poll_interval)


def wait_for_creation(
    path: str,
    timeout: Optional[float] = 30,
    poll_interval: float = 0.007,
    lock: Optional[Any] = None,
) -> None:
    """Wait for the creation of a path on the local filesystem.

    Args:
        path (str): Local path to wait on.
        timeout (float, optional): How long to wait before raising an error, in seconds. Defaults
            to ``60``.
        poll_interval (float): Check interval, in seconds. Defaults to ``0.007``.
        lock (Any): Lock that must be held when probing for the path.
    """
    _wait_for_path(path, True, timeout, poll_interval, lock)


def wait_for_deletion(path: str,
                      timeout: Optional[float] = 30,
                      poll_interval: float = 0.007,
                      lock: Optional[Any] = None) -> None:
    """Wait for the deletion of a path on the local filesystem.

    Args:
        path (str): Local path to wait on.
        timeout (float, optional): How long to wait before raising an error, in seconds. Defaults
            to ``60``.
        poll_interval (float): Check interval, in seconds. Defaults to ``0.007``.
        lock (Any): Lock that must be held when probing for the path.
    """
    _wait_for_path(path, False, timeout, poll_interval, lock)


def create_file(filename: str) -> None:
    """Create a file at the given path on the local filesystem.

    Raises an exception if the path already exists.

    Args:
        filename (str): Filename to create.
    """
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    file = open(filename, 'x')
    file.close()
