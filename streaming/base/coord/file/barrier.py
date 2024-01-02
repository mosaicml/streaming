# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for doing barrier-like work with files."""

import os
from time import sleep, time
from typing import Optional

__all__ = ['wait_for_creation', 'wait_for_deletion', 'create_file']


def _wait_for_path(path: str,
                   exist: bool,
                   timeout: Optional[float] = 60,
                   poll_interval: float = 0.007) -> None:
    """Wait for the creation or deletion of a path on the local filesystem.

    Args:
        path (str): Local path to wait on.
        exist (bool): Wait for existence if ``True`` or non-existence if ``False``.
        timeout (float, optional): How long to wait before raising an error, in seconds. Defaults
            to ``60``.
        poll_interval (float): Check interval, in seconds. Defaults to ``0.007``.
    """
    if timeout is not None and timeout <= 0:
        timeout_str = f'{timeout:.3f}'.rstrip('0').rstrip('.')
        raise ValueError(f'Timeout must be positive if provided, but got: {timeout_str}.')

    if poll_interval <= 0:
        poll_interval_str = f'{poll_interval:.3f}'.rstrip('0').rstrip('.')
        raise ValueError(f'Poll interval must be positive if provided, but got: ' +
                         f'{poll_interval_str}.')

    start = time()
    while True:
        if os.path.exists(path) == exist:
            break

        if timeout is not None:
            elapsed = time() - start
            if timeout <= elapsed:
                timeout_str = f'{timeout:.3f}'.rstrip('0').rstrip('.')
                elapsed_str = f'{elapsed:.3f}'.rstrip('0').rstrip('.')
                raise RuntimeError(f'Timed out while waiting for path to exist: path {path}, ' +
                                   f'timeout {timeout_str} sec, elapsed {elapsed_str} sec.')

        sleep(poll_interval)


def wait_for_creation(path: str,
                      timeout: Optional[float] = 60,
                      poll_interval: float = 0.007) -> None:
    """Wait for the creation of a path on the local filesystem.

    Args:
        path (str): Local path to wait on.
        timeout (float, optional): How long to wait before raising an error, in seconds. Defaults
            to ``60``.
        poll_interval (float): Check interval, in seconds. Defaults to ``0.007``.
    """
    _wait_for_path(path, True, timeout, poll_interval)


def wait_for_deletion(path: str,
                      timeout: Optional[float] = 60,
                      poll_interval: float = 0.007) -> None:
    """Wait for the deletion of a path on the local filesystem.

    Args:
        path (str): Local path to wait on.
        timeout (float, optional): How long to wait before raising an error, in seconds. Defaults
            to ``60``.
        poll_interval (float): Check interval, in seconds. Defaults to ``0.007``.
    """
    _wait_for_path(path, False, timeout, poll_interval)


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
