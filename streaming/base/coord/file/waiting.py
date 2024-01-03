# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Waiting on files."""

import os
from typing import Any, Optional

from streaming.base.coord.waiting import wait

__all__ = ['wait_for_creation', 'wait_for_deletion', 'create_file']


def wait_for_creation(
    path: str,
    timeout: Optional[float] = 30,
    tick: float = 0.007,
    lock: Optional[Any] = None,
) -> None:
    """Wait for the creation of a path on the local filesystem.

    Args:
        path (str): Local path to wait on the creation of.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.
        lock (Any, optional): Context manager (this is intended for locks) to be held when
            checking the predicate. Defaults to ``None``.
    """

    def stop():
        return os.path.exists(path)

    wait(stop, timeout, tick, lock)


def wait_for_deletion(
    path: str,
    timeout: Optional[float] = 30,
    tick: float = 0.007,
    lock: Optional[Any] = None,
) -> None:
    """Wait for the deletion of a path on the local filesystem.

    Args:
        path (str): Local path to wait on the deletion of.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.
        lock (Any, optional): Context manager (this is intended for locks) to be held when
            checking the predicate. Defaults to ``None``.
    """

    def stop():
        return not os.path.exists(path)

    wait(stop, timeout, tick, lock)


def create_file(filename: str) -> None:
    """Create a file at the given path on the local filesystem.

    Raises an exception if the path already exists.

    Args:
        filename (str): Filename to create.
    """
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'x'):
        pass
