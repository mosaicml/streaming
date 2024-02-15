# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Waiting on predicates."""

from contextlib import nullcontext
from time import sleep, time
from typing import Any, Callable, Optional

__all__ = ['wait']


def _say_duration(duration: float) -> str:
    """Pretty-print a duration.

    Args:
        duration (float): The duration as a float.

    Returns:
        str: The duration as a str.
    """
    return f'{duration:.3f}'.rstrip('0').rstrip('.')


def wait(
    stop: Callable[[], bool],
    timeout: Optional[float] = 30,
    tick: float = 0.007,
    lock: Optional[Any] = None,
) -> None:
    """Wait for the predicate to succeed.

    Args:
        stop (Callable[[], bool]): When this check returns True, you break out of the retry loop.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.
        lock (Any, optional): Context manager (this is intended for locks) to be held when
            checking the predicate. Defaults to ``None``.
    """
    start = time()

    if timeout is not None and timeout <= 0:
        raise ValueError(f'Timeout must be positive if provided, but got: ' +
                         f'{_say_duration(timeout)} sec.')

    if tick <= 0:
        raise ValueError(f'Tick must be positive if provided, but got: {_say_duration(tick)} sec.')

    if lock is not None:
        if not hasattr(lock, '__enter__'):
            raise ValueError(f'Lock must support `__enter__`, but got: {lock}.')

        if not hasattr(lock, '__exit__'):
            raise ValueError(f'Lock must support `__exit__`, but got: {lock}.')

        norm_lock = lock
    else:
        norm_lock = nullcontext()

    while True:
        with norm_lock:
            if stop():
                break

        if timeout is not None:
            now = time()
            if timeout <= now - start:
                raise RuntimeError(f'Wait timed out: timeout {_say_duration(timeout)} sec vs ' +
                                   f'elapsed {_say_duration(now - start)} sec.')

        sleep(tick)
