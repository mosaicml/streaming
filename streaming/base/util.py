# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for datasets."""

import collections.abc
import functools
import logging
import os
import random
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from time import sleep, time
from typing import Any, Callable, List, Sequence, Type, TypeVar, Union, cast, overload

import torch.distributed as dist

from streaming.base.constant import SHM_TO_CLEAN
from streaming.base.distributed import get_local_rank, maybe_init_dist
from streaming.base.shared.prefix import _get_path

logger = logging.getLogger(__name__)

TCallable = TypeVar('TCallable', bound=Callable)

__all__ = [
    'get_list_arg', 'wait_for_file_to_exist', 'bytes_to_int', 'number_abbrev_to_int',
    'clean_stale_shared_memory', 'get_import_exception_message', 'retry'
]


def get_list_arg(text: str) -> List[str]:
    """Pass a list as a command-line flag.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: Splits, if any.
    """
    return text.split(',') if text else []


def wait_for_file_to_exist(filename: str, poll_interval: float, timeout: float,
                           err_msg: str) -> None:
    """Wait for the file to exist till timeout seconds. Raise an Exception after that.

    Args:
        filename (str): A file name
        poll_interval (float): Number of seconds to wait before next polling
        timeout (float): Number of seconds to wait for a file to exist before raising an exception
        err_msg (str): Error message description for an exception

    Raises:
        RuntimeError: Raise an Exception if file does not exist after timeout
    """
    start_time = time()
    while True:
        sleep(poll_interval)
        if os.path.exists(filename):
            sleep(poll_interval)
            break
        dt = time() - start_time
        if dt > timeout:
            raise RuntimeError(f'{err_msg}' + f'{timeout:.3f} < {dt:.3f} secs.')


def bytes_to_int(bytes_str: Union[int, str]) -> int:
    """Convert human readable byte format to an integer.

    Args:
        bytes_str (Union[int, str]): Value to convert.

    Raises:
        ValueError: Invalid byte suffix.

    Returns:
        int: Integer value of bytes.
    """
    #input is already an int
    if isinstance(bytes_str, int) or isinstance(bytes_str, float):
        return int(bytes_str)

    units = {
        'kb': 1024,
        'mb': 1024**2,
        'gb': 1024**3,
        'tb': 1024**4,
        'pb': 1024**5,
        'eb': 1024**6,
        'zb': 1024**7,
        'yb': 1024**8,
    }
    # Convert a various byte types to an integer
    for suffix in units:
        bytes_str = bytes_str.lower().strip()
        if bytes_str.lower().endswith(suffix):
            try:
                return int(float(bytes_str[0:-len(suffix)]) * units[suffix])
            except ValueError:
                raise ValueError(''.join([
                    f'Unsupported value/suffix {bytes_str}. Supported suffix are ',
                    f'{["b"] + list(units.keys())}.'
                ]))
    else:
        # Convert bytes to an integer
        if bytes_str.endswith('b') and bytes_str[0:-1].isdigit():
            return int(bytes_str[0:-1])
        # Convert string representation of a number to an integer
        elif bytes_str.isdigit():
            return int(bytes_str)
        else:
            raise ValueError(''.join([
                f'Unsupported value/suffix {bytes_str}. Supported suffix are ',
                f'{["b"] + list(units.keys())}.'
            ]))


def number_abbrev_to_int(abbrev_str: Union[int, str]) -> int:
    """Convert human readable number abbreviations to an integer.

    Args:
        abbrev_str (Union[int, str]): Value to convert.

    Raises:
        ValueError: Invalid number suffix.

    Returns:
        int: Integer value of number abbreviation.
    """
    #input is already an int
    if isinstance(abbrev_str, int) or isinstance(abbrev_str, float):
        return int(abbrev_str)

    units = {
        'k': 10**3,
        'm': 10**6,
        'b': 10**9,
        't': 10**12,
    }
    # Convert a various abbreviation types to an integer
    for suffix in units:
        abbrev_str = abbrev_str.lower().strip()
        if abbrev_str.lower().endswith(suffix):
            try:
                return int(float(abbrev_str[0:-len(suffix)]) * units[suffix])
            except ValueError:
                raise ValueError(''.join([
                    f'Unsupported value/suffix {abbrev_str}. Supported suffix are ',
                    f'{list(units.keys())}.'
                ]))
    else:
        # Convert string representation of a number to an integer
        if abbrev_str.isdigit():
            return int(abbrev_str)
        else:
            raise ValueError(''.join([
                f'Unsupported value/suffix {abbrev_str}. Supported suffix are ',
                f'{list(units.keys())}.'
            ]))


def clean_stale_shared_memory() -> None:
    """Clean up all the leaked shared memory.

    In case of a distributed run, clean up happens on local rank 0 while other local ranks wait for
    the local rank 0 to finish.
    """
    # Initialize torch.distributed ourselves, if necessary.
    destroy_dist = maybe_init_dist()

    # Perform clean up on local rank 0
    if get_local_rank() == 0:
        for prefix_int in range(1000000):
            leaked_shm = False
            for shm_name in SHM_TO_CLEAN:
                name = _get_path(prefix_int, shm_name)
                try:
                    shm = BuiltinSharedMemory(name, True, 4)
                except FileExistsError:
                    shm = BuiltinSharedMemory(name, False, 4)
                    leaked_shm = True
                finally:
                    shm.close()  # pyright: ignore
                    shm.unlink()
            # Come out of loop if no leaked shared memory
            if not leaked_shm:
                break

    # Sync all ranks
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Delete the process group if Streaming initialized it.
    if destroy_dist:
        dist.destroy_process_group()


def get_import_exception_message(package_name: str, extra_deps: str) -> str:
    """Get import exception message.

    Args:
        package_name (str): Package name.

    Returns:
        str: Exception message.
    """
    return f'Streaming was installed without {package_name} support. ' + \
            f'To use {package_name} related packages with Streaming, run ' + \
            f'`pip install \'mosaicml-streaming[{package_name}]\'`.'


@overload
def retry(
    exc_class: Union[Type[Exception], Sequence[Type[Exception]]] = ...,
    num_attempts: int = ...,
    initial_backoff: float = ...,
    max_jitter: float = ...,
) -> Callable[[TCallable], TCallable]:
    ...


@overload
def retry(exc_class: TCallable) -> TCallable:
    # Use the decorator without parenthesis
    ...


# error: Type "(TCallable@retry) -> TCallable@retry" cannot be assigned to type
# "(func: Never) -> Never"
def retry(  # type: ignore
    exc_class: Union[TCallable, Type[Exception], Sequence[Type[Exception]]] = Exception,
    num_attempts: int = 3,
    initial_backoff: float = 1.0,
    max_jitter: float = 0.5,
):
    """Decorator to retry a function with backoff and jitter.

    Attempts are spaced out with
    ``initial_backoff * 2**num_attempts + random.random() * max_jitter`` seconds.

    Example:
    .. testcode::

        from streaming.base.util import retry

        num_tries = 0

        @retry(RuntimeError, num_attempts=3, initial_backoff=0.1)
        def flaky_function():
            global num_tries
            if num_tries < 2:
                num_tries += 1
                raise RuntimeError("Called too soon!")
            return "Third time's a charm."

        print(flaky_function())

    .. testoutput::

        Third time's a charm.

    Args:
        exc_class (Type[Exception] | Sequence[Type[Exception]]], optional): The exception class or
            classes to retry. Defaults to Exception.
        num_attempts (int, optional): The total number of attempts to make. Defaults to 3.
        initial_backoff (float, optional): The initial backoff, in seconds. Defaults to 1.0.
        max_jitter (float, optional): The maximum amount of random jitter to add. Defaults to 0.5.

            Increasing the ``max_jitter`` can help prevent overloading a resource when multiple
            processes in parallel are calling the same underlying function.
    """
    if num_attempts < 1:
        raise ValueError('num_attempts must be at-least 1')

    def wrapped_func(func: TCallable) -> TCallable:

        @functools.wraps(func)
        def new_func(*args: Any, **kwargs: Any):
            i = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exc_class as e:
                    if i + 1 == num_attempts:
                        logger.debug(f'Attempt {i + 1}/{num_attempts} failed with: {e}')
                        raise e
                    else:
                        sleep(initial_backoff * 2**i + random.random() * max_jitter)
                        logger.debug(f'Attempt {i + 1}/{num_attempts} failed with: {e}')
                        i += 1

        return cast(TCallable, new_func)

    if not isinstance(exc_class, collections.abc.Sequence) and not (isinstance(
            exc_class, type) and issubclass(exc_class, Exception)):
        # Using the decorator without (), like @retry_with_backoff
        func = cast(TCallable, exc_class)
        exc_class = Exception

        return wrapped_func(func)

    return wrapped_func
