# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for datasets."""

import os
from time import sleep, time
from typing import List

__all__ = ['get_list_arg']

# Time to wait, in seconds.
TICK = 0.007


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


def bytes_to_int(bytes_str: str) -> int:
    """Convert human readable byte format to an integer.

    Args:
        bytes_str (str): Value to convert.

    Raises:
        ValueError: Invalid byte suffix.

    Returns:
        int: Integer value of bytes.
    """
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
            return int(float(bytes_str[0:-len(suffix)]) * units[suffix])
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
