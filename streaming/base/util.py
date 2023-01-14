# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for datasets."""

import os
import shutil
from time import sleep, time
from typing import List

from streaming.base.world import World

__all__ = ['get_list_arg']


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
            raise RuntimeError(f'{err_msg}, bailing out: ' + f'{timeout:.3f} < {dt:.3f} sec.')


def wait_for_local_leader(world: World) -> None:
    """Wait for local rank 0.

    Args:
        world (World): World state.
    """
    dir_path = os.path.join(os.path.sep, 'tmp', 'streaming', 'local_sync')
    if world.is_local_leader:
        os.makedirs(dir_path, exist_ok=True)
    else:
        wait_for_file_to_exist(dir_path,
                               poll_interval=0.07,
                               timeout=60,
                               err_msg='Waiting for local rank 0')
        if os.path.islink(dir_path):
            os.unlink(dir_path)
        shutil.rmtree(dir_path)
