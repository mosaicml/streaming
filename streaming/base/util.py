# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for datasets."""

import multiprocessing as mp
from typing import List

__all__ = ['get_list_arg']


def get_list_arg(text: str) -> List[str]:
    """Pass a list as a command-line flag.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: Splits, if any.
    """
    return text.split(',') if text else []


def set_mp_start_method(platform: str) -> None:
    """Set the multiprocessing start method.

    Args:
        platform (str): Machine platform name
    """
    IS_MACOS = platform == 'darwin'

    # Set the multiprocessing start method to `fork` for MAC OS since
    # streaming uses a FileLock for sharing the resources between ranks
    # and workers and this works because fork doesn't pickle.
    if IS_MACOS:
        mp.set_start_method('fork', force=True)
