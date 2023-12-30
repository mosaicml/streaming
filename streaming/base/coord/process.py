# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility methods for coordination related to processes."""

from typing import Dict

from psutil import process_iter


def get_live_processes() -> Dict[int, int]:
    """Get the live processes by pid and creation time.

    Returns:
        Dict[int, int]: Mapping of pid to creation time in integer nanoseconds.
    """
    ret = {}
    for obj in process_iter(['pid', 'create_time']):
        ret[obj.pid] = int(obj.create_time() * 1e9)
    return ret
