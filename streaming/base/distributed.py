# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Helper methods to get the distributed attributes."""

import os

__all__ = ['get_rank', 'get_local_rank', 'get_local_world_size', 'get_world_size']


def get_rank() -> int:
    """Returns the rank of the current process, which is on ``[0; WORLD_SIZE - 1]``.

    Returns:
        int: The rank.
    """
    return int(os.environ.get('RANK', 0))


def get_world_size() -> int:
    """Returns the world size, which is the number of processes participating in this training run.

    Returns:
        int: The world size.
    """
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank() -> int:
    """Returns the local rank for the current process, which is on ``[0; LOCAL_WORLD_SIZE - 1]``.

    Returns:
        int: The local rank.
    """
    return int(os.environ.get('LOCAL_RANK', 0))


def get_local_world_size() -> int:
    """Returns the local world size, which is the number of processes for the current node.

    Returns:
        int: The local world size.
    """
    return int(os.environ.get('LOCAL_WORLD_SIZE', 1))
