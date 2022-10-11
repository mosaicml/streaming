# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Helper methods to get the distributed attributes."""

import os

from torch.utils.data import get_worker_info

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


def is_local_leader() -> bool:
    """Get whether we are the local leader.

    This is useful in situations where you need exactly one process per node to do something. May
    be a worker, or not.

    Returns:
        bool: Whether we are the local leader.
    """
    rank_of_node = get_local_rank()
    if rank_of_node:
        return False

    info = get_worker_info()
    worker_of_rank = info.id if info else 0
    return not worker_of_rank


def get_worker() -> int:
    """Get what worker we are out of all of them.

    This is useful for partitioning work across all participants.

    Returns:
        int: Global worker ID.
    """
    rank = get_rank()
    info = get_worker_info()
    if info:
        worker_of_rank = info.id
        workers_per_rank = info.num_workers
    else:
        worker_of_rank = 0
        workers_per_rank = 1
    return rank * workers_per_rank + worker_of_rank
