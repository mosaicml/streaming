# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Helper methods to get the distributed attributes."""

import os
from typing import List, Optional, TypeVar, cast

import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup

TObj = TypeVar('TObj')

import torch
from torch import Tensor
from torch import distributed as dist

__all__ = [
    'all_gather', 'barrier', 'broadcast', 'get_rank', 'get_local_rank', 'get_local_world_size',
    'get_world_size'
]


def get_rank(process_group: Optional[ProcessGroup] = None) -> int:
    """Returns the rank of the current process, which is on ``[0; WORLD_SIZE - 1]``.

    Args:
        process_group (ProcessGroup, optional): the process group used to determine rank

    Returns:
        int: The rank.
    """
    rank = int(os.environ.get('RANK', 0))
    if process_group is not None:
        rank = dist.get_rank(process_group)
    return rank


def get_world_size(process_group: Optional[ProcessGroup] = None) -> int:
    """Returns the world size, which is the number of processes participating in this training run.

    Args:
        process_group (ProcessGroup, optional): the process group used to determine world size

    Returns:
        int: The world size.
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if process_group is not None:
        world_size = dist.get_world_size(process_group)
    return world_size


def get_local_rank() -> int:
    """Returns the local rank for the current process, which is on ``[0; LOCAL_WORLD_SIZE - 1]``.

    Returns:
        int: The local rank.
    """
    return int(os.environ.get('LOCAL_RANK', 0))


def get_local_world_size(process_group: Optional[ProcessGroup] = None) -> int:
    """Returns the local world size, which is the number of processes for the current node.

    Args:
        process_group (ProcessGroup, optional): the process group used to determine local world size

    Returns:
        int: The local world size.
    """
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    if process_group is not None:
        group_ranks = dist.get_process_group_ranks(process_group)
        max_local_rank = torch.cuda.device_count() - 1
        local_world_size = len([rank for rank in group_ranks if rank < max_local_rank])
    return local_world_size


def barrier(process_group: Optional[ProcessGroup] = None) -> None:
    """Synchronizes all processes.

    Args:
        process_group (ProcessGroup, optional): the process group to synchronize
    """
    if process_group:
        dist.barrier(process_group)
    elif dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast(tensor: Tensor, src: int) -> None:
    """Broadcasts the tensor to the whole group.

    Args:
        tensor (Tensor): Data to be sent if src is the rank of current process, and tensor to be
            used to save received data otherwise.
        src (int): Source rank.
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src)


def all_gather(tensor_list: List[Tensor], tensor: Tensor) -> None:
    """Gathers tensors from the whole group in a list.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain correctly-sized tensors to be
            used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
    """
    if dist.is_available() and dist.is_initialized():
        dist.all_gather(tensor_list, tensor)


def all_gather_object(obj: TObj) -> List[TObj]:
    """Collect a pickle-able object from each rank and return a list of these objects.

    .. seealso:: :func:`torch.distributed.all_gather_object`

    Args:
        obj (TObj): Object to be gathered.

    Returns:
        List[TObj]: A list of objects indexed by rank.
    """
    if dist.is_available() and dist.is_initialized():
        obj_gather_list = [0 for _ in range(get_world_size())]
        dist.all_gather_object(obj_gather_list, obj)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on
        # rank zero or will just be None on non-rank-zero.
        return cast(List[TObj], obj_gather_list)
    world_size = get_world_size()
    if world_size == 1:
        return [obj]
    raise RuntimeError(''.join([
        f'The world_size({world_size}) > 1, but the distributed package is not available ',
        'or has not been initialized. Please check you have initialized the distributed ',
        'runtime and that PyTorch has been built with distributed support.'
    ]))


def maybe_init_dist(process_group: Optional[ProcessGroup] = None) -> bool:
    """Initialize torch.distributed ourselves, if necessary.

    Args:
        process_group (ProcessGroup, optional): the process group in use

    Returns:
        bool: Whether we initialized dist ourselves.
    """
    if get_world_size() == 1 or not dist.is_available() or dist.is_initialized(
    ) or process_group is not None:
        return False
    if torch.cuda.is_available() and dist.is_nccl_available():
        backend = 'nccl'
    else:
        backend = 'gloo'
    dist.init_process_group(backend=backend, rank=get_rank(), world_size=get_world_size())
    return True
