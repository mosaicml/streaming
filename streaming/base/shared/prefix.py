# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Synchronization primitives that live in shared memory.

For when using `threading` or `multiprocessing` from the python standard library won't do, because
we are coordinating separately instantiated pytorch worker processes.
"""

from multiprocessing import resource_tracker  # pyright: ignore
from multiprocessing.shared_memory import SharedMemory
from typing import List, Set, Tuple

import numpy as np
import torch
from torch import distributed as dist

from streaming.base.shared.memory import CreateSharedMemory
from streaming.base.world import World

# Time to wait, in seconds.
TICK = 0.07

# Time out to wait before raising exception
TIMEOUT = 60


def _pack_locals(dirnames: Set[str]) -> bytes:
    """Pack local dirnames.

    Args:
        dirnames (Set[str]): Unpacked local dirnames.

    Returns:
        bytes: Packed local dirnames.
    """
    text = '\0'.join(sorted(dirnames))
    data = text.encode('utf-8')
    size = 4 + len(data)
    return b''.join([np.int32(size).tobytes(), data])


def _unpack_locals(data: bytes) -> Set[str]:
    """Unpack local dirnames.

    Args:
        data (bytes): Packed local dirnames.

    Returns:
        Set[str]: Unpacked local dirnames.
    """
    size = np.frombuffer(data[:4], np.int32)[0]
    text = data[4:size].decode('utf-8')
    return set(text.split('\0'))


def get_shm_prefix(my_locals: List[str], world: World) -> Tuple[str, SharedMemory]:
    """Register or lookup our shared memory prefix.

    Args:
        my_locals (List[str]): Local working dir of each stream, relative to /. We need to verify
            that there is no overlap with any other currently running StreamingDataset.
        world (World): Information about nodes, ranks, and workers.

    Returns:
        Tuple[str, CreateSharedMemory]: Shared memory prefix and object. The name is required to be
            very short for Mac OSX.
    """
    # Check my locals for overlap.
    my_locals_set = set()
    for dirname in my_locals:
        if dirname in my_locals_set:
            raise ValueError(f'Reused local directory: {dirname}. Provide a different one.')
        my_locals_set.add(dirname)

    # Local leader goes first, checking and registering.
    if world.is_local_leader:
        # Local leader walks the existing shm prefixes starting from zero, verifying that there is
        # no local working directory overlap.  When attaching to an existing shm fails, we have
        # reached the end of the existing shms.
        for prefix_int in range(10**6):
            prefix = f'{prefix_int:06}'
            name = f'{prefix}_locals'
            try:
                shm = CreateSharedMemory(name, False).shm
            except:
                break
            their_locals_set = _unpack_locals(bytes(shm.buf))
            both = my_locals_set & their_locals_set
            if both:
                raise ValueError(f'Reused local directory: {sorted(my_locals_set)} vs ' +
                                 f'{sorted(their_locals_set)}. Provide a different one.')

        # Local leader registers the first available shm prefix, recording its locals.
        prefix = f'{prefix_int:06}'  # pyright: ignore
        name = f'{prefix}_locals'
        data = _pack_locals(my_locals_set)
        shm = CreateSharedMemory(name, True, len(data)).shm
        shm.buf[:len(data)] = data

    # Distributed barrier over all ranks, possibly setting up dist to do so.
    destroy_dist = False
    if 1 < world.num_ranks:
        if dist.is_available() and not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() and dist.is_nccl_available() else 'gloo'
            dist.init_process_group(backend=backend, rank=world.rank, world_size=world.num_ranks)
            destroy_dist = True
        dist.barrier()

    # Non-local leaders go next, searching for match.
    if not world.is_local_leader:
        for prefix_int in range(10**6):
            prefix = f'{prefix_int:06}'
            name = f'{prefix}_locals'
            try:
                shm = CreateSharedMemory(name, False).shm
            except:
                raise RuntimeError('Internal error: shm prefix was not registered by local leader')
            their_locals_set = _unpack_locals(bytes(shm.buf))
            if my_locals_set == their_locals_set:
                break

    # Distributed barrier, then tear down dist if we set it up.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if destroy_dist:
        dist.destroy_process_group()

    return prefix, shm  # pyright: ignore
