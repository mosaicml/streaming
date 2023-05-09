# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Register or look up the prefix to use for all shared resources.

The prefix is used by all workers using this StreamingDataset of this training job. This is used to
prevent shared shared resources like shmem from colliding.
"""

from time import sleep
from typing import Iterator, List, Set, Tuple

import numpy as np
import torch
from torch import distributed as dist

from streaming.base.shared import SharedMemory
from streaming.base.util import TICK
from streaming.base.world import World


def _each_prefix_int() -> Iterator[int]:
    """Get each possible prefix int to check in order.

    Returns:
        Iterator[int]: Each prefix int.
    """
    prefix_int = 0
    while True:
        yield prefix_int
        prefix_int += 1


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


def _check_self(my_locals: List[str]) -> Set[str]:
    """Check our local working directories for overlap.

    Args:
        my_locals (List[str]): Local dirs.

    Returns:
        Set[str]: Local dirs as a set.
    """
    my_locals_set = set()
    for dirname in my_locals:
        if dirname in my_locals_set:
            raise ValueError(f'Reused local directory: {dirname}. Provide a different one.')
        my_locals_set.add(dirname)
    return my_locals_set


def _check_and_find(my_locals_set: Set[str]) -> int:
    """Find the next available prefix while checking existing local dirs for overlap.

    Local leader walks the existing shm prefixes starting from zero, verifying that there is no
    local working directory overlap. When attaching to an existing shm fails, we have reached the
    end of the existing shms. We will register the next one.

    Args:
        my_locals_set (Set[str]): Our local working directories.

    Returns:
        int: Next available prefix int.
    """
    prefix_int = 0
    for prefix_int in _each_prefix_int():
        prefix = f'{prefix_int:06}'
        name = f'{prefix}_locals'
        try:
            shm = SharedMemory(name, False)
        except FileNotFoundError:
            break
        their_locals_set = _unpack_locals(bytes(shm.buf))
        both = my_locals_set & their_locals_set
        if both:
            raise ValueError(f'Reused local directory: {sorted(my_locals_set)} vs ' +
                             f'{sorted(their_locals_set)}. Provide a different one.')
    return prefix_int


def _check_and_find_retrying(my_locals_set: Set[str], retry: int) -> int:
    """Find the next available prefix while checking existing dirs for overlap.

    If an overlap is found, sleeps for a tick and then tries again, up to "retry" times. We allow
    this grace period because modifying python shared memory in a destructor intermediated through
    a numpy array appears to be racy.

    Args:
        my_locals_set (Set[str]): Our local working directories.
        retry (int): Number of retries upon failure before raising an exception.

    Returns:
        int: Next available prefix int.
    """
    if retry < 0:
        raise ValueError(f'Specify at least zero retries (provided {retry}).')
    errs = []
    for _ in range(1 + retry):
        try:
            return _check_and_find(my_locals_set)
        except ValueError as err:
            errs.append(err)
            sleep(TICK)
    raise errs[-1]


def get_shm_prefix(my_locals: List[str],
                   world: World,
                   retry: int = 100) -> Tuple[str, SharedMemory]:
    """Register or lookup our shared memory prefix.

    Args:
        my_locals (List[str]): Local working dir of each stream, relative to /. We need to verify
            that there is no overlap with any other currently running StreamingDataset.
        world (World): Information about nodes, ranks, and workers.
        retry (int): Number of retries upon failure before raising an excpetion. Defaults to ``7``.

    Returns:
        Tuple[str, SharedMemory]: Shared memory prefix and object. The name is required to be very
            short due to limitations of Python on Mac OSX.
    """
    # Check my locals for overlap.
    my_locals_set = _check_self(my_locals)

    # First, the local leader registers the first available shm prefix, recording its locals.
    if world.is_local_leader:
        prefix_int = _check_and_find_retrying(my_locals_set, retry)
        prefix = f'{prefix_int:06}'  # pyright: ignore
        name = f'{prefix}_locals'
        data = _pack_locals(my_locals_set)
        shm = SharedMemory(name, True, len(data))
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
        for prefix_int in _each_prefix_int():
            prefix = f'{prefix_int:06}'
            name = f'{prefix}_locals'
            try:
                shm = SharedMemory(name, False)
            except FileNotFoundError:
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
