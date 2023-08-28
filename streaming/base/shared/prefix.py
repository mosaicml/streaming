# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Register or look up the prefix to use for all shared resources.

The prefix is used by all workers using this StreamingDataset of this training job. This is used to
prevent shared resources like shared memory from colliding.
"""

from time import sleep
from typing import Iterator, List, Set, Tuple, Union

import numpy as np
from torch import distributed as dist

from streaming.base.constant import LOCALS, TICK
from streaming.base.shared import SharedMemory
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


def _get_path(prefix_int: int, name: str) -> str:
    """Get the name of the shared memory.

    Args:
        prefix (int): The prefix int.
        name (str): The name of the shared memory.

    Returns:
        str: Unique shared memory name.
    """
    return f'{prefix_int:06}_{name}'


def _pack_locals(dirnames: Set[str], prefix_int: int) -> bytes:
    """Pack local dirnames and prefix int.

    Args:
        dirnames (Set[str]): Unpacked local dirnames.
        prefix_int (int): Prefix int.

    Returns:
        bytes: Packed local dirnames and prefix int.
    """
    text = '\0'.join(sorted(dirnames)) + f'\0{prefix_int}'
    data = text.encode('utf-8')
    size = 4 + len(data)
    return b''.join([np.int32(size).tobytes(), data])


def _unpack_locals(data: bytes) -> Tuple[Set[str], int]:
    """Unpack local dirnames and prefix int.

    Args:
        data (bytes): Packed local dirnames and prefix int.

    Returns:
        Set[str]: Unpacked local dirnames and prefix int.
    """
    size = np.frombuffer(data[:4], np.int32)[0]
    text = data[4:size].decode('utf-8')
    text = text.split('\0')
    return set(text[:-1]), int(text[-1] or 0)


def _check_self(streams_local: List[str]) -> Set[str]:
    """Check our local working directories for overlap.

    Args:
        streams_local (List[str]): Local dirs.

    Returns:
        Set[str]: Local dirs as a set.
    """
    streams_local_set = set()
    for dirname in streams_local:
        if dirname in streams_local_set:
            raise ValueError(f'Reused local directory: {dirname}. Provide a different one.')
        streams_local_set.add(dirname)
    return streams_local_set


def _check_and_find(streams_local_set: Set[str], streams_remote: List[Union[str, None]]) -> int:
    """Find the next available prefix while checking existing local dirs for overlap.

    Local leader walks the existing shm prefixes starting from zero, verifying that there is no
    local working directory overlap when remote directories exist. When attaching to an existing
    shm fails, we have reached the end of the existing shms. We will register the next one.

    Args:
        streams_local_set (Set[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.

    Returns:
        int: Next available prefix int.
    """
    prefix_int = 0
    for prefix_int in _each_prefix_int():
        name = _get_path(prefix_int, LOCALS)
        try:
            shm = SharedMemory(name, False)
        except FileNotFoundError:
            break
        their_locals_set, _ = _unpack_locals(bytes(shm.buf))
        # If all the remote directories are None in a streams, get the new prefix_int and break
        # even if local directory matches between the StreamingDataset instantiation.
        if any(streams_remote):
            both = streams_local_set & their_locals_set
            if both:
                raise ValueError(
                    f'Reused local directory: {sorted(streams_local_set)} vs ' +
                    f'{sorted(their_locals_set)}. Provide a different one. If using ' +
                    f'a unique local directory, try deleting the local directory and ' +
                    f'call `streaming.base.util.clean_stale_shared_memory()` only once ' +
                    f'in your script to clean up the stale shared memory before ' +
                    f'instantiation of `StreamingDataset`.')
    return prefix_int


def _check_and_find_retrying(streams_local_set: Set[str], streams_remote: List[Union[str, None]],
                             retry: int) -> int:
    """Find the next available prefix while checking existing dirs for overlap.

    If an overlap is found, sleeps for a tick and then tries again, up to "retry" times. We allow
    this grace period because modifying python shared memory in a destructor intermediated through
    a numpy array appears to be racy.

    Args:
        streams_local_set (Set[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.
        retry (int): Number of retries upon failure before raising an exception.

    Returns:
        int: Next available prefix int.
    """
    if retry < 0:
        raise ValueError(f'Specify at least zero retries (provided {retry}).')
    errs = []
    for _ in range(1 + retry):
        try:
            return _check_and_find(streams_local_set, streams_remote)
        except ValueError as err:
            errs.append(err)
            sleep(TICK)
    raise errs[-1]


def get_shm_prefix(streams_local: List[str],
                   streams_remote: List[Union[str, None]],
                   world: World,
                   retry: int = 100) -> Tuple[int, SharedMemory]:
    """Register or lookup our shared memory prefix.

    Args:
        streams_local (List[str]): Local working dir of each stream, relative to /. We need to verify
            that there is no overlap with any other currently running StreamingDataset.
        streams_remote (List[Union[str, None]]): Remote working dir of each stream.
        world (World): Information about nodes, ranks, and workers.
        retry (int): Number of retries upon failure before raising an exception. Defaults to ``100``.

    Returns:
        Tuple[int, SharedMemory]: Shared memory integer prefix and object. The name is required to be very
            short due to limitations of Python on Mac OSX.
    """
    # Check my locals for overlap.
    streams_local_set = _check_self(streams_local)

    # First, the local leader registers the first available shm prefix, recording its locals.
    if world.is_local_leader:
        prefix_int = _check_and_find_retrying(streams_local_set, streams_remote, retry)
        name = _get_path(prefix_int, LOCALS)
        data = _pack_locals(streams_local_set, prefix_int)
        shm = SharedMemory(name, True, len(data))
        shm.buf[:len(data)] = data

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Non-local leaders go next, searching for match.
    if not world.is_local_leader:
        for prefix_int in _each_prefix_int():
            name = _get_path(prefix_int, LOCALS)
            try:
                shm = SharedMemory(name, False)
            except FileNotFoundError:
                raise RuntimeError(f'Internal error: shared memory prefix was not registered by ' +
                                   f'local leader')
            their_locals_set, their_prefix_int = _unpack_locals(bytes(shm.buf))
            if streams_local_set == their_locals_set and prefix_int == their_prefix_int:
                break

    return prefix_int, shm  # pyright: ignore
