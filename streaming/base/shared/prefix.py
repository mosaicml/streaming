# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Register or look up the prefix to use for all shared resources.

The prefix is used by all workers using this StreamingDataset of this training job. This is used to
prevent shared resources like shared memory from colliding.
"""

import os
from collections import Counter
from tempfile import gettempdir
from time import sleep
from typing import Iterator, Union

import numpy as np
from torch import distributed as dist

from streaming.base.constant import BARRIER_FILELOCK, CACHE_FILELOCK, LOCALS, SHM_TO_CLEAN, TICK
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


def _pack_locals(dirnames: list[str], prefix_int: int) -> bytes:
    """Pack local dirnames and prefix int.

    Args:
        dirnames (List[str]): Unpacked local dirnames.
        prefix_int (int): Prefix int.

    Returns:
        bytes: Packed local dirnames and prefix int.
    """
    text = '\0'.join(dirnames) + f'\0{prefix_int}'
    data = text.encode('utf-8')
    size = 4 + len(data)
    return b''.join([np.int32(size).tobytes(), data])


def _unpack_locals(data: bytes) -> tuple[list[str], int]:
    """Unpack local dirnames and prefix int.

    Args:
        data (bytes): Packed local dirnames and prefix int.

    Returns:
        List[str]: Unpacked local dirnames and prefix int.
    """
    size = np.frombuffer(data[:4], np.int32)[0]
    text = data[4:size].decode('utf-8')
    text = text.split('\0')
    return text[:-1], int(text[-1] or 0)


def _check_self(streams_local: list[str]) -> None:
    """Check our local working directories for overlap.

    Args:
        streams_local (List[str]): Local dirs.

    Raises:
        ValueError: If there is overlap.
    """
    occurrences = Counter(streams_local)
    duplicate_local_dirs = [dirname for dirname, count in occurrences.items() if count > 1]
    if duplicate_local_dirs:
        raise ValueError(
            f'Reused local directory: {duplicate_local_dirs}. Provide a different one.')


def _check_and_find(streams_local: list[str], streams_remote: list[Union[str, None]],
                    shm_name: str) -> int:
    """Find the next available prefix while checking existing local dirs for overlap.

    Local leader walks the existing shm prefixes starting from zero, verifying that there is no
    local working directory overlap when remote directories exist. When attaching to an existing
    shm fails, we have reached the end of the existing shms. We will register the next one.

    Args:
        streams_local (List[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.
        shm_name (str): The shared memory file name, e.g., LOCALS, BARRIER etc.

    Returns:
        int: Next available prefix int.
    """
    prefix_int = 0

    for prefix_int in _each_prefix_int():

        name = _get_path(prefix_int, shm_name)

        # Check if any shared memory filelocks exist for the current prefix
        try:
            filelock_exists = any(
                os.path.exists(os.path.join(gettempdir(), _get_path(prefix_int, filelock_name)))
                for filelock_name in [BARRIER_FILELOCK, CACHE_FILELOCK])
            if filelock_exists:
                continue
        except PermissionError:
            continue

        # Attempt to access shared memory by name. Use prefix_int if files do not exist
        try:
            shm = SharedMemory(name, False)
        except PermissionError:
            continue
        except FileNotFoundError:
            break

        if shm_name != LOCALS:
            continue

        their_locals, _ = _unpack_locals(bytes(shm.buf))

        # Do not check for a conflicting local directories across existing shared memory if
        # remote directories are None. Get the next prefix.
        if any(streams_remote):
            # Get the indices of the local directories which matches with the current
            # shared memory.
            matching_index = np.where(np.isin(streams_local, their_locals))[0]
            if matching_index.size > 0:
                for idx in matching_index:
                    # If there is a conflicting local directory for a non-None remote directory,
                    # raise an exception.
                    if streams_remote[idx] is not None:
                        raise ValueError(
                            f'Reused local directory: {streams_local} vs ' +
                            f'{their_locals}. Provide a different one. If using ' +
                            f'a unique local directory, try deleting the local directory and ' +
                            f'call `streaming.base.util.clean_stale_shared_memory()` only once ' +
                            f'in your script to clean up the stale shared memory before ' +
                            f'instantiation of `StreamingDataset`.')
    return prefix_int


def _check_and_find_retrying(streams_local: list[str], streams_remote: list[Union[str, None]],
                             shm_name: str, retry: int) -> int:
    """Find the next available prefix while checking existing dirs for overlap.

    If an overlap is found, sleeps for a tick and then tries again, up to "retry" times. We allow
    this grace period because modifying python shared memory in a destructor intermediated through
    a numpy array appears to be racy.

    Args:
        streams_local (List[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.
        shm_name (str): The shared memory file name, e.g., LOCALS, BARRIER etc.
        retry (int): Number of retries upon failure before raising an exception.

    Returns:
        int: Next available prefix int.
    """
    if retry < 0:
        raise ValueError(f'Specify at least zero retries (provided {retry}).')
    errs = []
    for _ in range(1 + retry):
        try:
            return _check_and_find(streams_local, streams_remote, shm_name)
        except ValueError as err:
            errs.append(err)
            sleep(TICK)
    raise errs[-1]


def get_shm_prefix(streams_local: list[str],
                   streams_remote: list[Union[str, None]],
                   world: World,
                   retry: int = 100) -> tuple[int, SharedMemory]:
    """Register or lookup our shared memory prefix.

    Args:
        streams_local (List[str]): Local working dir of each stream, relative to /.
            We need to verify that there is no overlap with any other currently
            running StreamingDataset.
        streams_remote (List[Union[str, None]]): Remote working dir of each stream.
        world (World): Information about nodes, ranks, and workers.
        retry (int): Number of retries upon failure before raising an exception.
            Defaults to ``100``.

    Returns:
        Tuple[int, SharedMemory]: Shared memory integer prefix and object. The name
            is required to be very short due to limitations of Python on Mac OSX.
    """
    # Check my locals for overlap.
    _check_self(streams_local)

    prefix_int = max([
        _check_and_find_retrying(streams_local, streams_remote, shm_name=shm_name, retry=retry)
        for shm_name in SHM_TO_CLEAN
    ])

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # First, the local leader registers the first available shm prefix, recording its locals.
    if world.is_local_leader:
        name = _get_path(prefix_int, LOCALS)
        data = _pack_locals(streams_local, prefix_int)
        shm = SharedMemory(name, True, len(data))
        shm.buf[:len(data)] = data

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Non-local leaders go next, searching for match.
    if not world.is_local_leader:
        name = _get_path(prefix_int, LOCALS)
        try:
            shm = SharedMemory(name, False)
        except FileNotFoundError:
            raise RuntimeError(f'Internal error: shared memory prefix={prefix_int} was not ' +
                               f'registered by local leader. This may be because you specified ' +
                               f'different ``local`` parameters from different ranks.')

        their_locals, their_prefix_int = _unpack_locals(bytes(shm.buf))
        if streams_local != their_locals or prefix_int != their_prefix_int:
            raise RuntimeError(f'Internal error: shared memory registered does not match ' +
                               f'local leader as streams_local or prefix_int not match. ' +
                               f'local leader: {their_locals} and {their_prefix_int}. ' +
                               f'expected: {streams_local} and {prefix_int}.')
    return prefix_int, shm  # pyright: ignore
