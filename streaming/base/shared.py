# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Synchronization primitives that live in shared memory.

For when using `threading` or `multiprocessing` from the python standard library won't do, because
we are coordinating separately instantiated pytorch worker processes.
"""

import atexit
import os
import shutil
from multiprocessing import resource_tracker  # pyright: ignore
from multiprocessing.shared_memory import SharedMemory
from time import sleep
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import torch
from filelock import FileLock
from torch import distributed as dist

from streaming.base.world import World

# Time to wait, in seconds.
TICK = 0.07

# Time out to wait before raising exception
TIMEOUT = 60


class SharedBarrier:
    """A barrier that works inter-process using a file lock and shared memory.

    We set the number of processes (and thereby initialize num_exit) on the first time this object
    is called. This is because the object is created in a per-rank process, and called by worker
    processes.

    Args:
        filelock_path (str): Path to lock file on local filesystem.
        shm_path (str): Shared memory object name in /dev/shm.
    """

    def __init__(self, filelock_path: str, shm_path: str) -> None:
        self.filelock_path = filelock_path
        self.created_shms = []
        self.opened_shms = []

        # Create three int32 fields in shared memory: num_enter, num_exit, flag.
        size = 3 * np.int32(0).nbytes
        shared_barrier_shm = CreateSharedMemory(name=shm_path, size=size)
        self._shm = shared_barrier_shm.shm

        # Create filelock.
        dirname = os.path.dirname(filelock_path)
        os.makedirs(dirname, exist_ok=True)
        self.lock = FileLock(filelock_path)

        self._arr = np.ndarray(3, buffer=self._shm.buf, dtype=np.int32)
        self._arr[0] = 0
        self._arr[1] = -1
        self._arr[2] = True

        def cleanup():
            """Directory clean up."""
            if os.path.islink(dirname):
                os.unlink(dirname)
            shutil.rmtree(dirname, ignore_errors=True)

        atexit.register(cleanup)

    @property
    def num_enter(self) -> int:
        """Get property num_enter.

        Returns:
            int: Number of processes that have entered the barrier.
        """
        return self._arr[0]

    @num_enter.setter
    def num_enter(self, num_enter: int) -> None:
        """Set property num_enter.

        Args:
            num_enter (int): Number of processes that have entered the barrier.
        """
        self._arr[0] = num_enter

    @property
    def num_exit(self) -> int:
        """Get property num_exit.

        Returns:
            int: Number of processes that have exited the barrier.
        """
        return self._arr[1]

    @num_exit.setter
    def num_exit(self, num_exit: int) -> None:
        """Set property num_exit.

        Args:
            num_exit (int): Number of processes that have exited the barrier.
        """
        self._arr[1] = num_exit

    @property
    def flag(self) -> bool:
        """Get property flag.

        Returns:
            bool: The flag value.
        """
        return bool(self._arr[2])

    @flag.setter
    def flag(self, flag: bool) -> None:
        """Set property flag.

        Args:
            flag (bool): The flag value.
        """
        self._arr[2] = bool(flag)

    def __call__(self, num_procs: int) -> None:
        """A set number of processes enter, wait, and exit the barrier.

        Args:
            num_procs (int): How many processes are sharing this barrier.
        """
        # Re-init the numpy array pointing to shared memory. Necessary when spawn is the
        # multiprocessing method used.
        self._arr = np.ndarray(3, buffer=self._shm.buf, dtype=np.int32)

        # Initialize num_exit to the number of processes.
        with self.lock:
            if self.num_exit == -1:
                self.num_exit = num_procs

        # If we are the first to arrive, wait for everyone to exit, then set flag to "don't go".
        self.lock.acquire()
        if not self.num_enter:
            self.lock.release()
            while self.num_exit != num_procs:
                sleep(TICK)
            self.lock.acquire()
            self.flag = False

        # Note that we entered.
        self.num_enter += 1

        # If we are the last to arrive, reset `enter` and `exit`, and set flag to "go".
        if self.num_enter == num_procs:
            self.num_enter = 0
            self.num_exit = 0
            self.flag = True
        self.lock.release()

        # Everybody waits until the flag is set to "go".
        while not self.flag:
            sleep(TICK)

        # Note that we exited.
        with self.lock:
            self.num_exit += 1
            if self.num_exit == num_procs:
                self.num_exit = -1


class CreateSharedMemory:
    """Create a new Shared Memory block or attach to an existing shared memory block.

    Args:
        name (Optional[str], optional): A unique shared memory block name. Defaults to ``None``.
        create (Optional[bool], optional): Creates a new shared memory block or attaches to an
            existing shared memory block. Defaults to ``None``.
        size (int, optional): A size of a shared memory block. Defaults to ``0``.
        auto_cleanup (bool, optional): Register atexit handler for cleanup or not.
            Defaults to ``True``.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 create: Optional[bool] = None,
                 size: int = 0,
                 auto_cleanup: bool = True):

        self.created_shms = []
        self.opened_shms = []
        shm = None
        # save the original register tracker function
        original_rtracker_reg = resource_tracker.register

        try:
            if create is True:
                # Creates a new shared memory block
                shm = SharedMemory(name, create, size)
                self.created_shms.append(shm)
            elif create is False:
                # Avoid tracking shared memory resources in a process who attaches to an existing
                # shared memory block because the process who created the shared memory is
                # responsible for destroying the shared memory block.
                resource_tracker.register = self.fix_register
                # Attaches to an existing shared memory block
                shm = SharedMemory(name, create, size)
                self.opened_shms.append(shm)
            else:
                try:
                    # Creates a new shared memory block
                    shm = SharedMemory(name, True, size)
                    self.created_shms.append(shm)
                except FileExistsError:
                    sleep(TICK)
                    resource_tracker.register = self.fix_register
                    # Attaches to an existing shared memory block
                    shm = SharedMemory(name, False, size)
                    self.opened_shms.append(shm)
            self.shm = shm
        finally:
            resource_tracker.register = original_rtracker_reg

        if auto_cleanup:
            # atexit handler doesn't get called if the program is killed by a signal not
            # handled by python or when os.exit() is called or for any python internal fatal error.
            atexit.register(self.cleanup)

    # Monkey-patched "multiprocessing.resource_tracker" to avoid unwanted resource tracker warnings.
    # PR to remove resource tracker unlinking: https://github.com/python/cpython/pull/15989
    def fix_register(self, name: str, rtype: str) -> Any:
        """Skip registering resource tracking for shared memory.

        Args:
            name (str): Name of a shared memory
            rtype (str): Name of a resource type

        Returns:
            Any: resource tracker or None
        """
        if rtype == 'shared_memory':
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)

    def fix_unregister(self, name: str, rtype: str) -> Any:
        """Skip un-registering resource tracking for shared memory.

        Args:
            name (str): Name of a shared memory
            rtype (str): Name of a resource type

        Returns:
            Any: resource tracker or None
        """
        if rtype == 'shared_memory':
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)

    def cleanup(self):
        """Clean up SharedMemory resources."""
        # save the original unregister tracker function
        original_rtracker_unreg = resource_tracker.unregister

        # Close each SharedMemory instance
        try:
            for shm in self.created_shms:
                shm.close()
                # Destroy the shared memory block
                shm.unlink()
            for shm in self.opened_shms:
                resource_tracker.unregister = self.fix_unregister
                shm.close()
        # skip the error if a child process already cleaned up the shared memory
        except FileNotFoundError:
            pass
        finally:
            resource_tracker.unregister = original_rtracker_unreg


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
