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
from typing import Any, Optional

import numpy as np
from filelock import FileLock

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


class CreateSharedMemory:
    """Create a new Shared Memory block or attach to an existing shared memory block.

    Args:
        name (Optional[str], optional): A unique shared memory block name. Defaults to None.
        create (Optional[bool], optional): Creates a new shared memory block or attaches to an existing shared memory block. Defaults to None.
        size (int, optional): A size of a shared memory block. Defaults to 0.
        auto_cleanup (bool, optional): Register atexit handler for cleanup or not. Defaults to True.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 create: Optional[bool] = None,
                 size: int = 0,
                 auto_cleanup: bool = True):

        self.created_shms = []
        self.opened_shms = []
        shm = None
        original_register = resource_tracker.register
        resource_tracker.register = self.fix_register

        try:
            if create is True:
                # Creates a new shared memory block
                shm = SharedMemory(name, create, size)
                self.created_shms.append(shm)
            elif create is False:
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
                    # Attaches to an existing shared memory block
                    shm = SharedMemory(name, False, size)
                    self.opened_shms.append(shm)
            self.shm = shm
        finally:
            resource_tracker.register = original_register

        if auto_cleanup:
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
        original_unregister = resource_tracker.unregister
        resource_tracker.unregister = self.fix_unregister
        # Close each SharedMemory instance
        try:
            for shm in self.created_shms:
                shm.close()
                shm.unlink()
            for shm in self.opened_shms:
                shm.close()
        # skip the error if a child process already cleaned up the SharedMemory
        except FileNotFoundError:
            pass
        finally:
            resource_tracker.unregister = original_unregister
