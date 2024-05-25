# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Improved quiet implementation of shared memory in pure python."""

import atexit
from multiprocessing import resource_tracker  # pyright: ignore
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
import signal, sys
from time import sleep
from typing import Any, Optional
import torch
import logging

from streaming.base.constant import TICK
logger = logging.getLogger(__name__)


class SharedMemory:
    """Improved quiet implementation of shared memory.

    Args:
        name (str, optional): A unique shared memory block name. Defaults to ``None``.
        create (bool, optional): Creates a new shared memory block or attaches to an existing
            shared memory block. Defaults to ``None``.
        size (int, optional): A size of a shared memory block. Defaults to ``0``.
        auto_cleanup (bool, optional): Register atexit handler for cleanup or not. Defaults to
            ``True``.
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
        logger.info(f"bigning debug shared memory init")
        print(f"bigning debug shared memory init")

        try:
            if create is True:
                # Creates a new shared memory block
                shm = BuiltinSharedMemory(name, create, size)
                self.created_shms.append(shm)
            elif create is False:
                # Avoid tracking shared memory resources in a process who attaches to an existing
                # shared memory block because the process who created the shared memory is
                # responsible for destroying the shared memory block.
                resource_tracker.register = self.fix_register
                # Attaches to an existing shared memory block
                shm = BuiltinSharedMemory(name, create, size)
                self.opened_shms.append(shm)
            else:
                try:
                    # Creates a new shared memory block
                    shm = BuiltinSharedMemory(name, True, size)
                    self.created_shms.append(shm)
                except FileExistsError:
                    sleep(TICK)
                    resource_tracker.register = self.fix_register
                    # Attaches to an existing shared memory block
                    shm = BuiltinSharedMemory(name, False, size)
                    self.opened_shms.append(shm)
            self.shm = shm
        finally:
            resource_tracker.register = original_rtracker_reg

        self.cleaned_up = False

        if auto_cleanup:
            # atexit handler doesn't get called if the program is killed by a signal not
            # handled by python or when os.exit() is called or for any python internal fatal error.
            atexit.register(self.cleanup)

            def signal_handler(sig, frame):
                signame = signal.Signals(sig).name
                log.warning(f"signal handler {sig=}, {signame=}")
                self.cleanup()
                sys.exit(0)
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGKILL, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)


    @property
    def buf(self) -> memoryview:
        """Internal buffer accessor.

        Returns:
            memoryview: Internal buffer.
        """
        return self.shm.buf

    # Monkey-patched "multiprocessing.resource_tracker" to skip unwanted resource tracker warnings.
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
        logger.info(f"bigning debug rank {torch.distributed.get_rank()} shared memory cleanup")
        logger.warning(f"bigning debug rank {torch.distributed.get_rank()} shared memory cleanup")
        print(f"bigning debug rank {torch.distributed.get_rank()} shared memory cleanup")
        if self.cleaned_up:
            return


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
            self.cleaned_up = True

