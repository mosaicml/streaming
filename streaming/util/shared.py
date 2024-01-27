# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for shared memory."""

from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory

import torch.distributed as dist

from streaming.constant import SHM_TO_CLEAN
from streaming.distributed import get_local_rank, maybe_init_dist
from streaming.shared.prefix import _get_path

__all__ = ['clean_stale_shared_memory']


def clean_stale_shared_memory() -> None:
    """Clean up all the leaked shared memory.

    In case of a distributed run, clean up happens on local rank 0 while other local ranks wait for
    the local rank 0 to finish.
    """
    # Initialize torch.distributed ourselves, if necessary.
    destroy_dist = maybe_init_dist()

    # Perform clean up on local rank 0
    if get_local_rank() == 0:
        for prefix_int in range(1000000):
            leaked_shm = False
            for shm_name in SHM_TO_CLEAN:
                name = _get_path(prefix_int, shm_name)
                try:
                    shm = BuiltinSharedMemory(name, True, 4)
                except FileExistsError:
                    shm = BuiltinSharedMemory(name, False, 4)
                    leaked_shm = True
                finally:
                    shm.close()  # pyright: ignore
                    shm.unlink()
            # Come out of loop if no leaked shared memory
            if not leaked_shm:
                break

    # Sync all ranks
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Delete the process group if Streaming initialized it.
    if destroy_dist:
        dist.destroy_process_group()
