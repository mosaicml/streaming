# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Objects that live in shared memory.

For when using `threading` or `multiprocessing` from the python standard library won't do, because
we are coordinating separately instantiated pytorch worker processes.
"""

from streaming.base.shared.barrier import SharedBarrier as SharedBarrier
from streaming.base.shared.memory import CreateSharedMemory as CreateSharedMemory
from streaming.base.shared.prefix import get_shm_prefix as get_shm_prefix

__all__ = ['SharedBarrier', 'CreateSharedMemory', 'get_shm_prefix']
