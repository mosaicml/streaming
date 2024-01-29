# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Objects that live in shared memory.

For when using `threading` or `multiprocessing` from the python standard library won't do, because
we are coordinating separately instantiated pytorch worker processes.
"""

from streaming.shared.array import SharedArray as SharedArray
from streaming.shared.barrier import SharedBarrier as SharedBarrier
from streaming.shared.memory import SharedMemory as SharedMemory
from streaming.shared.prefix import _get_path as _get_path
from streaming.shared.prefix import get_shm_prefix as get_shm_prefix
from streaming.shared.scalar import SharedScalar as SharedScalar

__all__ = ['SharedArray', 'SharedBarrier', 'SharedMemory', 'get_shm_prefix', 'SharedScalar']
