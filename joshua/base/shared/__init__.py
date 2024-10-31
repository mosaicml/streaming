# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Objects that live in shared memory.

For when using `threading` or `multiprocessing` from the python standard library won't do, because
we are coordinating separately instantiated pytorch worker processes.
"""

from joshua.base.shared.array import SharedArray as SharedArray
from joshua.base.shared.barrier import SharedBarrier as SharedBarrier
from joshua.base.shared.memory import SharedMemory as SharedMemory
from joshua.base.shared.prefix import _get_path as _get_path
from joshua.base.shared.prefix import get_shm_prefix as get_shm_prefix
from joshua.base.shared.scalar import SharedScalar as SharedScalar

__all__ = ['SharedArray', 'SharedBarrier', 'SharedMemory', 'get_shm_prefix', 'SharedScalar']
