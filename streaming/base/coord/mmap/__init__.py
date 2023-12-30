# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share data across processes with mmap()."""

from streaming.base.coord.mmap.array import MMapArray
from streaming.base.coord.mmap.barrier import MMapBarrier
from streaming.base.coord.mmap.buffer import MMapBuffer
from streaming.base.coord.mmap.number import MMapNumber

__all__ = ['MMapArray', 'MMapBarrier', 'MMapBuffer', 'MMapNumber']
