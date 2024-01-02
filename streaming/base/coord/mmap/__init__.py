# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share data across processes with mmap()."""

from streaming.base.coord.mmap.barrier import MemMapBarrier, barrier
from streaming.base.coord.mmap.base import MemMap
from streaming.base.coord.mmap.buffer import MemMapBuffer, buffer
from streaming.base.coord.mmap.ndarray import MemMapNDArray, ndarray
from streaming.base.coord.mmap.number import (
    MemMapFloat16, MemMapFloat32, MemMapFloat64, MemMapFloating, MemMapInexact, MemMapInt8,
    MemMapInt16, MemMapInt32, MemMapInt64, MemMapInteger, MemMapNumber, MemMapSignedInteger,
    MemMapUInt8, MemMapUInt16, MemMapUInt32, MemMapUInt64, MemMapUnsignedInteger, float16, float32,
    float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64)

__all__ = [
    'MemMapBarrier', 'barrier', 'MemMap', 'MemMapBuffer', 'buffer', 'MemMapNumber',
    'MemMapInteger', 'MemMapSignedInteger', 'MemMapInt8', 'MemMapInt16', 'MemMapInt32',
    'MemMapInt64', 'MemMapUnsignedInteger', 'MemMapUInt8', 'MemMapUInt16', 'MemMapUInt32',
    'MemMapUInt64', 'MemMapInexact', 'MemMapFloating', 'MemMapFloat16', 'MemMapFloat32',
    'MemMapFloat64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
    'float16', 'float32', 'float64', 'MemMapNDArray', 'ndarray'
]
