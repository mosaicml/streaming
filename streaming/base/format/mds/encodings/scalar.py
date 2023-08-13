# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Store scalars in MDS.

Class hierarachy (* denotes abstract):

    Scalar*
    ├── IntScalar*
    |   └── UInt8
    |   ├── UInt16
    |   ├── UInt32
    |   ├── UInt64
    |   ├── Int8
    |   ├── Int16
    |   ├── Int32
    |   └── Int64
    |       └── Int
    └── FloatScalar*
        ├── Float16
        ├── Float32
        └── Float64
            └── Float
"""

from typing import Any

import numpy as np

from streaming.base.format.mds.encodings.base import Encoding

__all__ = [
    'Int', 'Float', 'UInt8', 'UInt16', 'UInt32', 'UInt64', 'Int8', 'Int16', 'Int32', 'Int64',
    'Float16', 'Float32', 'Float64'
]


class Scalar(Encoding):
    """Scalar base class."""

    def __init__(self, dtype: type) -> None:
        self.dtype = dtype
        self.size = self.dtype().nbytes

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, self.dtype)[0]


class IntScalar(Scalar):
    """Int scalar base class."""

    def encode(self, obj: Any) -> bytes:
        num = self.dtype(obj)
        if obj != num:
            raise ValueError(f'Integer to encode is out of range for {self.dtype}: {obj}.')
        return num.tobytes()


class FloatScalar(Scalar):
    """Float scalar base class."""

    def encode(self, obj: Any) -> bytes:
        num = self.dtype(obj)
        return num.tobytes()


class UInt8(IntScalar):
    """Store uint8."""

    def __init__(self):
        super().__init__(np.uint8)


class UInt16(IntScalar):
    """Store uint16."""

    def __init__(self):
        super().__init__(np.uint16)


class UInt32(IntScalar):
    """Store uint32."""

    def __init__(self):
        super().__init__(np.uint32)


class UInt64(IntScalar):
    """Store uint64."""

    def __init__(self):
        super().__init__(np.uint64)


class Int8(IntScalar):
    """Store int8."""

    def __init__(self):
        super().__init__(np.int8)


class Int16(IntScalar):
    """Store int16."""

    def __init__(self):
        super().__init__(np.int16)


class Int32(IntScalar):
    """Store int32."""

    def __init__(self):
        super().__init__(np.int32)


class Int64(IntScalar):
    """Store int64."""

    def __init__(self):
        super().__init__(np.int64)


class Float16(FloatScalar):
    """Store float16."""

    def __init__(self):
        super().__init__(np.float16)


class Float32(FloatScalar):
    """Store float32."""

    def __init__(self):
        super().__init__(np.float32)


class Float64(FloatScalar):
    """Store float64."""

    def __init__(self):
        super().__init__(np.float64)


class Int(Int64):
    """Store int."""

    def decode(self, data: bytes) -> Any:
        ret = super().decode(data)
        return int(ret)


class Float(Float64):
    """Store float."""

    def decode(self, data: bytes) -> Any:
        ret = super().decode(data)
        return float(ret)
