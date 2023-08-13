# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Store scalars in MDS."""

from typing import Any

import numpy as np

from streaming.base.format.mds.encodings.base import Encoding

__all__ = [
    'Int', 'UInt8', 'UInt16', 'UInt32', 'UInt64', 'Int8', 'Int16', 'Int32', 'Int64', 'Float16',
    'Float32', 'Float64'
]


class Int(Encoding):
    """Store int64."""

    size = 8

    def encode(self, obj: int) -> bytes:
        self._validate(obj, int)
        return np.int64(obj).tobytes()

    def decode(self, data: bytes) -> int:
        return int(np.frombuffer(data, np.int64)[0])


class Scalar(Encoding):
    """Store scalar."""

    def __init__(self, dtype: type) -> None:
        self.dtype = dtype
        self.size = self.dtype().nbytes

    def encode(self, obj: Any) -> bytes:
        return self.dtype(obj).tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, self.dtype)[0]


class UInt8(Scalar):
    """Store uint8."""

    def __init__(self):
        super().__init__(np.uint8)


class UInt16(Scalar):
    """Store uint16."""

    def __init__(self):
        super().__init__(np.uint16)


class UInt32(Scalar):
    """Store uint32."""

    def __init__(self):
        super().__init__(np.uint32)


class UInt64(Scalar):
    """Store uint64."""

    def __init__(self):
        super().__init__(np.uint64)


class Int8(Scalar):
    """Store int8."""

    def __init__(self):
        super().__init__(np.int8)


class Int16(Scalar):
    """Store int16."""

    def __init__(self):
        super().__init__(np.int16)


class Int32(Scalar):
    """Store int32."""

    def __init__(self):
        super().__init__(np.int32)


class Int64(Scalar):
    """Store int64."""

    def __init__(self):
        super().__init__(np.int64)


class Float16(Scalar):
    """Store float16."""

    def __init__(self):
        super().__init__(np.float16)


class Float32(Scalar):
    """Store float32."""

    def __init__(self):
        super().__init__(np.float32)


class Float64(Scalar):
    """Store float64."""

    def __init__(self):
        super().__init__(np.float64)
