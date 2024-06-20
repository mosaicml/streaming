# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Number DBS types."""

from typing import Tuple as TyTuple
from typing import Union as TyUnion

import numpy as np

from streaming.base.format.dbs.type.base import FixLeaf, decode_float, decode_int


class Number(FixLeaf):
    """Number DBS type abstract base class."""

    pass


class PythonNumber(Number):
    """Python number DBS type abstract base class."""

    pass


class Bool(PythonNumber):
    """Bool DBS type.

    Notes:
    - Serializes to one byte per bool.
    - If you have very many bools, consider packing them together into an array and using tye `ndarray` DBS
      type.
    """

    py_type = bool

    def encode(self, obj: TyUnion[bool, int, np.integer]) -> bytes:
        return np.uint8(obj).tobytes()

    encoded_size = np.uint8().nbytes

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[bool, int]:
        num, offset = decode_int(data, offset, np.uint8)
        return self.py_type(num), offset


class Int(PythonNumber):
    """Int DBS type.

    Notes:
    - There are infinite ints. Compromises must be made.
    - We serialize `int` as the widest commonly used signed int type (i.e., `int64`).
    - This supports the range -9223372036854775808 to 9223372036854775807, inclusive.
    - If you need a specific range (e.g., `uint64`), use a specifically sized int DBS type instead.
    - We expect that users will have not very many individual int fields per sample, so 8 bytes per
      int is not too extravagant. If you need to economize on space more than we have done here,
      use a smaller-width int type instead (e.g., `int8`, `int16`, or `int32`).
    """

    py_type = int

    def encode(self, obj: TyUnion[int, np.integer]) -> bytes:
        num = np.int64(obj)
        if obj != num:
            raise ValueError(f'Overflow during int serialization: {obj}. Use `pickle` instead, ' +
                             f'or `bytes` + your own bigint serialization.')
        return np.int64(obj).tobytes()

    encoded_size = np.int64().nbytes

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[int, int]:
        num, offset = decode_int(data, offset, np.int64)
        return self.py_type(num), offset


class Float(PythonNumber):
    """Float DBS type.

    - There are infinite reals. Compromises must be made.
    - We serialize `float` as the widest commonly used float type (i.e., `float64`).
    - If you need a specific range, use a specific DBS type instead (`float16`, `float32`, or
      `float64`).
    """

    py_type = float

    def encode(self, obj: TyUnion[int, np.integer, float, np.floating]) -> bytes:
        return np.float64(obj).tobytes()

    encoded_size = np.float64().nbytes

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[float, int]:
        num, offset = decode_float(data, offset, np.float64)
        return self.py_type(num), offset


def _get(dtype: type) -> TyTuple[type, int]:
    """Get decoded type and encoded size from numpy data type.

    Args:
        dtype (type): Numpy data type.

    Returns:
        TyTuple[type, int]: Numpy data type and its serialized size.
    """
    return dtype, dtype().nbytes


class NumpyNumber(Number):
    """Numpy number DBS type abstract base class."""

    def encode(self, obj: TyUnion[int, np.integer, float, np.floating]) -> bytes:
        raise NotImplementedError

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[np.integer, int]:
        return decode_int(data, offset, self.py_type)


class NumpyInt(NumpyNumber):
    """Numpy int DBS type abstract base class."""

    def encode(self, obj: TyUnion[int, np.integer]) -> bytes:
        num = self.py_type(obj)
        if obj != num:
            raise ValueError(f'Integer {obj} is out of range for type {self.py_type}.')
        return num.tobytes()


class UInt8(NumpyInt):
    """np.uint8 DBS type."""

    py_type, encoded_size = _get(np.uint8)


class UInt16(NumpyInt):
    """np.uint16 DBS type."""

    py_type, encoded_size = _get(np.uint16)


class UInt32(NumpyInt):
    """np.uint32 DBS type."""

    py_type, encoded_size = _get(np.uint32)


class UInt64(NumpyInt):
    """np.uint64 DBS type."""

    py_type, encoded_size = _get(np.uint64)


class Int8(NumpyInt):
    """np.int8 DBS type."""

    py_type, encoded_size = _get(np.int8)


class Int16(NumpyInt):
    """np.int16 DBS type."""

    py_type, encoded_size = _get(np.int16)


class Int32(NumpyInt):
    """np.int32 DBS type."""

    py_type, encoded_size = _get(np.int32)


class Int64(NumpyInt):
    """np.int64 DBS type."""

    py_type, encoded_size = _get(np.int64)


class NumpyFloat(NumpyNumber):
    """Numpy float DBS type abstract base class."""

    def encode(self, obj: TyUnion[int, np.integer, float, np.floating]) -> bytes:
        num = self.py_type(obj)
        return num.tobytes()


class Float16(NumpyFloat):
    """np.float16 DBS type."""

    py_type, encoded_size = _get(np.float16)


class Float32(NumpyFloat):
    """np.float32 DBS type."""

    py_type, encoded_size = _get(np.float32)


class Float64(NumpyFloat):
    """np.float64 DBS type."""

    py_type, encoded_size = _get(np.float64)
