# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""The Streaming logical type hierarchy.

This is a common language of types which the type systems of all Streaming shard formats are mapped
to. A field is stored as its shard format-specific physical type, and loaded and returned as its
logical type.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import DTypeLike

__all__ = [
    'Type', 'Bytes', 'Str', 'Number', 'Decimal', 'Float', 'Float64', 'Float32', 'Float16', 'Int',
    'Int64', 'Int32', 'Int16', 'Int8', 'UInt64', 'UInt32', 'UInt16', 'UInt8', 'Bool', 'NDArray',
    'Image', 'JSON', 'Pickle'
]


class Type:
    """Logical type."""

    def get_signature(self) -> str:
        """Get a string representation of this logical type.

        Returns:
            str: String representation.
        """
        return self.__class__.__name__


class Bytes(Type):
    """Bytes logical type."""
    pass


class Str(Type):
    """UTF-8 string logical type."""
    pass


class Number(Type):
    """Number logical type."""
    pass


class Decimal(Number):
    """Decimal logical type."""
    pass


class Float(Number):
    """Native floating point logical type.

    This logical type refers to your programming language's default floating point type. Presumably
    the value will have been serialized at that precision or higher.

    For example, in Python/CPython, the language has its own ``float`` type, which is internally
    backed by a ``double`` in the implementation.
    """
    pass


class Float64(Float):
    """Float64 logical type."""
    pass


class Float32(Float64):
    """Float32 logical type."""
    pass


class Float16(Float32):
    """Float16 logical type."""
    pass


class Int(Number):
    """Arbitrary-precision integer logical type."""
    pass


class Int64(Int):
    """``int64`` logical type."""
    pass


class Int32(Int64):
    """``int32`` logical type."""
    pass


class Int16(Int32):
    """``int16`` logical type."""
    pass


class Int8(Int16):
    """``int8`` logical type."""
    pass


class UInt64(Int):
    """``uint64`` logical type."""
    pass


class UInt32(UInt64):
    """``uint32`` logical type."""
    pass


class UInt16(UInt32):
    """``uint16`` logical type."""
    pass


class UInt8(UInt16):
    """``uint8`` logical type."""
    pass


class Bool(UInt8):
    """``bool`` logical type."""
    pass


class NDArray(Type):
    """Numpy ndarray logical type.

    Args:
        shape (Tuple[int], optional): Optional shape requirement.
        dtype (DTypeLike, optional): Optional dtype requirement.
    """

    def __init__(
        self,
        shape: Optional[Tuple[int]] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype) if dtype else None

    def get_signature(self) -> str:
        logical_type = self.__class__.__name__
        shape = ','.join(map(str, self.shape)) if self.shape else ''
        dtype = self.dtype.name if self.dtype else ''
        return ':'.join([logical_type, shape, dtype])


class Image(Type):
    """PIL Image logical type."""
    pass


class JSON(Type):
    """JSON logical type."""
    pass


class Pickle(Type):
    """Pickle logical type."""
    pass
