# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share a single number across processes using mmap().

Class hierarchy:

    MemMap
    ├── ...
    ├── MemMapNumber
    │   ├── MemMapInexact
    │   │   └── MemMapFloating
    │   │       ├── MemMapFloat16
    │   │       ├── MemMapFloat32
    │   │       └── MemMapFloat64
    │   └── MemMapInteger
    │       ├── MemMapSignedInteger
    │       │   ├── MemMapInt16
    │       │   ├── MemMapInt32
    │       │   ├── MemMapInt64
    │       │   └── MemMapInt8
    │       └── MemMapUnsignedInteger
    │           ├── MemMapUInt16
    │           ├── MemMapUInt32
    │           ├── MemMapUInt64
    │           └── MemMapUInt8
    └── ...
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from streaming.base.coord.mmap.base import MemMap, Number, T

__all__ = [
    'MemMapNumber', 'MemMapInteger', 'MemMapSignedInteger', 'MemMapInt8', 'MemMapInt16',
    'MemMapInt32', 'MemMapInt64', 'MemMapUnsignedInteger', 'MemMapUInt8', 'MemMapUInt16',
    'MemMapUInt32', 'MemMapUInt64', 'MemMapInexact', 'MemMapFloating', 'MemMapFloat16',
    'MemMapFloat32', 'MemMapFloat64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
    'uint32', 'uint64', 'float16', 'float32', 'float64'
]


class MemMapNumber(MemMap[T]):
    """A number backed by a memory-mapped file.

    Args:
        filename (str): Path to file to memory map.
        dtype (DTypeLike, optional): The value's data type. Defaults to ``None``.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(
        self,
        filename: str,
        dtype: Optional[DTypeLike] = None,
        value: Optional[Union[Number, NDArray[np.number]]] = None,
    ) -> None:
        super().__init__(filename, 1, dtype, value)

    def get(self) -> T:
        """Get value.

        Returns:
            np.number: The value.
        """
        return np.frombuffer(self.mmap, self.dtype)[0]

    def set(self, value: Number) -> None:
        """Set value.

        Args:
            value (Number): The value.
        """
        dtype_class = getattr(np, self.dtype.name)
        self.mmap[:] = dtype_class.tobytes()


class MemMapInteger(MemMapNumber):
    """An integer backed by a mempry-mapped file."""

    pass


class MemMapSignedInteger(MemMapInteger):
    """A signed integer backecd by a memory-mapped file."""

    pass


class MemMapInt8(MemMapSignedInteger):
    """An int8 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.int8, value)


class MemMapInt16(MemMapSignedInteger):
    """An int16 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.int16, value)


class MemMapInt32(MemMapSignedInteger):
    """An int32 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.int32, value)


class MemMapInt64(MemMapSignedInteger):
    """An int64 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.int64, value)


class MemMapUnsignedInteger(MemMapInteger):
    """An unsigned integer backecd by a memory-mapped file."""

    pass


class MemMapUInt8(MemMapUnsignedInteger):
    """An int8 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.uint8, value)


class MemMapUInt16(MemMapUnsignedInteger):
    """An int16 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.uint16, value)


class MemMapUInt32(MemMapUnsignedInteger):
    """An int32 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.int32, value)


class MemMapUInt64(MemMapUnsignedInteger):
    """An int64 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.uint64, value)


class MemMapInexact(MemMapNumber):
    """An inexact number backed by a mempry-mapped file."""

    pass


class MemMapFloating(MemMapInexact):
    """A floating point number backecd by a memory-mapped file."""

    pass


class MemMapFloat16(MemMapFloating):
    """A float16 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.float16, value)


class MemMapFloat32(MemMapFloating):
    """A float32 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.float32, value)


class MemMapFloat64(MemMapFloating):
    """A float64 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(self, filename: str, value: Optional[Number] = None) -> None:
        super().__init__(filename, np.float64, value)


def int8(filename: str, value: Optional[Number] = None) -> MemMapInt8:
    """Get a int8 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapInt8: A int8 backed by a memory-mapped file.
    """
    return MemMapInt8(filename, value)


def int16(filename: str, value: Optional[Number] = None) -> MemMapInt16:
    """Get a int16 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapInt16: A int16 backed by a memory-mapped file.
    """
    return MemMapInt16(filename, value)


def int32(filename: str, value: Optional[Number] = None) -> MemMapInt32:
    """Get a int32 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapInt32: A int32 backed by a memory-mapped file.
    """
    return MemMapInt32(filename, value)


def int64(filename: str, value: Optional[Number] = None) -> MemMapInt64:
    """Get a int64 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapInt64: A int64 backed by a memory-mapped file.
    """
    return MemMapInt64(filename, value)


def uint8(filename: str, value: Optional[Number] = None) -> MemMapUInt8:
    """Get a uint8 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapUInt8: A uint8 backed by a memory-mapped file.
    """
    return MemMapUInt8(filename, value)


def uint16(filename: str, value: Optional[Number] = None) -> MemMapUInt16:
    """Get a uint16 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapUInt16: A uint16 backed by a memory-mapped file.
    """
    return MemMapUInt16(filename, value)


def uint32(filename: str, value: Optional[Number] = None) -> MemMapUInt32:
    """Get a uint32 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapUInt32: A uint32 backed by a memory-mapped file.
    """
    return MemMapUInt32(filename, value)


def uint64(filename: str, value: Optional[Number] = None) -> MemMapUInt64:
    """Get a uint64 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapUInt64: A uint64 backed by a memory-mapped file.
    """
    return MemMapUInt64(filename, value)


def float16(filename: str, value: Optional[Number] = None) -> MemMapFloat16:
    """Get a float16 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapFloat16: A float16 backed by a memory-mapped file.
    """
    return MemMapFloat16(filename, value)


def float32(filename: str, value: Optional[Number] = None) -> MemMapFloat32:
    """Get a float32 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapFloat32: A float32 backed by a memory-mapped file.
    """
    return MemMapFloat32(filename, value)


def float64(filename: str, value: Optional[Number] = None) -> MemMapFloat64:
    """Get a float64 backed by a memory-mapped file.

    Args:
        filename (str): File to memory map.
        value (Number, optional): If a number, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapFloat64: A float64 backed by a memory-mapped file.
    """
    return MemMapFloat64(filename, value)
