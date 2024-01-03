# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share an ndarray across processes using mmap()."""

from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from streaming.base.coord.mmap.base import MemMap, Number, T

__all__ = ['MemMapNDArray', 'ndarray']

IndexType = Union[int, slice, np.integer, NDArray[np.integer]]


class MemMapNDArray(MemMap[T]):
    """An ndarray backed by a memory-mapped file.

    Args:
        filename (str): Path to file to memory map.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (DTypeLike, optional): The value's data type. Defaults to ``None``.
        value (Number | NDArray[np.number], optional): If a number, creates as this value. If
            ``None``, attaches. Defaults to ``None``.
    """

    def __init__(
        self,
        filename: str,
        shape: Optional[Union[int, Tuple[int]]] = None,
        dtype: Optional[DTypeLike] = None,
        value: Optional[Union[Number, NDArray[np.number]]] = None,
    ) -> None:
        super().__init__(filename, shape, dtype, value)

    def numpy(self) -> NDArray:
        """Get a numpy array backed by our internal memory mapped buffer.

        This is a method instead of being cached due to adventures in fork/spawn issues.

        Returns:
            NDArray[T]: Our internal buffer as an ndarray.
        """
        return np.ndarray(self.shape, self.dtype, self.mmap, self.offset)

    def __getitem__(self, index: IndexType) -> Union[T, NDArray]:
        """Get the item at the index.

        Args:
            index (IndexType): The index(es).

        Returns:
            T | NDArray[T]: The item(s).
        """
        return self.numpy()[index]

    def __setitem__(self, index: IndexType, item: Any) -> None:
        """Set the item at the index.

        Args:
            index (IndexType): The index(es).
            item (Number): The item(s).
        """
        self.numpy()[index] = item


def ndarray(
    filename: str,
    shape: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[DTypeLike] = None,
    value: Optional[Union[Number, NDArray[np.number]]] = None,
) -> MemMapNDArray[np.dtype]:
    """Get an ndarray backed by a memory-mapped file.

    Args:
        filename (str): Path to file to memory map.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (DTypeLike, optional): The value's data type. Defaults to ``None``.
        value (Number | NDArray[np.number], optional): If a number, creates as this value. If
            ``None``, attaches. Defaults to ``None``.

    Returns:
        MemMapNDArray[T]: An ndarray backed by a memory-mapped file.
    """
    return MemMapNDArray(filename, shape, dtype, value)
