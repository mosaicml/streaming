# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share an array across processes using mmap()."""

import os
from mmap import mmap
from typing import Generic, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from streaming.base.coord.mmap.base import ensure_file

__all__ = ['MMapArray']

DType = TypeVar('DType', bound=np.number)

IndexType = Union[int, slice, NDArray[np.integer]]
DataType = Union[DType, NDArray[DType]]


class MMapArray(Generic[DType]):
    """Share an array across processes using mmap().

    Args:
        mode (str): Whether to ``create``, ``replace``, or ``attach``. Defaults to ``attach``.
        filename (str): Path to memory-mapped file.
        shape (int | Tuple[int], optional): Exact required shape, if known in advance. At most one
            wildcard ``-1`` is acceptable.
        dtype (DType): Data type of the number.
    """

    def __init__(
        self,
        *,
        mode: str = 'attach',
        filename: str,
        shape: Optional[Union[int, Tuple[int]]] = None,
        dtype: DType,
    ) -> None:
        self.mode = mode
        self.filename = filename
        self.shape = ensure_file(mode, filename, shape, dtype.nbytes)
        self.dtype = dtype

        self.file = open(filename, 'r+b', 0)
        self.data = mmap(self.file.fileno(), 0)

    def __len__(self) -> int:
        """Get the number of elements in the first axis of the array.

        Returns:
            int: Length of the first axis of the array.
        """
        return int(self.shape[0])

    def as_array(self) -> NDArray[DType]:
        """Get a numpy array backed by our internal memory mapped buffer.

        This is a method instead of being cached due to adventures in fork/spawn issues.

        Returns:
            NDArray[DType]: Our internal buffer as an ndarray.
        """
        return np.ndarray(self.shape, buffer=self.data, dtype=self.dtype)

    def __getitem__(self, index: IndexType) -> DataType:
        """Get the item at the index.

        Args:
            index (IndexType): The index(es).

        Returns:
            DataType; The item(s).
        """
        return self.as_array()[index]

    def __setitem__(self, index: IndexType, item: DataType) -> None:
        """Set the item at the index.

        Args:
            index (IndexType): The index(es).
            item (DataType): The item(s).
        """
        self.as_array()[index] = item

    def flush(self) -> None:
        """Flush the mmap."""
        self.data.flush()

    def close(self) -> None:
        """Close the mmap and file handle."""
        if not self.data.closed:
            self.data.close()

        if not self.file.closed:
            self.file.close()

    def delete(self) -> None:
        """Ensure everything is closed, then delete the file."""
        self.close()
        os.remove(self.filename)
