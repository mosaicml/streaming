# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share an array across processes using mmap()."""

import os
from typing import Generic, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from streaming.base.coord.mmap.buffer import MMapBuffer

__all__ = ['MMapArray']

DType = TypeVar('DType', bound=np.number)

IndexType = Union[int, NDArray[np.integer]]


class MMapArray(Generic[DType]):
    """Share an array across processes using mmap().

    Args:
        filename (str): File backing the internal MMapBuffer.
        dtype (DType): Data type of the number.
        shape (int | Tuple[int], optional): Exact required shape, if known in advance.
    """

    def __init__(self,
                 filename: str,
                 dtype: DType,
                 shape: Optional[Union[int, Tuple[int]]] = None) -> None:
        self.filename = filename
        self.dtype = dtype
        self.shape, self.num_bytes = self._ensure(filename, dtype, shape)
        self.buf = MMapBuffer(filename, self.num_bytes)

    @classmethod
    def _ensure(cls,
                filename: str,
                dtype: DType,
                shape: Optional[Union[int, Tuple[int]]] = None) -> Tuple[Tuple[int], int]:
        """Ensure the file exists, get its actual size, and compare to expected shape and dtype.

        Args:
            filename (str): File backing the internal MMapBuffer.
            dtype (DType): Data type of this array.
            shape (int | Tuple[int], optional): Exact required shape, if known in advance.

        Returns:
            Tuple[Tuple[int], int]: Pair of (array shape, file size).
        """
        if shape is None:
            if os.path.exists(filename):
                file_size = os.stat(filename).st_size
                dtype_size = dtype.nbytes
                if file_size % dtype_size:
                    raise ValueError(f'Data type size does not evenly divide file size: file ' +
                                     f'{filename}, file size {file_size}, dtype {dtype}, dtype ' +
                                     f'size {dtype_size}.')
                numel = file_size // dtype_size
                shape = numel,
                return shape, file_size
            else:
                raise ValueError(f'File does not exist: {filename}.')

        if not os.path.exists(filename):
            raise ValueError(f'File does not exist: {filename}.')

        if isinstance(shape, int):
            shape = shape,

        for dim in shape:
            if dim < 1:
                raise ValueError('Invalid shape: {shape}.')

        numel = int(np.prod(shape))
        dtype_size = dtype.nbytes
        file_size = numel * dtype_size
        stat = os.stat(filename)
        if stat.st_size != file_size:
            raise ValueError(f'File size mismatch: file {filename}, shape {shape}, dtype ' +
                             f'{dtype}, dtype size {dtype_size}, expected file size ' +
                             f'{file_size}, got file size {stat.st_size}.')

        return shape, file_size

    @classmethod
    def _write(cls, filename: str, dtype: DType, shape: Union[int, Tuple[int]]) -> None:
        """Initialize the array to all zeros of the specified shape and dtype.

        Args:
            filename (str): File backing the internal MMapBuffer.
            dtype (DType): Data type of this array.
            shape (int | Tupel[int]): Shape of this array.
        """
        if isinstance(shape, int):
            shape = shape,
        size = int(np.prod(shape)) * dtype.nbytes
        MMapBuffer._write(filename, size)

    @classmethod
    def create(cls, filename: str, dtype: DType, shape: Union[int, Tuple[int]]) -> Self:
        """Create and load a MMapArray from scratch.

        Args:
            filename (str): File backing the internal MMapBuffer.
            dtype (DType): Data type of this array.
            shape (int | Tupel[int]): Shape of this array.

        Returns:
            Self: Loaded MMapArray.
        """
        if os.path.exists(filename):
            raise ValueError('File already exists: {filename}.')

        cls._write(filename, dtype, shape)
        return cls(filename, dtype)

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
        return np.ndarray(self.shape, buffer=self.buf.data, dtype=self.dtype)

    def __getitem__(self, index: IndexType) -> DType:
        """Get the item at the index.

        Args:
            index (IndexType): The index.

        Returns:
            DType; The item.
        """
        return self.as_array()[index]

    def __setitem__(self, index: IndexType, item: DType) -> None:
        """Set the item at the index.

        Args:
            index (IndexType): The index.
            item (DType): The item.
        """
        self.as_array()[index] = item
