# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share a single number across processes using mmap()."""

import os
from typing import Generic

from typing_extensions import Self

from streaming.base.coord.mmap.array import DType, MMapArray

__init__ = ['MMapNumber']


class MMapNumber(Generic[DType]):
    """Share a single number across processes using mmap().

    Args:
        filename (str): File backing the internal MMapArray.
        dtype (DType): Data type of the number.
    """

    def __init__(self, filename: str, dtype: DType) -> None:
        self.arr = MMapArray(filename, dtype, 1)

    @classmethod
    def create(cls, filename: str, dtype: DType) -> Self:
        """Create and load an MMapNumber from scratch.

        Args:
            filename (str): File backing the internal MMapArray.
            dtype (DType): Data type of the number.
        """
        if os.path.exists(filename):
            raise ValueError('File already exists: {filename}.')

        MMapArray._write(filename, dtype, 1)
        return cls(filename, dtype)

    def get(self) -> DType:
        """Get our value.

        Returns:
            DType: Our value.
        """
        return self.arr[0]

    def set(self, value: DType) -> None:
        """Set our value.

        Args:
            value (DType): Our new value.
        """
        self.arr[0] = value
