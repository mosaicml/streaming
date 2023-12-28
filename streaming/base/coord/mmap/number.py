# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share a single number across processes using mmap()."""

from mmap import mmap
from typing import Generic

import numpy as np

from streaming.base.coord.mmap.array import DType
from streaming.base.coord.mmap.base import ensure_file

__init__ = ['MMapNumber']


class MMapNumber(Generic[DType]):
    """Share a single number across processes using mmap().

    Args:
        mode (str): Whether to ``create``, ``replace``, or ``attach``. Defaults to ``attach``.
        filename (str): Path to memory-mapped file.
        dtype (DType): Data type of the number.
    """

    def __init__(
        self,
        *,
        mode: str = 'attach',
        filename: str,
        dtype: DType,
    ) -> None:
        self.mode = mode
        self.filename = filename
        ensure_file(mode, filename, 1, 1)
        self.dtype = dtype
        self.file = open(filename, 'r+b', 0)
        self.data = mmap(self.file.fileno(), 0)

    def get(self) -> DType:
        """Get our value.

        Returns:
            DType: Our value.
        """
        return np.frombuffer(self.data, self.dtype)[0]

    def set(self, value: DType) -> None:
        """Set our value.

        Args:
            value (DType): Our new value.
        """
        self.data[:] = value.tobytes()
