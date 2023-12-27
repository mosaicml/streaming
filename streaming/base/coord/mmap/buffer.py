# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share a buffer across processes using mmap()."""

import os
from mmap import mmap
from typing import Optional

from typing_extensions import Self

__all__ = ['MMapBuffer']


class MMapBuffer:
    """Share a buffer across processes using mmap().

    Args:
        filename (str): File backing this buffer.
        size (int, optional): Exact required size, if known in advance.
    """

    def __init__(self, filename: str, size: Optional[int] = None) -> None:
        self.filename = filename
        self.size = self._ensure(filename, size)
        self.file = open(filename, 'r+b', 0)
        self.data = mmap(self.file.fileno(), 0)

    @classmethod
    def _ensure(cls, filename: str, size: Optional[int]) -> int:
        """Ensure the file exists, get its actual size, and compare to expected size.

        Args:
            filename (str): File backing this buffer.
            size (int, optional): Exact required size, if known in advance.

        Returns:
            int: Exact observed file size.
        """
        if size is None:
            if os.path.exists(filename):
                return os.stat(filename).st_size
            else:
                raise ValueError('File does not exist: {filename}.')

        if not os.path.exists(filename):
            raise ValueError('File does not exist: {filename}.')

        stat = os.stat(filename)
        if stat.st_size != size:
            raise ValueError(f'File size mismatch: file {filename}, expected {size}, got ' +
                             f'{stat.st_size}.')

        return size

    @classmethod
    def _write(cls, filename: str, size: int) -> None:
        """Initialize the buffer to all nulls of the specified size.

        Args:
            filename (str): File backing this bufffer.
            size (int): Size in bytes.
        """
        data = b'\0' * size
        with open(filename, 'wb') as out:
            out.write(data)

    @classmethod
    def create(cls, filename: str, size: int) -> Self:
        """Create and load an MMapBuffer from scratch.

        Args:
            filenmae (str): File backing this buffer.
            size (int): Size of the buffer/file.

        Returns:
            Self: Loaded MMapBuffer.
        """
        if os.path.exists(filename):
            raise ValueError('File already exists: {filename}.')

        cls._write(filename, size)
        return cls(filename)

    def __len__(self) -> int:
        """Get the number of bytes in the buffer.

        Returns:
            int: Number of bytes in the buffer.
        """
        return self.size
