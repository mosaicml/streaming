# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share a buffer across processes using mmap()."""

from mmap import mmap
from typing import Optional

from streaming.base.coord.mmap.base import ensure_file

__all__ = ['MMapBuffer']


class MMapBuffer:
    """Share a buffer across processes using mmap().

    Args:
        mode (str): Whether to ``create``, ``replace``, or ``attach``. Defaults to ``attach``.
        filename (str): Path to memory-mapped file.
        size (int, optional): Exact required size, if known in advance. Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        mode: str = 'attach',
        filename: str,
        size: Optional[int] = None,
    ) -> None:
        self.mode = mode
        self.filename = filename
        self.size, = ensure_file(mode, filename, size, 1)
        self.file = open(filename, 'r+b', 0)
        self.data = mmap(self.file.fileno(), 0)

    def __len__(self) -> int:
        """Get the number of bytes in the buffer.

        Returns:
            int: Number of bytes in the buffer.
        """
        return self.size
