# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Share a buffer across processes using mmap()."""

from typing import Optional

import numpy as np

from streaming.base.coord.mmap.base import MemMap

__all__ = ['MemMapBuffer', 'buffer']


class MemMapBuffer(MemMap[np.dtype]):
    """A buffer backed by a memory-mapped file.

    Args:
        filename (str): Path to file to memory map.
        size (int, optional): Exact required size in bytes, if known in advance. Defaults to
            ``None``.
        value (bytes, optional): If provided, creates as this value. If ``None``, attaches.
            Defaults to ``None``.
    """

    def __init__(
        self,
        filename: str,
        size: Optional[int] = None,
        value: Optional[bytes] = None,
    ) -> None:
        norm_value = np.frombuffer(value, np.uint8) if value is not None else None
        super().__init__(filename, size, np.uint8, norm_value)

    def __len__(self) -> int:
        """Get the number of bytes in the buffer.

        Returns:
            int: Number of bytes in the buffer.
        """
        return self.shape[0]

    def __getitem__(self, index: slice) -> bytes:
        """Get the data at the index(es).

        Args:
            index (slice): The index(es).

        Returns:
            bytes: The data.
        """
        return self.mmap[index]

    def __setitem__(self, index: slice, data: bytes) -> None:
        """Set the data at the index(es).

        Args:
            index (int | slice): The index(es).
            data (bytes): The data.
        """
        self.mmap[index] = data


def buffer(
    filename: str,
    size: Optional[int] = None,
    value: Optional[bytes] = None,
) -> MemMapBuffer:
    """Get a buffer backed by a memory-mapped file.

    Args:
        filename (str): Path to file to memory map.
        size (int, optional): Exact required size in bytes, if known in advance. Defaults to
            ``None``.
        value (bytes, optional): If provided, creates as this value. If ``None``, attaches.
            Defaults to ``None``.

    Returns:
        MemMapBuffer: A buffer backed by a memory-mapped file.
    """
    return MemMapBuffer(filename, size, value)
