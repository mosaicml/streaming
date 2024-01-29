# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A read-only list of items that can be fancy indexed like a numpy array."""

from typing import Any, Iterator, List, Union

import numpy as np
from numpy.typing import NDArray


class Array:
    """A read-only list of items that can be fancy indexed like a numpy array.

    It can index into instances of this class using ints, slices, numpy arrays, lists of ints,
    list of lists of ints, etc.

    We provide `__getitem__`. Subclasses must provide `size` and `get_item(0 <= idx < size)`.
    """

    @property
    def size(self) -> int:
        """Get the size of the array.

        Returns:
            int: Array size.
        """
        raise NotImplementedError

    def get_item(self, idx: int) -> Any:
        """Get the item at the index.

        Args:
            idx (int): The index.

        Returns:
            Any: The item.
        """
        raise NotImplementedError

    def _each_slice_index(self, at: slice) -> Iterator[int]:
        """Get each slice index.

        Args:
            at (slice): The slice.

        Returns:
            Iterator[int]: Its indices.
        """
        if at.start is None:
            start = 0
        else:
            start = at.start
            if -self.size <= start < 0:
                start += self.size

        if at.stop is None:
            stop = self.size
        else:
            stop = at.stop
            if -self.size <= stop < 0:
                stop += self.size

        if at.step is None:
            step = 1
        else:
            step = at.step

        if 0 < step:
            start = max(start, 0)
            stop = min(stop, self.size)
        else:
            stop = max(stop, -1)
            start = min(start, self.size - 1)

        yield from range(start, stop, step)

    def __getitem__(self, at: Union[int, slice, List[int], NDArray[np.int64]]) -> Any:
        """Get item(s) by index, slice, int list, or numpy array.

        Args:
            at (int | slice | List[int] | NDArray[np.int64]): Sample index(es).

        Returns:
            Any: An item if passed an int, or recursive int list(s) of items otherwise.
        """
        if isinstance(at, (int, np.integer)):
            if -self.size <= at < 0:
                at += self.size
            return self.get_item(at)
        elif isinstance(at, slice):
            items = []
            for idx in self._each_slice_index(at):
                item = self.get_item(idx)
                items.append(item)
            return items
        elif isinstance(at, list):
            items = []
            for sub in at:
                item = self.__getitem__(sub)
                items.append(item)
            return items
        elif isinstance(at, np.ndarray):  # pyright: ignore
            items = []
            for sub in at:
                item = self.__getitem__(sub)
                items.append(item)
            return items
        else:
            raise ValueError(f'Unsupported argument type passed to __getitem__: {type(at)}.')
