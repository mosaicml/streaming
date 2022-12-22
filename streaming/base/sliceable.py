# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A smart-indexable array of items."""

from typing import Any, Callable, Iterator, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Sliceable:
    """A smart-indexable array of items.

    Args:
        get_item (Callable): Method to get one item given its index ("dumb __getitem__").
    """

    def __init__(self, get_item: Callable, get_len: Callable) -> None:
        self._get_item = get_item
        self._get_len = get_len

    def _each_slice_index(self, at: slice) -> Iterator[int]:
        """Get each slice index.

        Args:
            at (slice): The slice.

        Returns:
            Iterator[int]: Its indices.
        """
        size = self._get_len()

        if at.start is None:
            start = 0
        else:
            if at.start < 0:
                start = at.start + size
            else:
                start = at.start
            assert 0 <= start < size

        if at.stop is None:
            stop = size
        else:
            if at.stop < 0:
                stop = at.stop + size
            else:
                stop = at.stop
            assert 0 <= stop < size

        if at.step is None:
            step = 1
        else:
            step = at.step

        yield from range(start, stop, step)

    def __getitem__(self, at: Union[int, slice, List[int], Tuple[int], NDArray[np.int64]]) -> Any:
        """Get item(s) by index, slice, list, tuple, or numpy array.

        Args:
            at (int | slice | List[int] | Tuple[int] | NDArray[np.int64]): Sample index(es).

        Returns:
            Any: An item if passed an int, or recursive list(s) or tuple(s) of items otherwise.
        """
        if isinstance(at, (int, np.integer)):
            if at < 0:
                at += self._get_len()
            return self._get_item(at)
        elif isinstance(at, slice):
            items = []
            for idx in self._each_slice_index(at):
                item = self._get_item(idx)
                items.append(item)
            return items
        elif isinstance(at, list):
            items = []
            for sub in at:
                item = self.__getitem__(sub)
                items.append(item)
            return items
        elif isinstance(at, tuple):
            items = []
            for sub in at:
                item = self.__getitem__(sub)
                items.append(item)
            return tuple(items)
        elif isinstance(at, np.ndarray):  # pyright: ignore
            items = []
            for sub in at:
                item = self.__getitem__(sub)
                items.append(item)
            return items
        else:
            raise ValueError(f'Unsupported argument type passed to __getitem__: {type(at)}.')
