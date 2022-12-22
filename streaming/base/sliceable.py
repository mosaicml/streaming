# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A smart-indexable array of items."""

from typing import Any, Callable, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Sliceable:
    """A smart-indexable array of items.

    Args:
        get_item (Callable): Method to get one item given its index ("dumb __getitem__").
    """

    def __init__(self, get_item: Callable) -> None:
        self.get_item = get_item

    def __getitem__(self, at: Union[int, slice, List[int], Tuple[int], NDArray[np.int64]]) -> Any:
        """Get item(s) by index, slice, list, tuple, or numpy array.

        Args:
            at (int | slice | List[int] | Tuple[int] | NDArray[np.int64]): Sample index(es).

        Returns:
            Any: An item if passed an int, or recursive list(s) or tuple(s) of items otherwise.
        """
        if isinstance(at, (int, np.integer)):
            return self.get_item(at)
        elif isinstance(at, slice):
            items = []
            for index in range(at.start, at.stop, at.step):
                item = self.get_item(index)
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
