# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""An ordered set that can be used as an LRU cache."""

from collections import OrderedDict
from typing import Any


class LastUsedOrderedSet(OrderedDict):
    """An ordered dict that can be used as an LRU cache.

    This is a subclass of OrderedDict, with some LRU-specific functions and all values as ``None``.
    """

    def setitem(self, key: Any, move_to_end: bool = True):
        """Set/add an item.

        Args:
            key (Any): key to be added.
            move_to_end (bool, optional): whether to move the item to the end, signifying most
                recent access. Defaults to ``True``.
        """
        super().__setitem__(key, None)
        self.move_to_end(key, last=move_to_end)

    def popLRU(self):
        """Pop the least recently used item (located at the front)."""
        return self.popitem(last=False)[0]

    def setuse(self, key: Any):
        """Mark an item as used, moving it to the end.

        Args:
            key (Any): key of element to move to the end, signifying most recent access.
        """
        self.setitem(key)
