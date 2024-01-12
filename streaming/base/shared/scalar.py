# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A numpy scalar of predetermined dtype that lives in shared memory."""

from typing import Any

from streaming.base.shared.array import SharedArray


class SharedScalar:
    """A numpy scalar of predetermined dtype that lives in shared memory.

    Args:
        dtype (type): Dtype of the array.
        name (str): Its name in shared memory.
    """

    def __init__(self, dtype: type, name: str) -> None:
        self.dtype = dtype
        self.name = name
        self.arr = SharedArray(1, dtype, name)

    def get(self) -> Any:
        """Get the value.

        Returns:
            Any: The value.
        """
        return self.arr[0]

    def set(self, value: Any) -> None:
        """Set the value.

        Args:
            value (Any): The value.
        """
        self.arr[0] = value
