# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A numpy array of predetermined shape and dtype that lives in shared memory."""

from typing import Any, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from streaming.base.shared.memory import SharedMemory


class SharedArray:
    """A numpy array of predetermined shape and dtype that lives in shared memory.

    Args:
        shape (Union[int, Tuple[int]]): Shape of the array.
        dtype (type): Dtype of the array.
        name (str): Its name in shared memory.
    """

    def __init__(self, shape: Union[int, Tuple[int]], dtype: type, name: str) -> None:
        self.shape = np.empty(shape).shape
        self.dtype = dtype
        self.name = name
        size = int(np.prod(shape) * dtype(0).nbytes)
        self.shm = SharedMemory(name=name, size=size)

    def numpy(self) -> NDArray:
        """Get as a numpy array.

        We can't just internally store and use this numpy array shared memory wrapper because it's
        not compatible with spawn.
        """
        return np.ndarray(self.shape, buffer=self.shm.buf, dtype=self.dtype)

    def __len__(self) -> int:
        """Get the length (i.e., size along the first axis).

        Returns:
            int: The length.
        """
        return int(self.shape[0])

    def __getitem__(self, index: Any) -> Any:
        """Get the scalar(s) at the given index, slice, or array of indices.

        Args:
            index (Any): The index, slice, or array of indices.

        Returns:
            The scalar(s) at the given location(s).
        """
        arr = self.numpy()
        return arr[index]

    def __setitem__(self, index: Any, value: Any) -> Any:
        """Set the scalar(s) at the given index, slice, or array of indices.

        Args:
            index (Any): The index, slice, or array of indices.
            value (Any): The scalar(s) at the given location(s).
        """
        arr = self.numpy()
        arr[index] = value
