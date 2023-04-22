# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A numpy array of predetermined shape that lives in shared memory."""

from typing import Any, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from streaming.base.shared.memory import SharedMemory


class SharedArray:
    """A numpy array of predetermined shape that lives in shared memory.

    Args:
        name (str): Its name in shared memory.
        shape (Union[int, Tuple[int]]): Shape of the array.
        dtype (np.dtype): Dtype of the array.
    """

    def __init__(self, name: str, shape: Union[int, Tuple[int]], dtype: type) -> None:
        size = np.prod(shape) * dtype(0).nbytes
        self.shm = SharedMemory(name=name, size=size)
        self.shape = np.empty(shape).shape
        self.dtype = dtype

    def array(self) -> NDArray:
        """Get as a numpy array.

        We can't just internally store and use this numpy array shared memory wrapper because it's
        not compatible with spawn.
        """
        return np.ndarray(self.shape, buffer=self.shm.buf, dtype=self.dtype)

    def __getitem__(self, index: Any) -> Any:
        """Get the scalar(s) at the given index, slice, or array of indices.

        Args:
            index (Any): The index, slice, or array of indices.

        Returns:
            The scalar(s) at the given location(s).
        """
        arr = np.ndarray(self.shape, buffer=self.shm.buf, dtype=self.dtype)
        return arr[index]

    def __setitem__(self, index: Any, value: Any) -> Any:
        """Set the scalar(s) at the given index, slice, or array of indices.

        Args:
            index (Any): The index, slice, or array of indices.
            value (Any): The scalar(s) at the given location(s).
        """
        arr = np.ndarray(self.shape, buffer=self.shm.buf, dtype=self.dtype)
        arr[index] = value
