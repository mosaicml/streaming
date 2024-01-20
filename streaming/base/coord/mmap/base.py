# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Base functionality for sharing data across processes using mmap()."""

import os
from mmap import mmap
from typing import Generic, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from streaming.base.coord.file.waiting import wait_for_creation
from streaming.base.coord.mmap.file import (get_file_header_size, get_file_size, read_file_header,
                                            write_file)

__all__ = ['T', 'MemMap', 'Number']

Number = Union[int, float, complex, np.number]


def _get_wildcarded_shape(shape: Optional[Union[int, Tuple[int]]]) -> Optional[Tuple[int]]:
    """Normalize and validate a shape argument.

    Args:
        shape (int | Tuple[int], optional): Input shape.

    Returns:
        Tuple[int], optional: Normalized shape containing at least one dimension, and at most one
            wildcard, or else ``None`` to mean accept any shape.
    """
    if shape is None:
        return None

    if shape == ():
        wild_shape = 1,
    elif isinstance(shape, int):
        wild_shape = shape,
    else:
        wild_shape = shape

    num_wild = 0
    for dim in wild_shape:
        if dim == -1:
            num_wild += 1
        elif dim < 1:
            raise ValueError(f'Each dimension must be a positive integer, with at most one ' +
                             f'wildcard, but got shape: {shape}.')

    if 1 < num_wild:
        raise ValueError(f'Shape contains multiple ({num_wild}) wildcards: {shape}.')

    return wild_shape


def _get_exact_shape(shape: Optional[Union[int, Tuple[int]]]) -> Tuple[int]:
    """Normalize and validate a shape argument.

    Args:
        shape (int | Tuple[int], optional): Input shape.

    Returns:
        Tuple[int]: Normalized shape containing at least one dimension, and no wildcards.
    """
    if shape is None:
        exact_shape = 1,
    elif isinstance(shape, int):
        exact_shape = shape,
    else:
        exact_shape = shape

    for dim in exact_shape:
        if dim < 1:
            raise ValueError(f'Each dimension must be a positive integer, but got shape: {shape}.')

    return exact_shape


def _normalize_ndarray(value: NDArray[np.number]) -> NDArray[np.number]:
    """Normalize an input ndarray to rule out the zero ndim case.

    Args:
        value (NDArray[np.number]): Input ndarray.

    Returns:
        NDArray[np.number]: Normalized ndarray.
    """
    if not value.ndim:
        return np.expand_dims(value, 0)

    return value


def _normalize_number(value: Number, dtype: Optional[np.dtype]) -> NDArray[np.number]:
    """Normalize an input number (np.number, int, float, etc). to ndarray.

    Args:
        value (Number): Input number.
        dtype (np.dtype, optional): Explicitly provided dtype, if any.

    Returns:
        NDArray[np.number]: Normalized number.
    """
    if isinstance(value, np.number):
        arr = np.array([value])
    elif dtype is not None:
        arr = np.array([value], dtype)
    else:
        arr = np.array([value])
    return arr


def _is_shape_nonempty(shape: Tuple[int]) -> bool:
    """Tell whether the shape is empty on any axes.

    Args:
        shape (Tuple[int]): The shape.

    Returns:
        bool: Whether the shape is valid.
    """
    for dim in shape:
        if dim < 1:
            return False

    return True


def _accepts_shape(shape: Tuple[int], wildcarded_shape: Optional[Tuple[int]]) -> bool:
    """Tell whether the shape restriction accepts the observed shape.

    Args:
        shape (Tuple[int]): Observed shape.
        wildcarded_shape (Tuple[int], optional): Shape restriction, if provided.

    Returns:
        bool: Whether acceptable.
    """
    if wildcarded_shape is None:
        return True

    if len(shape) != len(wildcarded_shape):
        return False

    for dim, wild_dim in zip(shape, wildcarded_shape):
        if wild_dim != dim and wild_dim != -1:
            return False

    return True


def _accepts_dtype(have: np.dtype, want: Optional[np.dtype]) -> bool:
    """Tell whether the dtype restriction accepts the observed dtype.

    Args:
        have (np.dtype): Observed dtype.
        want (np.dtype, optional): Dtype restriction.

    Returns:
        bool: Whether acceptable.
    """
    if not want:
        return True

    return have == want


def _create_file(
    filename: str,
    value: Union[Number, NDArray[np.number]],
    shape: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[np.dtype] = None,
) -> Tuple[Tuple[int], np.dtype]:
    """Create the file backing the memory mapping given optional explicit shape and dtype.

    Args:
        filename (str): Path to file to memory map.
        value (Number | NDArray[np.number], optional): Creates as this value.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (np.dtype, optional): Exact required data type, if known in advance. Defaults to
            ``None``.

    Returns:
        Tuple[Tuple[int], np.dtype, int]: The file's exact shape and dtype.
    """
    # The file must not already exist.
    if os.path.exists(filename):
        raise ValueError(f'`value` was provided, so we are initializing the file to memory map ' +
                         f'ourselves. Out of caution, we require that the file not already ' +
                         f'exist. But it does: {filename}.')

    # What are we initializing it with?
    if isinstance(value, np.ndarray):
        # Normalize the ndarray to have at least one axis.
        arr = _normalize_ndarray(value)

        # Its normalized shape is the target shape.
        exact_shape = arr.shape

        # It must not have any zero-length axes.
        if not _is_shape_nonempty(arr.shape):
            raise ValueError(f'``value` must be non-empty on each axis, but got shape: {shape}.')

        # If shape is provided explicitly as well, they must match.
        wildcarded_shape = _get_wildcarded_shape(shape)
        if not _accepts_shape(arr.shape, wildcarded_shape):
            raise ValueError(f'`value` and `shape` were both provided, but they do not match: ' +
                             f'`value` shape {value.shape} vs explicit `shape` {shape}.')
    else:  # Number.
        # Normalize the Number to be an ndarray of shape (1,).
        arr = _normalize_number(value, dtype)

        # If shape is provided, broadcast the value to all elements, else one element.
        exact_shape = _get_exact_shape(shape)

    # If dtype is provided explicitly as well, they must match.
    if not _accepts_dtype(arr.dtype, dtype):
        raise ValueError(f'`value` and `dtype` were both provided, but they do not match: ' +
                         f'`value` dtype {arr.dtype} vs explicit `dtype` {dtype}.')

    # Create the file (dense or sparse).
    write_file(arr, exact_shape, filename)

    return exact_shape, arr.dtype


def _check_file(
    filename: str,
    shape: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[np.dtype] = None,
    timeout: Optional[float] = 30,
    tick: float = 0.007,
) -> Tuple[Tuple[int], np.dtype]:
    """Validate the file backing the memory mapping given optional explicit shape and dtype.

    Args:
        filename (str): Path to file to memory map.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (np.dtype, optional): Exact required data type, if known in advance. Defaults to
            ``None``.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.

    Returns:
        Tuple[Tuple[int], np.dtype]: The file's exact shape and dtype.
    """
    # Wait for the file to exist.
    wait_for_creation(filename, timeout, tick)

    # Read the shape and dtpye in the file header.
    with open(filename, 'rb') as file:
        got_shape, got_dtype = read_file_header(file)

    # From those, derive the expected file size, and compare to actual.
    want_size = get_file_size(got_shape, got_dtype)
    got_size = os.stat(filename).st_size
    if got_size != want_size:
        raise ValueError(f'`File size did not match what we expected given shape and dtype as ' +
                         f'stated in its header: file {filename}, shape {got_shape}, dtype ' +
                         f'{got_dtype}, expected size {want_size}, actual size {got_size}.')

    # If shape is provided explicitly as well, they must match.
    wildcarded_shape = _get_wildcarded_shape(shape)
    if not _accepts_shape(got_shape, wildcarded_shape):
        raise ValueError(f'`shape` was explicitly provided, but does not match what is in the ' +
                         f'file: `filename` {filename}, `shape` {shape}, actual shape {got_size}.')

    # If dtype is provided explicitly as well, they must match.
    if not _accepts_dtype(got_dtype, dtype):
        dtype_name = dtype.name if dtype else None
        raise ValueError(f'`dtype` was explicitly provided, but does not match what is in the ' +
                         f'file: `filename` {filename}, `dtype` {dtype_name}, actual dtype ' +
                         f'{got_dtype.name}.')

    return got_shape, got_dtype


def _ensure_file(
    filename: str,
    shape: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[np.dtype] = None,
    value: Optional[Union[Number, NDArray[np.number]]] = None,
    timeout: Optional[float] = 30,
    tick: float = 0.007,
) -> Tuple[Tuple[int], np.dtype]:
    """Create the file backing the memory mapping given optional explicit shape and dtype.

    Args:
        filename (str): Path to file to memory map.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (np.dtype, optional): Exact required data type, if known in advance. Defaults to
            ``None``.
        value (Number | NDArray[np.number], optional): If a number, creates as this value. If
            ``None``, attaches. Defaults to ``None``.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.

    Returns:
        Tuple[Tuple[int], np.dtype, int]: The file's exact shape and dtype.
    """
    if value is not None:
        return _create_file(filename, value, shape, dtype)
    else:
        return _check_file(filename, shape, dtype, timeout, tick)


T = TypeVar('T', bound=np.dtype)


class MemMap(Generic[T]):
    """An ndarray backed by a memory-mapped file.

    Format:

        [dtype: str, padded with nulls to uint64().itemsize]
        [ndim: uint64]
        [shape: ndim x uint64]
        [data: prod(shape) * dtype]

    Args:
        filename (str): Path to file to memory map.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (DTypeLike, optional): Exact required data type, if known in advance. Defaults to
            ``None``.
        value (Number | NDArray[np.number], optional): If a number, creates as this value. If
            ``None``, attaches. Defaults to ``None``.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.
    """

    def __init__(
        self,
        filename: str,
        shape: Optional[Union[int, Tuple[int]]] = None,
        dtype: Optional[DTypeLike] = None,
        value: Optional[Union[Number, NDArray[np.number]]] = None,
        timeout: Optional[float] = 30,
        tick: float = 0.007,
    ) -> None:
        norm_dtype = np.dtype(dtype) if dtype is not None else None
        self.shape, self.dtype = _ensure_file(filename, shape, norm_dtype, value, timeout, tick)
        self.offset = get_file_header_size(self.shape)
        self.filename = filename
        self.file = open(filename, 'r+b', 0)
        self.mmap = mmap(self.file.fileno(), 0)

    def flush(self) -> None:
        """Flush the mmap."""
        self.mmap.flush()

    def close(self) -> None:
        """Close the mmap and file handle."""
        if not self.mmap.closed:
            self.mmap.close()
        if not self.file.closed:
            self.file.close()

    def delete(self) -> None:
        """Ensure everything is closed, then delete the file."""
        self.close()
        os.remove(self.filename)

    def __del__(self) -> None:
        """Destructor."""
        self.close()
