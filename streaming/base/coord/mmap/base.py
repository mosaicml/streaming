# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Base functionality for sharing data across processes using mmap()."""

import os
from mmap import mmap
from typing import IO, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from streaming.base.coord.file.barrier import wait_for_creation

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


# For memory alignment purposes.
_biggest_uint = np.uint64


def _get_file_header_size(shape: Tuple[int]) -> int:
    """Get the expected size of the header in bytes.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
    """
    return (1 + 1 + len(shape)) * _biggest_uint().itemsize


def _encode_file_header(shape: Tuple[int], dtype: np.dtype) -> bytes:
    """Serialize shape/dtype info.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (DTypeLike): Normalized ndarray dtype.

    Returns:
        bytes: Header data.
    """
    max_itemsize = _biggest_uint().itemsize
    dtype_bytes = np.dtype(dtype).name.encode('utf-8')[:max_itemsize]
    pad_bytes = b'\0' * (max_itemsize - len(dtype_bytes))
    ndim_bytes = np.uint64(len(shape)).tobytes()
    shape_bytes = np.array(shape, np.uint64).tobytes()
    return dtype_bytes + pad_bytes + ndim_bytes + shape_bytes


def _read_file_header(file: IO[bytes]) -> Tuple[Tuple[int], np.dtype]:
    """Deserialize shape/dtype info.

    Args:
        data (bytes): Header data.

    Returns:
        Tuple[Tuple[int], Tuple[np.number]]: Shape and type.
    """
    max_itemsize = _biggest_uint().itemsize

    # Check against minimum possible length.
    to_read = 2 * max_itemsize
    data = file.read(to_read)
    if len(data) < to_read:
        raise ValueError(f'Header data is too short: read {len(data)} bytes, but need at the ' +
                         f'very least {to_read} bytes.')

    # Get dtype.
    part = data[:max_itemsize]
    part = part[:part.index(b'\0')]
    dtype_name = part.decode('utf-8')
    dtype = np.dtype(dtype_name)

    # Get ndim.
    part = data[max_itemsize:]
    ndim, = np.frombuffer(part, np.uint64)
    if ndim < 0:
        raise ValueError(f'Header ndim is negative: {ndim}.')

    # Check against required length in our case.
    to_read = ndim * max_itemsize
    data = file.read(to_read)
    if len(data) < to_read:
        offset = 2 * max_itemsize
        raise ValueError(f'Header data is too short: got {offset + len(data)} total bytes, but ' +
                         f'need {offset + to_read} total bytes.')

    # Get shape.
    arr = np.frombuffer(data, np.int64)
    shape = tuple(arr.tolist())

    return shape, dtype


def _write_sparse_file(shape: Tuple[int], dtype: np.dtype, filename: str) -> None:
    """Write a sparse file of the given size, which reads as zeros unless otherwise written.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (np.dtype): Normalized ndarray dtype.
        filename (str): Path to file.
    """
    # Write header to a temp file.
    tmp_filename = filename + '.tmp'
    header = _encode_file_header(shape, dtype)
    with open(tmp_filename, 'wb') as out:
        out.write(header)

    # Truncate to desired sparse size.
    file_size = _get_file_size(shape, dtype)
    os.truncate(tmp_filename, file_size)

    # Rename to final name.
    os.rename(tmp_filename, filename)


def _write_dense_file(arr: NDArray[np.number], filename: str) -> None:
    """Write a regular file with the given ndarray.

    Args:
        arr (NDArray[np.number]): Array to write.
        filename (str): Path to file.
    """
    # Write header and body to a temp file.
    tmp_filename = filename + '.tmp'
    with open(tmp_filename, 'wb') as out:
        header = _encode_file_header(arr.shape, arr.dtype)
        out.write(header)
        out.write(arr.tobytes())

    # Rename to final name.
    os.rename(tmp_filename, filename)


def _write_file(arr: NDArray[np.number], shape: Tuple[int], filename: str) -> None:
    """Write the ndarray as either a regular or sparse file.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (np.dtype): Normalized ndarray dtype.
        filename (str): Path to file.
    """
    if (arr == 0).all():
        _write_sparse_file(shape, arr.dtype, filename)
    else:
        if arr.size == 1:
            arr = arr.flatten()
            arr = arr.repeat(np.prod(shape))
            arr = arr.reshape(shape)
        _write_dense_file(arr, filename)


def _get_file_body_size(shape: Tuple[int], dtype: np.dtype) -> int:
    """Get the expected size of the body in bytes.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (np.dtype): Normalized ndarray dtype.
    """
    numel = int(np.prod(shape)) if shape else 1
    norm_dtype = np.dtype(dtype)
    return numel * norm_dtype.itemsize


def _get_file_size(shape: Tuple[int], dtype: np.dtype) -> int:
    """Get the expected size of the file in bytes.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (np.dtype): Normalized ndarray dtype.
    """
    header_size = _get_file_header_size(shape)
    body_size = _get_file_body_size(shape, dtype)
    return header_size + body_size


def _create_file(
    filename: str,
    value: Union[Number, NDArray[np.number]],
    shape: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[np.dtype] = None,
) -> Tuple[Tuple[int], np.dtype, int]:
    """Create the file backing the memory mapping given optional explicit shape and dtype.

    Args:
        filename (str): Path to file to memory map.
        value (Number | NDArray[np.number], optional): Creates as this value.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (np.dtype, optional): Exact required data type, if known in advance. Defaults to
            ``None``.

    Returns:
        Tuple[Tuple[int], np.dtype, int]: The file's exact shape, dtype, and offset.
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
    _write_file(arr, exact_shape, filename)

    # Get the offset of the array part.
    offset = _get_file_header_size(exact_shape)

    return exact_shape, arr.dtype, offset


def _check_file(
    filename: str,
    shape: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[np.dtype] = None,
) -> Tuple[Tuple[int], np.dtype, int]:
    """Validate the file backing the memory mapping given optional explicit shape and dtype.

    Args:
        filename (str): Path to file to memory map.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (np.dtype, optional): Exact required data type, if known in advance. Defaults to
            ``None``.

    Returns:
        Tuple[Tuple[int], np.dtype, int]: The file's exact shape, dtype, and offset.
    """
    # The file must already exist.
    if not os.path.exists(filename):
        raise ValueError(f'`value` was not provided, so attaching the file, but it does not ' +
                         f'exist: {filename}.')

    # Read the shape and dtpye in the file header.
    with open(filename, 'rb') as file:
        got_shape, got_dtype = _read_file_header(file)

    # From those, derive the expected file size, and compare to actual.
    want_size = _get_file_size(got_shape, got_dtype)
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

    # Get the offset of the array part.
    offset = _get_file_header_size(got_shape)

    return got_shape, got_dtype, offset


def _ensure_file(
    filename: str,
    shape: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[np.dtype] = None,
    value: Optional[Union[Number, NDArray[np.number]]] = None,
) -> Tuple[Tuple[int], np.dtype, int]:
    """Create the file backing the memory mapping given optional explicit shape and dtype.

    Args:
        filename (str): Path to file to memory map.
        shape (int | Tuple[int], optional): Exact required number of elements along each axis, if
            known in advance. At most one wildcard ``-1`` may be used. Defaults to ``None``.
        dtype (np.dtype, optional): Exact required data type, if known in advance. Defaults to
            ``None``.
        value (Number | NDArray[np.number], optional): If a number, creates as this value. If
            ``None``, attaches. Defaults to ``None``.

    Returns:
        Tuple[Tuple[int], np.dtype, int]: The file's exact shape, dtype, and offset.
    """
    if value is not None:
        return _create_file(filename, value, shape, dtype)
    else:
        return _check_file(filename, shape, dtype)


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
    """

    def __init__(
        self,
        filename: str,
        shape: Optional[Union[int, Tuple[int]]] = None,
        dtype: Optional[DTypeLike] = None,
        value: Optional[Union[Number, NDArray[np.number]]] = None,
        timeout: Optional[float] = 60,
        poll_interval: float = 0.007,
    ) -> None:
        norm_dtype = np.dtype(dtype) if dtype is not None else None
        self.shape, self.dtype, self.offset = _ensure_file(filename, shape, norm_dtype, value)

        if value is None:
            wait_for_creation(filename, timeout, poll_interval)

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
