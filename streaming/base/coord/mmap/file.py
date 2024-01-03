# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Base functionality for sharing data across processes using mmap()."""

import os
from typing import IO, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'get_file_header_size', 'encode_file_header', 'get_file_body_size', 'encode_file_body',
    'get_file_size', 'encode_file', 'read_file_header', 'write_sparse_file', 'write_dense_file',
    'write_file'
]

# For memory alignment purposes.
_biggest_uint = np.uint64


def get_file_header_size(shape: Tuple[int]) -> int:
    """Get the expected size of the header in bytes.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
    """
    return (1 + 1 + len(shape)) * _biggest_uint().itemsize


def encode_file_header(shape: Tuple[int], dtype: np.dtype) -> bytes:
    """Serialize shape/dtype metadata.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (np.dtype): Normalized ndarray dtype.

    Returns:
        bytes: Header data.
    """
    max_itemsize = _biggest_uint().itemsize
    dtype_bytes = np.dtype(dtype).name.encode('utf-8')[:max_itemsize]
    pad_bytes = b'\0' * (max_itemsize - len(dtype_bytes))
    ndim_bytes = np.uint64(len(shape)).tobytes()
    shape_bytes = np.array(shape, np.uint64).tobytes()
    return dtype_bytes + pad_bytes + ndim_bytes + shape_bytes


def get_file_body_size(shape: Tuple[int], dtype: np.dtype) -> int:
    """Get the expected size of the body in bytes.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (np.dtype): Normalized ndarray dtype.
    """
    numel = int(np.prod(shape)) if shape else 1
    norm_dtype = np.dtype(dtype)
    return numel * norm_dtype.itemsize


def encode_file_body(arr: NDArray[np.number]) -> bytes:
    """Serialize data.

    Args:
        arr (NDArray[np.number]): Array to write.

    Returns:
        bytes: Body data.
    """
    return arr.tobytes()


def get_file_size(shape: Tuple[int], dtype: np.dtype) -> int:
    """Get the expected size of the file in bytes.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (np.dtype): Normalized ndarray dtype.
    """
    header_size = get_file_header_size(shape)
    body_size = get_file_body_size(shape, dtype)
    return header_size + body_size


def encode_file(arr: NDArray[np.number]) -> bytes:
    """Serialize metadata and data.

    Args:
        arr (NDArray[np.number]): Array to write.

    Returns:
        bytes: Body data.
    """
    header_data = encode_file_header(arr.shape, arr.dtype)
    body_data = encode_file_body(arr)
    return header_data + body_data


def read_file_header(file: IO[bytes]) -> Tuple[Tuple[int], np.dtype]:
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
    arr = np.frombuffer(part, np.uint64)
    ndim = int(arr[0])
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


def write_sparse_file(shape: Tuple[int], dtype: np.dtype, filename: str) -> None:
    """Write a sparse file of the given size, which reads as zeros unless otherwise written.

    Args:
        shape (Tuple[int]): Normalized ndarray shape.
        dtype (np.dtype): Normalized ndarray dtype.
        filename (str): Path to file.
    """
    # Write header to a temp file.
    tmp_filename = filename + '.tmp'
    header = encode_file_header(shape, dtype)
    with open(tmp_filename, 'wb') as out:
        out.write(header)

    # Truncate to desired sparse size.
    file_size = get_file_size(shape, dtype)
    os.truncate(tmp_filename, file_size)

    # Rename to final name.
    os.rename(tmp_filename, filename)


def write_dense_file(arr: NDArray[np.number], filename: str) -> None:
    """Write a regular file with the given ndarray.

    Args:
        arr (NDArray[np.number]): Array to write.
        filename (str): Path to file.
    """
    # Encode the file.
    data = encode_file(arr)

    # Write header and body to a temp file.
    tmp_filename = filename + '.tmp'
    with open(tmp_filename, 'wb') as out:
        out.write(data)

    # Rename to final name.
    os.rename(tmp_filename, filename)


def write_file(arr: NDArray[np.number], shape: Tuple[int], filename: str) -> None:
    """Write the ndarray as either a regular or sparse file.

    Args:
        arr (NDArray[np.number]): Array to write or value to broadcast.
        shape (Tuple[int]): Normalized ndarray shape.
        filename (str): Path to file.
    """
    if (arr == 0).all():
        write_sparse_file(shape, arr.dtype, filename)
    else:
        if arr.size == 1:
            arr = arr.flatten()
            arr = arr.repeat(np.prod(shape))
            arr = arr.reshape(shape)
        write_dense_file(arr, filename)
