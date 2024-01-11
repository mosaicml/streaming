# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Base functionality for sharing data across processes using mmap()."""

import os
from typing import Optional, Tuple, Union

import numpy as np

__all__ = ['ensure_file']


def _normalize_shape(shape: Optional[Union[int, Tuple[int]]]) -> \
        Tuple[Tuple[int], int, Optional[int]]:
    """Normalize and validate a shape argument.

    Args:
        shape (int | Tuple[int], optional): Input shape.

    Returns:
        Tuple[Tuple[int], int, Optional[int]]: Normalized shape, number of elements without the
            wildcard if present, and bytes per element.
    """
    if shape is None:
        shape = -1,
    elif isinstance(shape, int):
        shape = shape,

    num_wild = 0
    for dim in shape:
        if dim == -1:
            num_wild += 1
        elif dim < 1:
            raise ValueError(f'Each dimension must be a positive integer, with at most one ' +
                             f'wildcard, but got shape: {shape}.')

    if 1 < num_wild:
        raise ValueError(f'Shape contains multiple ({num_wild}) wildcards: {shape}.')

    numel = int(np.prod(shape))
    if numel < 0:
        numel = -numel
        wild_index = shape.index(-1)
    else:
        wild_index = None

    return shape, numel, wild_index


def ensure_file(mode: str, filename: str, shape: Optional[Union[int, Tuple[int]]],
                unit: int) -> Tuple[int]:
    """Ensure file existence and size according to mode.

    Args:
        mode (str): Whether to ``create``, ``replace``, or ``attach``. Defaults to ``attach``.
        filename (str): Path to memory-mapped file.
        shape (int | Tuple[int], optional): Exact required number of units, along each axis, if
            known in advance. At most one wildcard ``-1`` is acceptable.
        unit (int): Stride of a single value in bytes.

    Returns:
        int: Resulting exact shape.
    """
    want_shape, want_numel, want_wild_index = _normalize_shape(shape)

    if unit < 1:
        raise ValueError(f'{unit} must be a positive integer, but got: {unit}.')

    # Normalize file existence by mode.
    if mode == 'create':
        if os.path.exists(filename):
            raise ValueError(f'File alreadfy exists: {filename}.')
    elif mode == 'replace':
        if os.path.exists(filename):
            os.remove(filename)
    elif mode == 'attach':
        if not os.path.exists(filename):
            raise ValueError(f'File does not exist: {filename}.')
    else:
        modes = {'create', 'replace', 'attach'}
        raise ValueError(f'`mode` must be either replace,one of {sorted(modes)}, but got: {mode}.')

    # Perform the work.
    if os.path.exists(filename):
        # Use size info to validate the pre-existing file.
        got_size = os.stat(filename).st_size
        if want_wild_index is None:
            want_size = want_numel * unit
            if got_size != want_size:
                raise ValueError(f'File is the wrong size: file {filename}, expected shape ' +
                                 f'{want_shape}, expected unit {unit}, expected size ' +
                                 f'{want_size}, actual size {got_size}.')
            got_shape = want_numel,
        else:
            want_size = want_numel * unit
            if got_size % want_size:
                raise ValueError(f'File size is not evenly divisible: file {filename}, expected ' +
                                 f'shape {want_shape}, expected unit {unit}, expected size to ' +
                                 f'be divisible by {want_size}.')
            wild_value = got_size // want_size
            got_shape = list(want_shape)
            got_shape[want_wild_index] = wild_value
            got_shape = tuple(got_shape)
    else:
        # Use size info to create the (initially sparse) file.
        if want_wild_index is not None:
            raise ValueError(f'You must provide `shape`, without wildcards, in order to size ' +
                             f'the file: {filename}.')
        with open(filename, 'wb') as out:
            out.write(b'')
        os.truncate(filename, want_numel * unit)
        got_shape = want_shape

    # Return resulting exact shape.
    return got_shape
