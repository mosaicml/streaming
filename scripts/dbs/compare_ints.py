# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Chart int serialization efficiency (byte widths when serialized).

Results are compared:
- Across many orders of magnitude (rows).
- Across many DBS types (columns).
"""

import json
import pickle
from argparse import ArgumentParser, Namespace
from typing import Iterator, Optional

import numpy as np

uint_dtypes = np.uint8, np.uint16, np.uint32, np.uint64
int_dtypes = np.int8, np.int16, np.int32, np.int64


def parse_args() -> Namespace:
    """Porse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--powers_of_ten',
                      type=int,
                      default=24,
                      help='Size of the range of ints to chart, both in the positive and ' +
                      'negative directions')
    args.add_argument('--col_width',
                      type=int,
                      default=3,
                      help='Widths of the columns describing int serialized sizes')
    return args.parse_args()


def norm_len(text: str, size: int) -> str:
    """Normalie text length to the given size by padding. Must fit.

    Args:
        text (str): Original text.
        size (int): Desired text length.

    Returns:
        str: Normalized text.
    """
    if size < len(text):
        raise ValueError(f'Field is too big for its column: {text} (len(text) vs {size} chrs).')
    return text.rjust(size)


def dtype_to_col_name(dtype: type, col_width: int) -> str:
    """Convert numpy data type to fixed-length column name.

    Args:
        dtype (type): Numpy data type.
        col_width (int): Pad to desired column width (must fit).

    Returns
        str: Fixed-length column name.
    """
    s = dtype.__name__
    s = s.replace('uint', 'u')
    s = s.replace('int', 'i')
    return norm_len(s, col_width)


def each_col_name(powers_of_ten: int, col_width: int) -> Iterator[str]:
    """Get each column name.

    Args:
        powers_of_ten (int): Numbeer of powers of ten to chart, which affects some column widths.
        col_width (int): Pad to desired column width (must fit).

    Returns:
        Iterator[str]: Iterator over column names.
    """
    num_commas = powers_of_ten // 3
    yield 'int'.rjust(1 + powers_of_ten + num_commas)
    for dtype in uint_dtypes + int_dtypes:
        yield dtype_to_col_name(dtype, col_width)
    yield norm_len('jsn', col_width)
    yield norm_len('pkl', col_width)


def chr_to_div(ch: str) -> str:
    """Convert a character of header to a character of divider.

    Args:
        ch (str): Character of header.

    Returns:
        str: Character of divider.
    """
    if ch == ' ':
        return ' '
    elif ch == '|':
        return '+'
    else:
        return '-'


def each_int(powers_of_ten: int) -> Iterator[int]:
    """Get each int to chart.

    Args:
        powers_of_ten (int): Number of powers of ten to chart, both positive and negative.

    Returns:
        Iterator[int]: Iterator over ints (rows).
    """
    for i in filter(bool, range(-powers_of_ten, powers_of_ten + 1)):
        mul = 1 if 0 <= i else -1
        exp = abs(i) - 1
        yield mul * 10**exp


def get_size_as(val: int, dtype: type) -> Optional[int]:
    """Get the size of the int as a dtype, if applicable.

    Args:
        val (int): Int to convert.
        dtype (type): Numpy data type to convert to.

    Returns:
        Optional[int]: Data type size in bytes on success, or None on failure.
    """
    try:
        np_val = dtype(val)
    except OverflowError:
        return None
    if val != np_val:
        return None
    return np_val.nbytes


def each_field(val: int, powers_of_ten: int, col_width: int) -> Iterator[str]:
    """Each each field str to chsrt.

    Args:
        val (int): Int to convert to various types/dtypes.
        powers_of_ten (int): Number of powers of ten to chart, both positive and negative.
        col_width (int): Widths of the columns showing the various serialized sizes of the int.

    Returns:
        Iterator[str}: Each field of a row of the chart.
    """
    val_str = f'{val:,}'
    num_commas = powers_of_ten // 3
    yield norm_len(val_str, 1 + powers_of_ten + num_commas)
    for dtype in uint_dtypes + int_dtypes:
        size = get_size_as(val, dtype)
        size_str = str(size) if size else ''
        yield norm_len(size_str, col_width)
    jsn_str = json.dumps(val)
    jsn_data = jsn_str.encode('utf-8')
    jsn_field = str(len(jsn_data))
    yield norm_len(jsn_field, col_width)
    pkl_data = pickle.dumps(val)
    pkl_field = str(len(pkl_data))
    yield norm_len(pkl_field, col_width)


def main(args: Namespace) -> None:
    """Chart how efficiently int serializes across a range of values and types/dtypes."""
    header = ' | '.join(each_col_name(args.powers_of_ten, args.col_width))
    print(header)
    divider = ''.join(map(chr_to_div, header))
    print(divider)
    for val in each_int(args.powers_of_ten):
        print(' | '.join(each_field(val, args.powers_of_ten, args.col_width)))


if __name__ == '__main__':
    main(parse_args())
