# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Conversions between human-friendly string forms and int/float."""

from typing import List, Union

__all__ = ['bytes_to_int', 'get_list_arg', 'number_abbrev_to_int']


def get_list_arg(text: str) -> List[str]:
    """Pass a list as a comma-delimted string.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: Splits, if any.
    """
    return text.split(',') if text else []


def bytes_to_int(bytes_str: Union[int, str]) -> int:
    """Convert human readable byte format to an integer.

    Args:
        bytes_str (Union[int, str]): Value to convert.

    Raises:
        ValueError: Invalid byte suffix.

    Returns:
        int: Integer value of bytes.
    """
    #input is already an int
    if isinstance(bytes_str, int) or isinstance(bytes_str, float):
        return int(bytes_str)

    units = {
        'kb': 1024,
        'mb': 1024**2,
        'gb': 1024**3,
        'tb': 1024**4,
        'pb': 1024**5,
        'eb': 1024**6,
        'zb': 1024**7,
        'yb': 1024**8,
    }
    # Convert a various byte types to an integer
    for suffix in units:
        bytes_str = bytes_str.lower().strip()
        if bytes_str.lower().endswith(suffix):
            try:
                return int(float(bytes_str[0:-len(suffix)]) * units[suffix])
            except ValueError:
                raise ValueError(''.join([
                    f'Unsupported value/suffix {bytes_str}. Supported suffix are ',
                    f'{["b"] + list(units.keys())}.'
                ]))
    else:
        # Convert bytes to an integer
        if bytes_str.endswith('b') and bytes_str[0:-1].isdigit():
            return int(bytes_str[0:-1])
        # Convert string representation of a number to an integer
        elif bytes_str.isdigit():
            return int(bytes_str)
        else:
            raise ValueError(''.join([
                f'Unsupported value/suffix {bytes_str}. Supported suffix are ',
                f'{["b"] + list(units.keys())}.'
            ]))


def number_abbrev_to_int(abbrev_str: Union[int, str]) -> int:
    """Convert human readable number abbreviations to an integer.

    Args:
        abbrev_str (Union[int, str]): Value to convert.

    Raises:
        ValueError: Invalid number suffix.

    Returns:
        int: Integer value of number abbreviation.
    """
    #input is already an int
    if isinstance(abbrev_str, int) or isinstance(abbrev_str, float):
        return int(abbrev_str)

    units = {
        'k': 10**3,
        'm': 10**6,
        'b': 10**9,
        't': 10**12,
    }
    # Convert a various abbreviation types to an integer
    for suffix in units:
        abbrev_str = abbrev_str.lower().strip()
        if abbrev_str.lower().endswith(suffix):
            try:
                return int(float(abbrev_str[0:-len(suffix)]) * units[suffix])
            except ValueError:
                raise ValueError(''.join([
                    f'Unsupported value/suffix {abbrev_str}. Supported suffix are ',
                    f'{list(units.keys())}.'
                ]))
    else:
        # Convert string representation of a number to an integer
        if abbrev_str.isdigit():
            return int(abbrev_str)
        else:
            raise ValueError(''.join([
                f'Unsupported value/suffix {abbrev_str}. Supported suffix are ',
                f'{list(units.keys())}.'
            ]))
