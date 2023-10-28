# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Conversions between human-friendly string forms and int/float."""

from collections import defaultdict
from typing import Dict, List, Union

__all__ = [
    'get_list_arg', 'normalize_dec_bytes', 'normalize_bin_bytes', 'normalize_bytes',
    'normalize_count', 'normalize_duration'
]


def get_list_arg(text: str, sep: str = ',') -> List[str]:
    """Pass a list as a comma-delimited string.

    Args:
        text (str): Text to parse.

    Returns:
        List[str]: List of items.
    """
    if not text:
        return []

    return text.split(sep)


def parse_str2str(text: str, sep: str = ',', eq: str = '=') -> Dict[str, str]:
    """Pass a dict as a comma- and equals-delimited string.

    Args:
        text (str): Text to parse.
        sep (str): Separator text. Defaults to ``,``.
        eq (str): Assignment text. Deffaults to ``=``.

    Returns:
        Dict[str, str]: Mapping of str to str.
    """
    if not text:
        return {}

    ret = {}
    parts = text.split(sep)
    for part in parts:
        key, val = part.split(eq)
        if key in ret:
            raise ValueError(f'Repeated key: {key} (text: {text}).')
        ret[key] = val
    return ret


def _normalize_arg(text: str, units: Dict[str, int], to_type: type) -> Union[int, float]:
    """Normalize a human-friendly unit string to number.

    Args:
        text (str): Human-friendly string.
        units (Dict[str, Any]): Mapping of unit name to value.
        to_type (Union[int, float]): The return type.

    Returns:
        type: Computer-friendly number.
    """
    # Must be non-empty.
    if not text:
        raise ValueError(f'Text is empty.')

    # Drop commas and underscores (useful to demarcate thousands '1,337' or '1_337').
    text = text.replace(',', '')
    text = text.replace('_', '')

    # Must start with a digit.
    char = text[0]
    if not char.isdigit():
        raise ValueError(f'Text must start with a digit, but got {text[0]} instead (input: ' +
                         f'{text}).')

    # Must alternative between numbers and units, starting with a number.
    in_num = True
    part = []
    parts = []
    for char in text:
        is_digit = char.isdigit() or char == '.'
        if in_num:
            if is_digit:
                part.append(char)
            else:
                part = ''.join(part)
                parts.append(part)
                part = [char]
                in_num = False
        else:
            if is_digit:
                part = ''.join(part)
                parts.append(part)
                part = [char]
                in_num = True
            else:
                part.append(char)
    part = ''.join(part)
    parts.append(part)

    # If just a number, that's it.
    if len(parts) == 1:
        part, = parts
        try:
            return to_type(part)
        except:
            raise ValueError(f'Simple text must be numeric, but got {part} instead (input: ' +
                             f'{text}).')

    # Pair up numbers and units.
    if len(parts) % 2:
        if '' in units:
            # Special case where the implied unit is the empty string, i.e. the smallest unit.
            parts.append('')
        else:
            # If not just a number, each number must be paired with a corresponding unit.
            raise ValueError(f'Text must contain pairs of number and unit, but got an odd ' +
                             f'number of parts instead: {parts} (input: {text}).')

    # Assign parts as numbers and units.
    part_nums = []
    part_units = []
    for i, part in enumerate(parts):
        if i % 2:
            part_units.append(part)
        else:
            part_nums.append(part)

    # Each number before the last one must be integral
    for i, num in enumerate(part_nums[:-0]):
        try:
            part_nums[i] = int(num)
        except:
            raise ValueError(f'Non-final numbers must be integral, but got part {i} as {num} ' +
                             f'instead (input: {text}).')

    # The last number may be fractional.
    try:
        part_nums[-1] = to_type(part_nums[-1])
    except:
        raise ValueError(f'Final number must be numeric, but got {part_nums[-1]} instead ' +
                         f'(input: {text}.')

    # Each unit must be known to us.
    part_muls = []
    for i, unit in enumerate(part_units):
        mul = units.get(unit)
        if mul is None:
            raise ValueError(f'Unit is unknown: {unit} in part {i} (input: {text}).')
        part_muls.append(mul)

    # Each unit must be used at most once.
    unit2count = defaultdict(int)
    for i, unit in enumerate(part_units):
        unit2count[unit] += 1
    for unit in sorted(unit2count):
        count = unit2count[unit]
        if count != 1:
            raise ValueError(f'Unit is reused: {unit} is used {count} times (input: {text}).')

    # Units must be listed in descending order of size.
    prev_mul = part_muls[0]
    for i in range(1, len(part_muls)):
        mul = part_muls[i]
        if mul < prev_mul:
            prev_mul = mul
        else:
            unit = part_units[i]
            raise ValueError(f'Units are out of order: {unit} in part {i} (input: {text}).')

    # The number of any given part must not exceed the size of the next biggest part's unit.
    #
    # (Otherwise you would just roll its overage into the next biggest part.)
    for i in range(1, len(part_muls)):
        parent_mul = part_muls[i - 1]
        mul = part_muls[i]
        num = part_nums[i]
        if parent_mul < mul * num:
            parent_unit = part_units[i - 1]
            unit = part_units[i]
            raise ValueError(f'The number of any non-initial part must not exceed the ratio of ' +
                             f'the unit of the next biggest part to its own unit (otherwise it ' +
                             f'should have been rolled into the bigger part): part {i} having ' +
                             f'{num} of {unit} ({mul}x) vs parent part {i - 1} in units of ' +
                             f'{parent_unit} ({parent_mul}x) (input: {text}).')

    # Collect parts.
    ret = 0
    for num, mul in zip(part_nums, part_muls):
        ret += num * mul
    return ret


def _normalize_num(arg: Union[int, float, str], units: Dict[str, int],
                   to_type: type) -> Union[int, float]:
    """Normalize from human-friendly argument to number.

    Args:
        arg (Union[int, float, str]): Human-friendly argument.
        units (Dict[str, Any]): Mapping of unit name to value.
        to_type (type): The return type.

    Returns:
        Union[int, float]: Numeric argument.
    """
    if isinstance(arg, (int, float)):
        return to_type(arg)
    else:
        return _normalize_arg(arg, units, to_type)


def _normalize_int(arg: Union[int, str], units: Dict[str, int]) -> int:
    """Normalize from human-friendly argument to int.

    Args:
        arg (Union[int, str]): Human-friendly argument.
        units (Dict[str, int]): Mapping of unit name to value.

    Returns:
        int: Integral argument.
    """
    return _normalize_num(arg, units, int)  # pyright: ignore


def _normalize_float(arg: Union[int, float, str], units: Dict[str, int]) -> int:
    """Normalize from human-friendly argument to float.

    Args:
        arg (Union[int, float, str]): Human-friendly argument.
        units (Dict[str, int]): Mapping of unit name to value.

    Returns:
        float: Floating argument.
    """
    return _normalize_num(arg, units, float)  # pyright: ignore


def _get_units(base: int, names: List[str]) -> Dict[str, int]:
    """Generate units mapping given a base and names of powers of that base.

    Args:
        base (int): Base to exponentiate.
        names (List[str]): Name of each power of base.

    Returns:
        Dic[str, int]: Mapping of unit name to value.
    """
    units = {}
    for i, name in enumerate(names):
        if name in units:
            raise ValueError(f'Reused unit name: {name}.')
        units[name] = base**i
    return units


_dec_bytes_units = _get_units(1000, 'b kb mb tb pb eb zb yb rb qb'.split())


def normalize_dec_bytes(bytes: Union[int, str]) -> int:
    """Normalize from human-friendly base-1000 bytes to int.

    Args:
        bytes (Union[int, str]): Human-friendly base-1000 bytes.

    Returns:
        int: Integral bytes.
    """
    return _normalize_int(bytes, _dec_bytes_units)


_bin_bytes_units = _get_units(1024, 'ib kib mib tib pib eib zib yib rib qib'.split())


def normalize_bin_bytes(bytes: Union[int, str]) -> int:
    """Normalize from human-friendly base-1024 bytes to int.

    Args:
        bytes (Union[int, str]): Human-friendly base-1024 bytes.

    Returns:
        int: Integral bytes.
    """
    return _normalize_int(bytes, _bin_bytes_units)


def normalize_bytes(bytes: Union[int, str]) -> int:
    """Normalize from human-friendly base-1000 or base-1024 bytes to int.

    Args:
        bytes (Union[int, str]): Human-friendly base-1000 or base-1024 bytes.

    Returns:
        int: Integral bytes.
    """
    for norm in [normalize_dec_bytes, normalize_bin_bytes]:
        try:
            return norm(bytes)
        except:
            pass
    raise ValueError('Invalid bytes: {bytes}.')


_count_units = _get_units(1000, ' k m b t'.split(' '))


def normalize_count(count: Union[int, str]) -> int:
    """Normalize from human-friendly count to int.

    Args:
        count (Union[int, str]): Human-friendly count.

    Returns:
        int: Integral count.
    """
    ret = _normalize_int(count, _count_units)
    if ret < 0:
        raise ValueError(f'Counts cannot be negative, but got {ret} (input: {count}).')
    return ret


_duration_units = {
    's': 1,
    'm': 60,
    'h': 60 * 60,
    'd': 24 * 60 * 60,
}


def normalize_duration(duration: Union[int, float, str]) -> float:
    """Normalize from human-friendly duration to float.

    Args:
        duration (Union[int, float, str]): Human-friendly duration.

    Returns:
        float: Float duration.
    """
    return _normalize_float(duration, _duration_units)
