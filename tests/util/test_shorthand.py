# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import pytest

from streaming.util.shorthand import get_list_arg, normalize_bytes, normalize_count


@pytest.mark.parametrize(('text', 'expected_output'), [('hello,world', ['hello', 'world']),
                                                       ('hello', ['hello']), ('', [])])
def test_get_list_arg(text: str, expected_output: List[Optional[str]]):
    output = get_list_arg(text)
    assert output == expected_output


def test_normalize_bytes():
    input_to_expected = [
        ('1234', 1234),
        ('1b', 1),
        ('50b', 50),
        ('100kib', 102400),
        ('75mb', 75000000),
        ('75mib', 78643200),
        ('1.39gib', 1492501135),
        ('2tib', 2199023255552),
        ('3pib', 3377699720527872),
        ('1.11eib', 1279742870113600143),
        ('1.09zib', 1286844866581978320732),
        ('2.0yib', 2417851639229258349412352),
        ('7yb', 7000000000000000000000000),
        (1234, 1234),
        (1, 1),
        (0.5 * 1024, 512),
        (100 * 1024, 102400),
        (75 * 1024**2, 78643200),
        (75 * 1024 * 1024, 78643200),
        (35.78, 35),
        (325388903.203984, 325388903),
    ]
    for size_pair in input_to_expected:
        output = normalize_bytes(size_pair[0])
        assert output == size_pair[1]


def test_normalize_bytes_except():
    input_data = [
        '',
        '12kbb',
        '27mxb',
        '79kkb',
        '50B',
        ' 100 kb',
        '75MB',
        '75 mb',
        '1.39Gb',
    ]
    for value in input_data:
        with pytest.raises(ValueError):
            _ = normalize_bytes(value)


def test_normalize_count():
    input_to_expected = [
        ('1234', 1234),
        ('1k', 1000),
        ('50k', 50000),
        ('100k', 100000),
        ('75m', 75000000),
        ('1.39b', 1390000000),
        ('2t', 2000000000000),
        (1234, 1234),
        (1, 1),
        (0.5 * 1000, 500),
        (100 * 1000, 100000),
        (75 * 1000**2, 75000000),
        (75 * 1000 * 1000, 75000000),
        (35.78, 35),
        (325388903.203984, 325388903),
    ]
    for size_pair in input_to_expected:
        output = normalize_count(size_pair[0])
        assert output == size_pair[1]


def test_normalize_count_except():
    input_data = [
        '',
        '12kbb',
        '27mxb',
        '79bk',
        '79bb',
        '79 b    m',
        'p 64',
        '64p',
        '50K',
        ' 100 k',
        '75M',
        '75 m',
        '1.39B',
        '3 T',
    ]
    for value in input_data:
        with pytest.raises(ValueError):
            _ = normalize_count(value)
