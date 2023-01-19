# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from streaming.base.array import Array


class Range(Array):
    """A minimum viable instantation of an Array."""

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items

    @property
    def size(self) -> int:
        return self.num_items

    def get_item(self, idx: int) -> int:
        assert 0 <= idx < self.num_items
        return idx


size = 100
theirs = np.arange(size)
mine = Range(size)


def check(i: Union[int, slice, List[int], NDArray[np.int64]]):
    try:
        t = theirs[i]
    except:
        t = None
    if isinstance(t, np.ndarray):
        t = t.tolist()

    try:
        m = mine[i]
    except:
        m = None

    assert t == m


def test_int_from_front():
    for i in range(size):
        check(i)


def test_int_from_back():
    for i in range(-size, 0):
        check(i)


def test_int_out_of_range():
    for i in range(-size * 4, size * 4, 10):
        check(i)


def each_slice():
    yield from [
        slice(0),
        slice(0, 0),
        slice(0, 1),
        slice(1, 2, 3),
        slice(2, 3, 1),
        slice(0, 6, 2),
        slice(0, 10),
        slice(10, 10),
        slice(10, 20, 2),
        slice(20, 10, -1),
        slice(1337, 42, -3),
        slice(-3, 3),
        slice(1337, 42, -5),
        slice(1337, 4, -5),
        slice(1337, -4, -5),
        slice(1337, -42, -5),
        slice(1337, -1337, -5),
        slice(1338, 42, -5),
        slice(-1337, 42, 5),
        slice(-1337, 42, -5),
        slice(42, 101, -1),
        slice(42, 100, -1),
        slice(42, 99, -1),
        slice(42, 42, -1),
        slice(42, 1, -1),
        slice(42, 0, -1),
        slice(42, -1, -1),
        slice(42, -99, -1),
        slice(42, -100, -1),
        slice(42, -101, -1),
        slice(42, 101, -3),
        slice(42, 100, -3),
        slice(42, 99, -3),
        slice(42, 1, -3),
        slice(42, 0, -3),
        slice(42, -1, -3),
        slice(42, -99, -3),
        slice(42, -100, -3),
        slice(42, -101, -3),
    ]


def test_slice():
    for i in each_slice():
        check(i)


def test_list():
    for i in each_slice():
        i = mine._each_slice_index(i)
        i = list(i)
        check(i)


def test_array():
    for i in each_slice():
        i = mine._each_slice_index(i)
        i = list(i)
        i = np.array(i, dtype=np.int64)
        check(i)


def test_array_2d():
    i = np.arange(4)
    i = i * 3 + 7
    i = i.reshape(2, 2)
    check(i)


def test_array_3d():
    i = np.arange(8)
    i = i * 3 + 7
    i = i.reshape(2, 2, 2)
    check(i)


def test_list_2d():
    i = np.arange(4)
    i = i * 3 + 7
    i = i.reshape(2, 2)
    assert mine[i.tolist()] == theirs[i].tolist()


def test_list_3d():
    i = np.arange(8)
    i = i * 3 + 7
    i = i.reshape(2, 2, 2)
    assert mine[i.tolist()] == theirs[i].tolist()
