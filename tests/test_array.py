# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import pytest
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


def check(np_arr: NDArray[np.int64], my_arr: Range, i: Union[int, slice, List[int],
          NDArray[np.int64]]):
    try:
        t = np_arr[i]
    except:
        t = None
    if isinstance(t, np.ndarray):
        t = t.tolist()

    try:
        m = my_arr[i]
    except:
        m = None

    assert t == m


@pytest.fixture
def np_arange():
    return np.arange(100)


@pytest.fixture
def my_arange():
    return Range(100)


def test_int_from_front(np_arange, my_arange):
    for i in range(100):
        check(np_arange, my_arange, i)


def test_int_from_back(np_arange, my_arange):
    for i in range(-100, 0):
        check(np_arange, my_arange, i)


def test_int_out_of_range(np_arange, my_arange):
    for i in range(-100 * 4, 100 * 4, 10):
        check(np_arange, my_arange, i)


@pytest.mark.parametrize('start', [-142, -100, -99, -42, -1, 0, 42, 99, 100, 142])
@pytest.mark.parametrize('stop', [-142, -100, -99, -42, -1, 0, 42, 99, 100, 142])
@pytest.mark.parametrize('step', [-3, -1, 1, 3])
def test_slice(start, stop, step, np_arange, my_arange):
    i = slice(start, stop, step)
    check(np_arange, my_arange, i)


@pytest.fixture
def slices():
    return [
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
    ]


def test_specific_slice(np_arange, my_arange, slices):
    for i in slices:
        check(np_arange, my_arange, i)


def test_list(np_arange, my_arange, slices):
    for i in slices:
        i = my_arange._each_slice_index(i)
        i = list(i)
        check(np_arange, my_arange, i)


def test_array(np_arange, my_arange, slices):
    for i in slices:
        i = my_arange._each_slice_index(i)
        i = list(i)
        i = np.array(i, dtype=np.int64)
        check(np_arange, my_arange, i)


def test_array_2d(np_arange, my_arange):
    i = np.arange(4)
    i = i * 3 + 7
    i = i.reshape(2, 2)
    check(np_arange, my_arange, i)


def test_array_3d(np_arange, my_arange):
    i = np.arange(8)
    i = i * 3 + 7
    i = i.reshape(2, 2, 2)
    check(np_arange, my_arange, i)


def test_list_2d(np_arange, my_arange):
    i = np.arange(4)
    i = i * 3 + 7
    i = i.reshape(2, 2)
    assert my_arange[i.tolist()] == np_arange[i].tolist()


def test_list_3d(np_arange, my_arange):
    i = np.arange(8)
    i = i * 3 + 7
    i = i.reshape(2, 2, 2)
    assert my_arange[i.tolist()] == np_arange[i].tolist()
