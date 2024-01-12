# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import numpy as np
import pytest
from numpy.typing import NDArray

from streaming.base.array import Array


class Range(Array):
    """A minimum viable instantiation of an Array."""

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items

    @property
    def size(self) -> int:
        return self.num_items

    def get_item(self, idx: int) -> int:
        assert 0 <= idx < self.num_items
        return idx


def validate(
    np_arr: NDArray[np.int64],
    strmg_arr: Range,
    i: Union[int, slice, List[int], NDArray[np.int64]],
):
    try:
        x = np_arr[i]
    except:
        x = None
    if isinstance(x, np.ndarray):
        x = x.tolist()

    try:
        y = strmg_arr[i]
    except:
        y = None

    print(f'{x=}, {y=}')
    assert x == y


@pytest.fixture
def np_arange():
    return np.arange(100)


@pytest.fixture
def strmg_arange():
    return Range(100)


def test_int_from_front(np_arange: NDArray[np.int64], strmg_arange: Range):
    for i in range(100):
        validate(np_arange, strmg_arange, i)


def test_int_from_back(np_arange: NDArray[np.int64], strmg_arange: Range):
    for i in range(-100, 0):
        validate(np_arange, strmg_arange, i)


def test_int_out_of_range(np_arange: NDArray[np.int64], strmg_arange: Range):
    for i in range(-100 * 4, 100 * 4, 10):
        validate(np_arange, strmg_arange, i)


@pytest.mark.parametrize('start', [-142, -100, -99, -42, -1, 0, 42, 99, 100, 142])
@pytest.mark.parametrize('stop', [-142, -100, -99, -42, -1, 0, 42, 99, 100, 142])
@pytest.mark.parametrize('step', [-3, -1, 1, 3])
def test_slice(start: int, stop: int, step: int, np_arange: NDArray[np.int64],
               strmg_arange: Range):
    i = slice(start, stop, step)
    validate(np_arange, strmg_arange, i)


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


def test_specific_slice(np_arange: NDArray[np.int64], strmg_arange: Range, slices: List[slice]):
    for i in slices:
        validate(np_arange, strmg_arange, i)


def test_list_1d(np_arange: NDArray[np.int64], strmg_arange: Range, slices: List[slice]):
    for i in slices:
        i = strmg_arange._each_slice_index(i)
        i = list(i)
        validate(np_arange, strmg_arange, i)


def test_list_2d(np_arange: NDArray[np.int64], strmg_arange: Range):
    i = np.arange(20)
    i = i * 2 + 7
    i = i.reshape(5, 4)
    assert strmg_arange[i.tolist()] == np_arange[i].tolist()


def test_list_3d(np_arange: NDArray[np.int64], strmg_arange: Range):
    i = np.arange(36)
    i = i * 2 + 7
    i = i.reshape(3, 3, 4)
    assert strmg_arange[i.tolist()] == np_arange[i].tolist()


def test_array_1d(np_arange: NDArray[np.int64], strmg_arange: Range, slices: List[slice]):
    for i in slices:
        i = strmg_arange._each_slice_index(i)
        i = list(i)
        i = np.array(i, dtype=np.int64)
        validate(np_arange, strmg_arange, i)


def test_array_2d(np_arange: NDArray[np.int64], strmg_arange: Range):
    i = np.arange(12)
    i = i * 3 + 7
    i = i.reshape(3, 4)
    validate(np_arange, strmg_arange, i)


def test_array_3d(np_arange: NDArray[np.int64], strmg_arange: Range):
    i = np.arange(18)
    i = i * 3 + 7
    i = i.reshape(3, 2, 3)
    validate(np_arange, strmg_arange, i)
