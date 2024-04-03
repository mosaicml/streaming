# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Tuple

import numpy as np
import pytest

from streaming import MDSWriter, Stream, StreamingDataset


def walk(dataset: StreamingDataset) -> List[int]:
    return [sample['value'] for sample in dataset]


def float_eq(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


@pytest.mark.usefixtures('local_remote_dir')
@pytest.fixture()
def root(local_remote_dir: Tuple[str, str]):
    root, _ = local_remote_dir
    columns = {'value': 'int'}
    for i in range(4):
        subroot = os.path.join(root, str(i))
        with MDSWriter(out=subroot, columns=columns) as out:
            begin = i * 2
            end = (i + 1) * 2
            for value in range(begin, end):
                sample = {'value': value}
                out.write(sample)
    yield root


def test_mix_none(root: str):
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        stream = Stream(local=subroot)
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == list(range(8))
    for stream in streams:
        assert float_eq(stream.proportion, 0.25)
        assert stream.repeat == 1
        assert stream.choose == 2


def test_mix_choose_same(root: str):
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        stream = Stream(local=subroot, choose=2)
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == list(range(8))
    for stream in streams:
        assert float_eq(stream.proportion, 0.25)
        assert stream.repeat == 1
        assert stream.choose == 2


def test_mix_choose_mul(root: str):
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        stream = Stream(local=subroot, choose=4)
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7]
    for stream in streams:
        assert float_eq(stream.proportion, 0.25)
        assert stream.repeat == 2
        assert stream.choose == 4


def test_mix_choose_range(root: str):
    choices = [0, 2, 4, 6]
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        stream = Stream(local=subroot, choose=choices[i])
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == [2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7]
    for i, stream in enumerate(streams):
        assert float_eq(stream.proportion, i / 6)
        assert stream.repeat == i
        assert stream.choose == i * 2


def test_mix_repeat(root: str):
    repeat = [0, 1, 2, 3]
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        stream = Stream(local=subroot, repeat=repeat[i])
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == [2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7]
    for i, stream in enumerate(streams):
        assert float_eq(stream.proportion, i / 6)
        assert stream.repeat == i
        assert stream.choose == i * 2


def test_mix_repeat_and_choose(root: str):
    weights = [
        (0, None),
        (None, 2),
        (2, None),
        (None, 6),
    ]
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        repeat, choose = weights[i]
        stream = Stream(local=subroot, repeat=repeat, choose=choose)
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == [2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7]
    for i, stream in enumerate(streams):
        assert float_eq(stream.proportion, i / 6)
        assert stream.repeat == i
        assert stream.choose == i * 2


def test_mix_repeat_choose_none(root: str):
    weights = [
        (0, None),
        (None, None),
        (2, None),
        (None, 6),
    ]
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        repeat, choose = weights[i]
        stream = Stream(local=subroot, repeat=repeat, choose=choose)
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == [2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7]
    for i, stream in enumerate(streams):
        assert float_eq(stream.proportion, i / 6)
        assert stream.repeat == i
        assert stream.choose == i * 2


def test_mix_proportion_equal(root: str):
    proportion = [1, 1, 1, 1]
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        stream = Stream(local=subroot, proportion=proportion[i])
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == [0, 1, 2, 3, 4, 5, 6, 7]
    for i, stream in enumerate(streams):
        assert float_eq(stream.proportion, 0.25)
        assert stream.repeat == 1
        assert stream.choose == 2


def test_mix_proportion_range(root: str):
    proportion = [0, 1 / 6, 2 / 6, 3 / 6]
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        stream = Stream(local=subroot, proportion=proportion[i])
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, epoch_size=12, num_canonical_nodes=1, batch_size=1)
    assert dataset.num_samples == 8
    assert walk(dataset) == [2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7]
    for i, stream in enumerate(streams):
        assert float_eq(stream.proportion, i / 6)
        assert stream.repeat == i
        assert stream.choose == i * 2


def test_mix_balance(root: str):
    streams = []
    for i in range(4):
        subroot = os.path.join(root, str(i))
        stream = Stream(local=subroot, choose=3)
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, num_canonical_nodes=1, batch_size=1)
    counts = np.zeros(8, np.int64)
    for _ in range(1000):
        for value in walk(dataset):
            counts[value] += 1
    rates = counts / counts.sum() * len(counts)
    for rate in rates:
        assert 0.975 < rate < 1.025
