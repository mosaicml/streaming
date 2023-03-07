# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from shutil import rmtree
from typing import List

from streaming import MDSWriter, Stream, StreamingDataset


def walk(dataset: StreamingDataset) -> List[int]:
    return [sample['value'] for sample in dataset]


def float_eq(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


def test_mixing():
    root = '/tmp/foo'
    columns = {'value': 'int'}
    for i in range(4):
        subroot = os.path.join(root, str(i))
        with MDSWriter(out=subroot, columns=columns) as out:
            begin = i * 2
            end = (i + 1) * 2
            for value in range(begin, end):
                sample = {'value': value}
                out.write(sample)

    def test_mix_none():
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            stream = Stream(local=subroot)
            streams.append(stream)
        dataset = StreamingDataset(streams=streams)
        assert dataset.num_samples == 8
        assert walk(dataset) == list(range(8))
        for stream in streams:
            assert float_eq(stream.proportion, 0.25)
            assert stream.repeat == 1
            assert stream.samples == 2

    def test_mix_samples_same():
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            stream = Stream(local=subroot, samples=2)
            streams.append(stream)
        dataset = StreamingDataset(streams=streams)
        assert dataset.num_samples == 8
        assert walk(dataset) == list(range(8))
        for stream in streams:
            assert float_eq(stream.proportion, 0.25)
            assert stream.repeat == 1
            assert stream.samples == 2

    def test_mix_samples_mul():
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            stream = Stream(local=subroot, samples=4)
            streams.append(stream)
        dataset = StreamingDataset(streams=streams)
        assert dataset.num_samples == 8
        assert walk(dataset) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
        for stream in streams:
            assert float_eq(stream.proportion, 0.25)
            assert stream.repeat == 2
            assert stream.samples == 4

    def test_mix_samples_range():
        samples = [0, 2, 4, 6]
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            stream = Stream(local=subroot, samples=samples[i])
            streams.append(stream)
        dataset = StreamingDataset(streams=streams)
        assert dataset.num_samples == 8
        assert walk(dataset) == [2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7]
        for i, stream in enumerate(streams):
            assert float_eq(stream.proportion, i / 6)
            assert stream.repeat == i
            assert stream.samples == i * 2

    def test_mix_repeat():
        repeat = [0, 1, 2, 3]
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            stream = Stream(local=subroot, repeat=repeat[i])
            streams.append(stream)
        dataset = StreamingDataset(streams=streams)
        assert dataset.num_samples == 8
        assert walk(dataset) == [2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7]
        for i, stream in enumerate(streams):
            assert float_eq(stream.proportion, i / 6)
            assert stream.repeat == i
            assert stream.samples == i * 2

    def test_mix_repeat_and_samples():
        weights = [
            (0, None),
            (None, 2),
            (2, None),
            (None, 6),
        ]
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            repeat, samples = weights[i]
            stream = Stream(local=subroot, repeat=repeat, samples=samples)
            streams.append(stream)
        dataset = StreamingDataset(streams=streams)
        assert dataset.num_samples == 8
        assert walk(dataset) == [2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7]
        for i, stream in enumerate(streams):
            assert float_eq(stream.proportion, i / 6)
            assert stream.repeat == i
            assert stream.samples == i * 2

    def test_mix_repeat_samples_none():
        weights = [
            (0, None),
            (None, None),
            (2, None),
            (None, 6),
        ]
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            repeat, samples = weights[i]
            stream = Stream(local=subroot, repeat=repeat, samples=samples)
            streams.append(stream)
        dataset = StreamingDataset(streams=streams)
        assert dataset.num_samples == 8
        assert walk(dataset) == [2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7]
        for i, stream in enumerate(streams):
            assert float_eq(stream.proportion, i / 6)
            assert stream.repeat == i
            assert stream.samples == i * 2

    def test_mix_proportion_equal():
        proportion = [1, 1, 1, 1]
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            stream = Stream(local=subroot, proportion=proportion[i])
            streams.append(stream)
        dataset = StreamingDataset(streams=streams)
        assert dataset.num_samples == 8
        assert walk(dataset) == [0, 1, 2, 3, 4, 5, 6, 7]
        for i, stream in enumerate(streams):
            assert float_eq(stream.proportion, 0.25)
            assert stream.repeat == 1
            assert stream.samples == 2

    def test_mix_proportion_range():
        proportion = [0, 1 / 6, 2 / 6, 3 / 6]
        streams = []
        for i in range(4):
            subroot = os.path.join(root, str(i))
            stream = Stream(local=subroot, proportion=proportion[i])
            streams.append(stream)
        dataset = StreamingDataset(streams=streams, samples_per_epoch=12)
        assert dataset.num_samples == 8
        assert walk(dataset) == [2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7]
        for i, stream in enumerate(streams):
            assert float_eq(stream.proportion, i / 6)
            assert stream.repeat == i
            assert stream.samples == i * 2

    test_mix_none()
    test_mix_samples_same()
    test_mix_samples_mul()
    test_mix_samples_range()
    test_mix_repeat()
    test_mix_repeat_and_samples()
    test_mix_repeat_samples_none()
    test_mix_proportion_equal()
    test_mix_proportion_range()

    rmtree(root)
