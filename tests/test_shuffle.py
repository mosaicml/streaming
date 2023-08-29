# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np

from streaming.base.shuffle import (get_shuffle_py1b, get_shuffle_py1br, get_shuffle_py1s,
                                    get_shuffle_py2s)


def check(get_shuffle: Callable) -> None:
    shard_sizes = 1 + np.arange(100)
    dataset_size = sum(shard_sizes)
    block_size = 300
    for num_canonical_nodes in [1, 2, 3]:
        for seed in [0, 1, 2]:
            lists = []
            for epoch in [0, 1, 2]:
                ids = get_shuffle(shard_sizes, num_canonical_nodes, seed, epoch, block_size)
                assert sorted(ids) == list(range(len(ids)))
                parts = []
                for i in range(num_canonical_nodes):
                    begin = dataset_size * i // num_canonical_nodes
                    end = dataset_size * (i + 1) // num_canonical_nodes
                    part = ids[begin:end]
                    parts.append(sorted(part))
                lists.append(parts)
            lists = list(zip(*lists))
            for parts in lists:
                for i in range(1, len(parts)):
                    assert parts[0] == parts[i]


def test_shuffle_py1b():
    check(get_shuffle_py1b)


def test_shuffle_py1br():
    check(get_shuffle_py1br)


def test_shuffle_py1s():
    check(get_shuffle_py1s)


def test_shuffle_py2s():
    check(get_shuffle_py2s)
