# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np

from streaming.base.shuffle import (get_shuffle_py1b, get_shuffle_py1br, get_shuffle_py1e,
                                    get_shuffle_py1s, get_shuffle_py2s)


def check(get_shuffle: Callable) -> None:
    shard_sizes = 1 + np.arange(100)
    dataset_size = sum(shard_sizes)
    block_size = 300
    for num_canonical_nodes in [1, 2, 3]:
        for seed in [0, 1, 2]:
            # lists is the list of sorted ids seen by every canonical node in every epoch
            # for example:
            # [[epoch0_CN_a, epoch0_CN_b], [epoch1_CN_a, epoch1_CN_b], [epoch2_CN_a, epoch2_CN_b]]]
            lists = []
            for epoch in [0, 1, 2]:
                ids = get_shuffle(shard_sizes, num_canonical_nodes, seed, epoch, block_size)
                assert sorted(ids) == list(range(len(ids)))
                # parts is a list of the sorted ids seen by each canonical node in an epoch
                parts = []
                for i in range(num_canonical_nodes):
                    begin = dataset_size * i // num_canonical_nodes
                    end = dataset_size * (i + 1) // num_canonical_nodes
                    # get the section of ids corresponding to this canonical node
                    part = ids[begin:end]
                    parts.append(sorted(part))
                lists.append(parts)
            # want to make sure the sample ids seen by each canonical node
            # in each epoch is the same
            lists = list(zip(*lists))
            # each element of `lists` is now a tuple containing the lists of samples
            # seen by a canonical node over all the epochs.
            for parts in lists:
                # make sure all other epochs are the same as epoch 0.
                for i in range(1, len(parts)):
                    assert parts[0] == parts[i]


def test_shuffle_py1b():
    check(get_shuffle_py1b)


def test_shuffle_py1br():
    check(get_shuffle_py1br)


def test_shuffle_py1e():
    check(get_shuffle_py1e)


def test_shuffle_py1s():
    check(get_shuffle_py1s)


def test_shuffle_py2s():
    check(get_shuffle_py2s)
