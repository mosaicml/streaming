# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from shutil import rmtree
from typing import Tuple

import pytest

from streaming import MDSWriter, StreamingDataset


def one(remote: str, local: str):
    """
    With shard eviction disabled.
    """
    dataset = StreamingDataset(remote=remote, local=local)
    for _ in range(3):
        for sample in dataset:  # pyright: ignore
            pass
    assert os.listdir(remote) == os.listdir(local)
    rmtree(local)


def two(remote: str, local: str):
    """
    With no shard evictions because cache_limit is bigger than the dataset.
    """
    dataset = StreamingDataset(remote=remote, local=local, cache_limit=1_000_000)
    for _ in range(3):
        for sample in dataset:  # pyright: ignore
            pass
    rmtree(local)


def three(remote: str, local: str):
    """
    With shard eviction because cache_limit is smaller than the whole dataset.
    """
    dataset = StreamingDataset(remote=remote, local=local, cache_limit=100_000)
    for _ in range(3):
        for sample in dataset:  # pyright: ignore
            pass
    rmtree(local)


def four(remote: str, local: str):
    """
    Manually downloading and evicting shards.
    """
    dataset = StreamingDataset(remote=remote, local=local)

    for shard_id in range(dataset.num_shards):
        dataset.download_shard(shard_id)

    assert os.listdir(remote) == os.listdir(local)

    for shard_id in range(dataset.num_shards):
        dataset.evict_shard(shard_id)

    assert os.listdir(local) == ['index.json']

    for sample_id in range(dataset.num_samples):
        dataset[sample_id]

    assert os.listdir(remote) == os.listdir(local)


def five(remote: str, local: str):
    """
    Shard eviction with an excess of shards already present.
    """
    dataset = StreamingDataset(remote=remote, local=local, cache_limit=100_000)
    for _ in range(3):
        for sample in dataset:  # pyright: ignore
            pass
    rmtree(local)


funcs = one, two, three, four, five


@pytest.mark.usefixtures('local_remote_dir')
def test_eviction_nozip(local_remote_dir: Tuple[str, str]):
    num_samples = 20_000
    local, remote = local_remote_dir
    columns = {'data': 'bytes'}
    compression = None
    hashes = None
    size_limit = 10_000

    with MDSWriter(out=remote,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for _ in range(num_samples):
            sample = {'data': b'\0' * 10}
            out.write(sample)

    for func in funcs:
        func(remote, local)
    """
    # With impossible shard eviction settings because cache_limit is set too low.
    try:
        dataset = StreamingDataset(remote=remote, local=local, cache_limit=1_000)
        for sample in dataset:
            pass
        assert False
    except RuntimeError:
        pass
    """
