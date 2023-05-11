# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from shutil import rmtree
from typing import Tuple

import pytest

from streaming import MDSWriter, StreamingDataset


def one(remote: str, local: str, keep_zip: bool):
    """
    With shard eviction disabled.
    """
    dataset = StreamingDataset(remote=remote, local=local, keep_zip=keep_zip)
    for _ in range(3):
        for sample in dataset:  # pyright: ignore
            pass

    if keep_zip:
        if dataset.shards[0].compression:
            # Local has raw + zip, remote has zip.
            assert set(
                filter(lambda f: f == 'index.json' or f.endswith('.zstd'),
                       os.listdir(local))) == set(os.listdir(remote))
        else:
            # Local has raw + zip, remote has raw.
            assert set(filter(lambda f: not f.endswith('.zstd'), os.listdir(local))) == \
                set(os.listdir(remote))
    else:
        if dataset.shards[0].compression:
            # Local has raw, remote has zip.
            assert set(os.listdir(local)) == set(
                map(lambda f: f.replace('.zstd', ''), os.listdir(remote)))
        else:
            # Local has raw, remote has raw.
            assert set(os.listdir(local)) == set(os.listdir(remote))

    rmtree(local)


def two(remote: str, local: str, keep_zip: bool):
    """
    With no shard evictions because cache_limit is bigger than the dataset.
    """
    dataset = StreamingDataset(remote=remote,
                               local=local,
                               keep_zip=keep_zip,
                               cache_limit=1_000_000)
    for _ in range(3):
        for sample in dataset:  # pyright: ignore
            pass
    rmtree(local)


def three(remote: str, local: str, keep_zip: bool):
    """
    With shard eviction because cache_limit is smaller than the whole dataset.
    """
    dataset = StreamingDataset(remote=remote, local=local, keep_zip=keep_zip, cache_limit=50_000)
    for _ in range(3):
        for sample in dataset:  # pyright: ignore
            pass
    rmtree(local)


def four(remote: str, local: str, keep_zip: bool):
    """
    Manually downloading and evicting shards.
    """
    dataset = StreamingDataset(remote=remote, local=local, keep_zip=keep_zip)

    for shard_id in range(dataset.num_shards):
        dataset.download_shard(shard_id)

    full = set(os.listdir(local))

    for shard_id in range(dataset.num_shards):
        dataset.evict_shard(shard_id)

    assert os.listdir(local) == ['index.json']

    for sample_id in range(dataset.num_samples):
        dataset[sample_id]

    assert set(os.listdir(local)) == full


def five(remote: str, local: str, keep_zip: bool):
    """
    Shard eviction with an excess of shards already present.
    """
    dataset = StreamingDataset(remote=remote, local=local, keep_zip=keep_zip, cache_limit=50_000)
    for _ in range(3):
        for sample in dataset:  # pyright: ignore
            pass
    rmtree(local)


def six(remote: str, local: str, keep_zip: bool):
    """
    With impossible shard eviction settings because cache_limit is set too low.
    """
    with pytest.raises(ValueError):
        dataset = StreamingDataset(remote=remote, local=local, cache_limit=1_000)
        for _ in dataset:
            pass


funcs = one, two, three, four, five


@pytest.mark.usefixtures('local_remote_dir')
def test_eviction_nozip(local_remote_dir: Tuple[str, str]):
    num_samples = 5_000
    local, remote = local_remote_dir
    columns = {'data': 'bytes'}
    compression = None
    hashes = None
    size_limit = 500

    with MDSWriter(out=remote,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for _ in range(num_samples):
            sample = {'data': b'\0'}}
            out.write(sample)

    for func in funcs:
        func(remote, local, False)


@pytest.mark.usefixtures('local_remote_dir')
def test_eviction_zip_nokeep(local_remote_dir: Tuple[str, str]):
    num_samples = 5_000
    local, remote = local_remote_dir
    columns = {'data': 'bytes'}
    compression = 'zstd'
    hashes = None
    size_limit = 500

    with MDSWriter(out=remote,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for _ in range(num_samples):
            sample = {'data': b'\0'}
            out.write(sample)

    for func in funcs:
        func(remote, local, False)
    assert True


@pytest.mark.usefixtures('local_remote_dir')
def test_eviction_zip_keep(local_remote_dir: Tuple[str, str]):
    num_samples = 5_000
    local, remote = local_remote_dir
    columns = {'data': 'bytes'}
    compression = 'zstd'
    hashes = None
    size_limit = 500

    with MDSWriter(out=remote,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for _ in range(num_samples):
            sample = {'data': b'\0'}
            out.write(sample)

    for func in funcs:
        func(remote, local, True)
