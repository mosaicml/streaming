# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import operator
import os
from shutil import rmtree
from typing import Any, Tuple

import pytest
from torch.utils.data import DataLoader

from streaming import MDSWriter, StreamingDataset
from tests.common.utils import convert_to_mds


def validate(remote: str, local: str, dataset: StreamingDataset, keep_zip: bool,
             is_shard_evicted: bool):
    """Validate the number of files in a local directory in comparison to remote directory."""
    if is_shard_evicted:
        ops = operator.lt
    else:
        ops = operator.eq
    if keep_zip:
        if dataset.shards[0].compression:
            # Local has raw + zip, remote has zip.
            assert ops(
                set(filter(lambda f: f == 'index.json' or f.endswith('.zstd'), os.listdir(local))),
                set(os.listdir(remote)))
        else:
            # Local has raw + zip, remote has raw.
            assert ops(set(filter(lambda f: not f.endswith('.zstd'), os.listdir(local))),  \
                set(os.listdir(remote)))
    else:
        if dataset.shards[0].compression:
            # Local has raw, remote has zip.
            assert ops(set(os.listdir(local)),
                       {f.replace('.zstd', '') for f in os.listdir(remote)})
        else:
            # Local has raw, remote has raw.
            assert ops(set(os.listdir(local)), set(os.listdir(remote)))


def shard_eviction_disabled(remote: str, local: str, keep_zip: bool):
    """
    With shard eviction disabled.
    """
    dataset = StreamingDataset(remote=remote, local=local, keep_zip=keep_zip, batch_size=1)
    for _ in range(2):
        for sample in dataset:  # pyright: ignore
            pass

    validate(remote, local, dataset, keep_zip, False)
    rmtree(local, ignore_errors=False)


def shard_eviction_too_high(remote: str, local: str, keep_zip: bool):
    """
    With no shard evictions because cache_limit is bigger than the dataset.
    """
    dataset = StreamingDataset(remote=remote,
                               local=local,
                               keep_zip=keep_zip,
                               cache_limit=1_000_000,
                               batch_size=1)
    dataloader = DataLoader(dataset=dataset, num_workers=8)
    for _ in range(2):
        for _ in dataloader:
            pass
    validate(remote, local, dataset, keep_zip, False)
    rmtree(local, ignore_errors=False)


def shard_eviction(remote: str, local: str, keep_zip: bool):
    """
    With shard eviction because cache_limit is smaller than the whole dataset.
    """
    cache_limit = '120kb' if keep_zip else '100kb'
    dataset = StreamingDataset(remote=remote,
                               local=local,
                               keep_zip=keep_zip,
                               cache_limit=cache_limit,
                               batch_size=1)
    dataloader = DataLoader(dataset=dataset, num_workers=8)
    for _ in range(2):
        for _ in dataloader:
            pass
    validate(remote, local, dataset, keep_zip, True)
    rmtree(local, ignore_errors=False)


def manual_shard_eviction(remote: str, local: str, keep_zip: bool):
    """
    Manually downloading and evicting shards.
    """
    dataset = StreamingDataset(remote=remote, local=local, keep_zip=keep_zip, batch_size=1)

    for shard_id in range(dataset.num_shards):
        dataset.prepare_shard(shard_id)

    full = set(os.listdir(local))

    for shard_id in range(dataset.num_shards):
        dataset.evict_shard(shard_id)

    assert os.listdir(local) == ['index.json']

    for sample_id in range(dataset.num_samples):
        dataset[sample_id]

    assert set(os.listdir(local)) == full
    rmtree(local, ignore_errors=False)


def cache_limit_too_low(remote: str, local: str, keep_zip: bool):
    """
    With impossible shard eviction settings because cache_limit is set too low.
    """
    with pytest.raises(ValueError):
        dataset = StreamingDataset(remote=remote, local=local, cache_limit='1kb', batch_size=1)
        for _ in dataset:
            pass
    rmtree(local, ignore_errors=False)


funcs = [
    shard_eviction_disabled, shard_eviction_too_high, shard_eviction, manual_shard_eviction,
    cache_limit_too_low
]


@pytest.mark.usefixtures('local_remote_dir')
@pytest.mark.parametrize('func', list(funcs))
def test_eviction_nozip(local_remote_dir: Tuple[str, str], func: Any):
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
            sample = {'data': b'\0'}
            out.write(sample)

    func(remote, local, False)


@pytest.mark.usefixtures('local_remote_dir')
@pytest.mark.parametrize('func', list(funcs))
def test_eviction_zip_nokeep(local_remote_dir: Tuple[str, str], func: Any):
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

    func(remote, local, False)


@pytest.mark.usefixtures('local_remote_dir')
@pytest.mark.parametrize('func', list(funcs))
def test_eviction_zip_keep(local_remote_dir: Tuple[str, str], func: Any):
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

    func(remote, local, True)


@pytest.mark.parametrize('cache_limit', ['2048'])
@pytest.mark.usefixtures('local_remote_dir')
def test_cache_limit_lower_than_index_json(local_remote_dir: Any, cache_limit: str):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=1000,
                   compression=None,
                   size_limit=2048)

    with pytest.raises(ValueError, match='Minimum cache usage.*is larger than the cache limit*'):
        _ = StreamingDataset(local=local_dir,
                             remote=remote_dir,
                             shuffle=False,
                             batch_size=4,
                             cache_limit=cache_limit)


@pytest.mark.parametrize('cache_limit', ['5kb'])
@pytest.mark.parametrize('compression', [None, 'zstd:7'])
@pytest.mark.usefixtures('local_remote_dir')
def test_cache_limit_lower_than_few_shards(local_remote_dir: Any, cache_limit: str,
                                           compression: str):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=1000,
                   compression=compression,
                   size_limit=2048)

    with pytest.raises(ValueError,
                       match='Cache limit.*is too low. Increase the `cache_limit` to*'):
        _ = StreamingDataset(local=local_dir,
                             remote=remote_dir,
                             shuffle=False,
                             batch_size=4,
                             cache_limit=cache_limit)
