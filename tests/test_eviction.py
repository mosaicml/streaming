# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from shutil import rmtree
from typing import Tuple

import pytest
from tqdm import tqdm

from streaming import MDSWriter, StreamingDataset


@pytest.mark.usefixtures('local_remote_dir')
def test_eviction_nozip(local_remote_dir: Tuple[str, str]):
    num_samples = 50_000
    local, remote = local_remote_dir
    columns = {'data': 'bytes'}
    compression = None
    hashes = None
    size_limit = 5_000

    with MDSWriter(out=remote,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for _ in range(num_samples):
            sample = {'data': b'\0' * 50}
            out.write(sample)

    # With shard eviction disabled.
    dataset = StreamingDataset(remote=remote, local=local)
    for sample in dataset:
        pass
    del dataset
    assert os.listdir(remote) == os.listdir(local)
    rmtree(local)

    # With no shard evictions because cache_limit is bigger than the dataset.
    dataset = StreamingDataset(remote=remote, local=local, cache_limit=500_000_000)
    for _ in range(3):
        for sample in tqdm(dataset):
            pass
    del dataset
    rmtree(local)
    """
    # With shard evictions.
    dataset = StreamingDataset(remote=remote, local=local, cache_limit=500_000)
    for _ in range(3):
        for sample in tqdm(dataset):
            pass
    del dataset
    rmtree(local)

    # With impossible shard eviction settings because cache_limit is set too low.
    try:
        dataset = StreamingDataset(remote=remote, local=local, cache_limit=1_000)
        for sample in dataset:
            pass
        assert False
    except RuntimeError:
        pass
    """
