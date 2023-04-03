# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import shutil
from typing import Tuple

import pytest

from streaming import MDSWriter, StreamingDataset


@pytest.mark.usefixtures('local_remote_dir')
def test_laziness(local_remote_dir: Tuple[str, str]):
    num_samples = 10_000
    local, remote = local_remote_dir
    columns = {'value': 'int'}
    compression = None
    hashes = None
    size_limit = 1_000

    with MDSWriter(out=remote,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for i in range(num_samples):
            sample = {'value': i}
            out.write(sample)

    # Verify __getitem__ accesses.
    dataset = StreamingDataset(local=remote)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i
    del dataset

    # Verify __iter__ -> __getitem__ accesses.
    dataset = StreamingDataset(local=remote)
    for i, sample in zip(range(num_samples), dataset):
        assert sample['value'] == i
    del dataset

    # Verify __getitem__ downloads/accesses.
    dataset = StreamingDataset(local=local, remote=remote)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i
    del dataset

    shutil.rmtree(local)

    # Verify __iter__ -> __getitem__ downloads/accesses.
    dataset = StreamingDataset(local=local, remote=remote)
    for i, sample in zip(range(num_samples), dataset):
        assert sample['value'] == i
    del dataset

    # Re-verify __getitem__ downloads/accesses.
    dataset = StreamingDataset(local=local, remote=remote)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i
