# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple

import pytest

from streaming import MDSWriter, StreamingDataset


def one(remote: str, local: str):
    """
    Verify __getitem__ accesses.
    """
    dataset = StreamingDataset(local=remote, batch_size=1)
    for i in range(dataset.num_samples):
        sample = dataset[i]
        assert sample['value'] == i


def two(remote: str, local: str):
    """
    Verify __iter__ -> __getitem__ accesses.
    """
    dataset = StreamingDataset(local=remote, num_canonical_nodes=1, batch_size=1)
    for i, sample in zip(range(dataset.num_samples), dataset):
        assert sample['value'] == i


def three(remote: str, local: str):
    """
    Verify __getitem__ downloads/accesses.
    """
    dataset = StreamingDataset(local=local, remote=remote, batch_size=1)
    for i in range(dataset.num_samples):
        sample = dataset[i]
        assert sample['value'] == i


def four(remote: str, local: str):
    """
    Verify __iter__ -> __getitem__ downloads/accesses.
    """
    dataset = StreamingDataset(local=local, remote=remote, num_canonical_nodes=1, batch_size=1)
    for i, sample in zip(range(dataset.num_samples), dataset):
        assert sample['value'] == i
    del dataset


def five(remote: str, local: str):
    """
    Re-verify __getitem__ downloads/accesses.
    """
    dataset = StreamingDataset(local=local, remote=remote, batch_size=1)
    for i in range(dataset.num_samples):
        sample = dataset[i]
        assert sample['value'] == i


@pytest.mark.usefixtures('local_remote_dir')
@pytest.mark.parametrize('func', [one, two, three, four, five])
def test_laziness(local_remote_dir: Tuple[str, str], func: Any):
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

    func(remote, local)
