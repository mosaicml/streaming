# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import shutil

from streaming import MDSWriter, StreamingDataset


def test_laziness():
    num_samples = 100_000
    local = 'my-local'
    remote = 'my-remote'
    columns = {'value': 'int'}
    compression = None
    hashes = None
    size_limit = 10_000

    with MDSWriter(remote, columns, compression, hashes, size_limit) as out:
        for i in range(num_samples):
            sample = {'value': i}
            out.write(sample)

    # Verify __getitem__() downloads/accesses.
    dataset = StreamingDataset(remote)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i

    # Verify _get_downloaded_item() accesses.
    dataset = StreamingDataset(remote)
    for i, sample in zip(range(num_samples), dataset):
        assert sample['value'] == i

    # Re-verify _get_downloaded_item() accesses.
    dataset = StreamingDataset(local, remote)
    for i, sample in zip(range(num_samples), dataset):
        assert sample['value'] == i

    shutil.rmtree(local)

    # Verify __getitem__() downloads/accesses.
    dataset = StreamingDataset(local, remote)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i

    # Re-verify __getitem__() downloads/accesses.
    dataset = StreamingDataset(local, remote)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i

    shutil.rmtree(remote)
    shutil.rmtree(local)
