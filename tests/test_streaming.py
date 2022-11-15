# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, Tuple

import pytest
from torch.utils.data import DataLoader

from streaming.base import StreamingDataset
from tests.common.datasets import SequenceDataset, write_synthetic_streaming_dataset


@pytest.mark.parametrize('batch_size', [128])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [0, 1, 8])
@pytest.mark.parametrize('num_samples', [9867, 30_000])
@pytest.mark.parametrize('size_limit', [8_192])
@pytest.mark.usefixtures('remote_local')
def test_dataloader_single_device(remote_local: Tuple[str, str], batch_size: int, drop_last: bool,
                                  num_workers: int, num_samples: int, size_limit: int):
    dataset = SequenceDataset(num_samples)
    columns = dict(zip(dataset.column_names, dataset.column_encodings))
    remote, local = remote_local
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=dataset,
                                      size_limit=size_limit)

    # Build a StreamingDataset
    dataset = StreamingDataset(local=local,
                               remote=remote,
                               shuffle=True,
                               batch_size=batch_size,
                               shuffle_seed=123)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=drop_last)

    # Expected number of batches based on batch_size and drop_last
    expected_num_batches = (num_samples // batch_size) if drop_last else math.ceil(num_samples /
                                                                                   batch_size)
    expected_num_samples = expected_num_batches * batch_size if drop_last else num_samples

    # Iterate over DataLoader
    rcvd_batches = 0
    sample_order = []

    for batch_ix, batch in enumerate(dataloader):
        rcvd_batches += 1

        # Every batch should be complete except (maybe) final one
        if batch_ix + 1 < expected_num_batches:
            assert len(batch['id']) == batch_size
        else:
            if drop_last:
                assert len(batch['id']) == batch_size
            else:
                assert len(batch['id']) <= batch_size

        sample_order.extend(batch['id'][:])

    # Test dataloader length
    assert len(dataloader) == expected_num_batches
    assert rcvd_batches == expected_num_batches

    # Test that all samples arrived with no duplicates
    assert len(set(sample_order)) == expected_num_samples
    if not drop_last:
        assert len(set(sample_order)) == num_samples


@pytest.mark.parametrize('batch_size', [None, 1, 2])
@pytest.mark.parametrize('seed', [987])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('num_workers', [0, 1, 8])
@pytest.mark.usefixtures('mds_dataset_dir')
def test_dataloader_determinism(mds_dataset_dir: Any, batch_size: int, seed: int, shuffle: bool,
                                num_workers: int):
    remote_dir, local_dir = mds_dataset_dir

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    # Append sample ID
    sample_order = []
    for batch in dataloader:
        sample_order.append(batch['id'][:])

    # Build StreamingDataset again to test deterministic sample ID
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    # Append sample ID
    second_sample_order = []
    for batch in dataloader:
        second_sample_order.append(batch['id'][:])

    assert len(sample_order) == len(second_sample_order)
    assert sample_order == second_sample_order
