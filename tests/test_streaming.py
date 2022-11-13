# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
import shutil
import time
from typing import Tuple

import pytest
from torch.utils.data import DataLoader

from streaming.base import Dataset
from tests.common.datasets import *
from tests.common.utils import *

logger = logging.getLogger(__name__)


@pytest.mark.xfail(
    reason='Fetches shard greedily. See https://mosaicml.atlassian.net/browse/CO-1042')
@pytest.mark.parametrize('batch_size', [None, 1, 2])
@pytest.mark.parametrize('remote_arg', ['none', 'same', 'different'])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.usefixtures('remote_local')
def test_reader(remote_local: Tuple[str, str], batch_size: int, remote_arg: str, shuffle: bool):
    num_samples = 117
    size_limit = 1 << 8
    dataset = SequenceDataset(num_samples)
    columns = dict(zip(dataset.column_names, dataset.column_encodings))
    if remote_arg == 'none':
        remote, local = remote_local
        dirname = local
        remote = None
    elif remote_arg == 'same':
        remote, local = remote_local
        dirname = local
        remote = local
    elif remote_arg == 'different':
        remote, local = remote_local
        dirname = remote
    else:
        assert False, f'Unknown value of remote_arg: {remote_arg}'

    write_synthetic_streaming_dataset(dirname=dirname,
                                      columns=columns,
                                      samples=dataset,
                                      size_limit=size_limit)

    # Build Dataset
    dataset = Dataset(local=local, remote=remote, shuffle=shuffle, batch_size=batch_size)

    # Test basic sample order
    rcvd_samples = 0
    shuffle_matches = 0
    for ix, sample in enumerate(dataset):
        rcvd_samples += 1
        id = sample['id']
        data = sample['data']
        expected_id = f'{ix:06}'
        expected_data = 3 * ix
        if shuffle:
            shuffle_matches += (expected_id == id)
        else:
            assert id == expected_id, f'sample ix={ix} has id={id}, expected {expected_id}'
            assert data == expected_data, f'sample ix={ix} has data={data}, expected {expected_data}'

    # If shuffling, there should be few matches
    # The probability of k matches in a random permutation is ~1/(e*(k!))
    if shuffle:
        assert shuffle_matches < 10

    # Test length
    assert rcvd_samples == num_samples, f'Only received {rcvd_samples} samples, expected {num_samples}'
    assert len(
        dataset
    ) == num_samples, f'Got dataset length={len(dataset)} samples, expected {num_samples}'


@pytest.mark.parametrize(
    'missing_file',
    [
        'index',
        'shard',
    ],
)
@pytest.mark.usefixtures('remote_local')
def test_reader_download_fail(remote_local: Tuple[str, str], missing_file: str):
    num_samples = 117
    size_limit = 1 << 8
    dataset = SequenceDataset(num_samples)
    columns = dict(zip(dataset.column_names, dataset.column_encodings))
    remote, local = remote_local
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=dataset,
                                      size_limit=size_limit)

    if missing_file == 'index':
        os.remove(os.path.join(remote, 'index.json'))
    elif missing_file == 'shard':
        os.remove(os.path.join(remote, 'shard.00000.mds'))

    # Build and iterate over a streaming Dataset
    try:
        dataset = Dataset(local=local, remote=remote, shuffle=False, timeout=1)
        for _ in dataset:
            pass
    except FileNotFoundError as e:
        logger.debug(f'Successfully raised error: {e}')


@pytest.mark.parametrize('created_ago', [0.5, 3])
@pytest.mark.parametrize('timeout', [1])
@pytest.mark.parametrize('compression', [None])
@pytest.mark.usefixtures('remote_local')
def test_reader_after_crash(remote_local: Tuple[str, str], created_ago: float, timeout: float,
                            compression: str) -> None:
    compression_ext = f'.{compression.split(":")[0]}' if compression is not None else ''
    num_samples = 117
    size_limit = 1 << 8
    dataset = SequenceDataset(num_samples)
    columns = dict(zip(dataset.column_names, dataset.column_encodings))
    remote, local = remote_local
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=dataset,
                                      size_limit=size_limit,
                                      compression=compression)

    if not os.path.exists(local):
        os.mkdir(local)

    shutil.copy(os.path.join(remote, f'index.json'), os.path.join(local, f'index.json.tmp'))
    shutil.copy(os.path.join(remote, f'shard.00003.mds{compression_ext}'),
                os.path.join(local, f'shard.00003.mds.tmp{compression_ext}'))
    time.sleep(created_ago)

    dataset = Dataset(local=local, remote=remote, shuffle=False, timeout=timeout)

    # Iterate over dataset and make sure there are no TimeoutErrors
    for _ in dataset:
        pass


@pytest.mark.parametrize(
    'share_remote_local',
    [
        True,
        False,
    ],
)
@pytest.mark.usefixtures('remote_local')
def test_reader_getitem(remote_local: Tuple[str, str], share_remote_local: bool) -> None:
    num_samples = 117
    size_limit = 1 << 8
    dataset = SequenceDataset(num_samples)
    columns = dict(zip(dataset.column_names, dataset.column_encodings))
    remote, local = remote_local
    if share_remote_local:
        local = remote
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=dataset,
                                      size_limit=size_limit)

    # Build a streaming Dataset
    dataset = Dataset(local=local, remote=remote, shuffle=False)

    # Test retrieving random sample
    _ = dataset[17]


@pytest.mark.parametrize('batch_size', [128])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [1, 8])
@pytest.mark.parametrize('num_samples', [9867, 100_000])
@pytest.mark.parametrize('size_limit', [8_192, 65_536])
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

    # Build a streaming Dataset
    dataset = Dataset(local=local, remote=remote, shuffle=True, batch_size=batch_size)

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
