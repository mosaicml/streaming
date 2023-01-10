# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import tempfile
import time
from typing import Any

import pytest

from streaming.base import StreamingDataset
from tests.common.datasets import SequenceDataset, write_mds_dataset

logger = logging.getLogger(__name__)


@pytest.fixture(scope='function')
def mds_dataset_dir():
    try:
        mock_dir = tempfile.TemporaryDirectory()
        remote_dir = os.path.join(mock_dir.name, 'remote')
        local_dir = os.path.join(mock_dir.name, 'local')
        num_samples = 117
        size_limit = 1 << 8
        dataset = SequenceDataset(num_samples)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))

        write_mds_dataset(dirname=remote_dir,
                          columns=columns,
                          samples=dataset,
                          size_limit=size_limit)
        yield remote_dir, local_dir
    finally:
        mock_dir.cleanup()  # pyright: ignore


@pytest.mark.parametrize('batch_size', [None, 1, 2])
@pytest.mark.parametrize('remote_arg', ['none', 'same', 'different'])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.usefixtures('mds_dataset_dir')
def test_dataset_sample_order(mds_dataset_dir: Any, batch_size: int, remote_arg: str,
                              shuffle: bool):
    num_samples = 117
    remote_dir, local_dir = mds_dataset_dir
    if remote_arg == 'none':
        local_dir = remote_dir
        remote_dir = None
    elif remote_arg == 'same':
        local_dir = remote_dir
    elif remote_arg == 'different':
        pass
    else:
        assert False, f'Unknown value of remote_arg: {remote_arg}'

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size)

    # Test basic sample order
    rcvd_samples = 0
    shuffle_matches = 0
    for ix, sample in enumerate(dataset):
        rcvd_samples += 1
        id = sample['id']
        data = sample['sample']
        expected_id = f'{ix:06}'
        expected_data = 3 * ix
        if shuffle:
            shuffle_matches += (expected_id == id)
        else:
            assert id == expected_id, f'sample ix={ix} has id={id}, expected {expected_id}'
            assert data == expected_data, f'sample ix={ix} has data={data}, expected {expected_data}'

    # If shuffling, there should be few matches
    if shuffle:
        assert shuffle_matches < num_samples // 2

    # Test length
    assert rcvd_samples == num_samples, f'Only received {rcvd_samples} samples, expected {num_samples}'
    assert len(
        dataset
    ) == num_samples, f'Got dataset length={len(dataset)} samples, expected {num_samples}'


@pytest.mark.parametrize('batch_size', [None, 1, 2])
@pytest.mark.parametrize('seed', [987])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.usefixtures('mds_dataset_dir')
def test_dataset_determinism(mds_dataset_dir: Any, batch_size: int, seed: int, shuffle: bool):
    remote_dir, local_dir = mds_dataset_dir

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Append sample ID
    sample_order = []
    for sample in dataset:
        sample_order.append(sample['id'])

    del dataset

    # Build StreamingDataset again to test deterministic sample ID
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Append sample ID
    second_sample_order = []
    for sample in dataset:
        second_sample_order.append(sample['id'])

    assert len(sample_order) == len(second_sample_order)
    assert sample_order == second_sample_order


@pytest.mark.parametrize(
    'missing_file',
    ['index'],
)
@pytest.mark.usefixtures('mds_dataset_dir')
def test_reader_download_fail(mds_dataset_dir: Any, missing_file: str):
    remote_dir, local_dir = mds_dataset_dir

    if missing_file == 'index':
        os.remove(os.path.join(remote_dir, 'index.json'))

    # Build and iterate over a StreamingDataset
    with pytest.raises(FileNotFoundError) as exc_info:
        dataset = StreamingDataset(local=local_dir,
                                   remote=remote_dir,
                                   shuffle=False,
                                   download_timeout=1)
        for _ in dataset:
            pass
    assert exc_info.match(r'.*No such file or directory*')


@pytest.mark.parametrize('created_ago', [0.5, 1.0])
@pytest.mark.parametrize('download_timeout', [1])
@pytest.mark.usefixtures('mds_dataset_dir')
def test_reader_after_crash(mds_dataset_dir: Any, created_ago: float, download_timeout: float):
    remote_dir, local_dir = mds_dataset_dir

    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    shutil.copy(os.path.join(remote_dir, f'index.json'),
                os.path.join(local_dir, f'index.json.tmp'))
    shutil.copy(os.path.join(remote_dir, f'shard.00003.mds'),
                os.path.join(local_dir, f'shard.00003.mds.tmp'))
    time.sleep(created_ago)

    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=False,
                               download_timeout=download_timeout)

    # Iterate over dataset and make sure there are no TimeoutErrors
    for _ in dataset:
        pass


@pytest.mark.parametrize(
    'share_remote_local',
    [
        True,
        # False,
    ],
)
@pytest.mark.usefixtures('mds_dataset_dir')
@pytest.mark.parametrize('index', [17])
def test_reader_getitem(mds_dataset_dir: Any, share_remote_local: bool, index: int):
    remote_dir, local_dir = mds_dataset_dir
    if share_remote_local:
        local_dir = remote_dir

    # Build a StreamingDataset
    dataset = StreamingDataset(local=local_dir, remote=remote_dir, shuffle=False)

    # Test retrieving random sample
    sample = dataset[index]
    assert sample['id'] == f'{index:06}'
    assert sample['sample'] == 3 * index
