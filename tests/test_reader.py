# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
from numpy.typing import NDArray

from streaming.base import StreamingDataset
from tests.common.datasets import SequenceDataset, write_mds_dataset
from tests.common.utils import copy_all_files

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

        write_mds_dataset(out_root=remote_dir,
                          columns=columns,
                          samples=dataset,
                          size_limit=size_limit)
        yield remote_dir, local_dir
    finally:
        mock_dir.cleanup()  # pyright: ignore


@pytest.mark.parametrize('batch_size', [None, 1, 2])
@pytest.mark.parametrize('remote_arg', ['none', 'same', 'different'])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('seed', [5151])
@pytest.mark.parametrize('num_canonical_nodes', [1])
@pytest.mark.usefixtures('mds_dataset_dir')
def test_dataset_sample_order(mds_dataset_dir: Any, batch_size: int, remote_arg: str,
                              shuffle: bool, seed: int, num_canonical_nodes: int):
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
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes)

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
            assert data == expected_data, \
                f'sample ix={ix} has data={data}, expected {expected_data}'

    # If shuffling, there should be few matches
    if shuffle:
        assert shuffle_matches < num_samples // 2

    # Test length
    assert rcvd_samples == num_samples, \
        f'Only received {rcvd_samples} samples, expected {num_samples}'
    assert len(
        dataset
    ) == num_samples, f'Got dataset length={len(dataset)} samples, expected {num_samples}'


@pytest.mark.parametrize('batch_size', [None, 1, 2])
@pytest.mark.parametrize('seed', [8988])
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
@pytest.mark.parametrize('seed', [7777])
@pytest.mark.usefixtures('mds_dataset_dir')
def test_reader_download_fail(mds_dataset_dir: Any, missing_file: str, seed: int):
    remote_dir, local_dir = mds_dataset_dir

    if missing_file == 'index':
        os.remove(os.path.join(remote_dir, 'index.json'))

    # Build and iterate over a StreamingDataset
    with pytest.raises(FileNotFoundError) as exc_info:
        dataset = StreamingDataset(local=local_dir,
                                   remote=remote_dir,
                                   shuffle=False,
                                   download_timeout=1,
                                   shuffle_seed=seed)
        for _ in dataset:
            pass
    assert exc_info.match(r'.*No such file or directory*')


@pytest.mark.parametrize('created_ago', [0.5, 1.0])
@pytest.mark.parametrize('download_timeout', [1])
@pytest.mark.parametrize('seed', [2569])
@pytest.mark.usefixtures('mds_dataset_dir')
def test_reader_after_crash(mds_dataset_dir: Any, created_ago: float, download_timeout: float,
                            seed: int):
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
                               download_timeout=download_timeout,
                               shuffle_seed=seed)

    # Iterate over dataset and make sure there are no TimeoutErrors
    for _ in dataset:
        pass


def _validate_sample(index: Union[int, slice, List[int], NDArray[np.int64]],
                     output_sample: Union[Dict, List], total_samples: int):
    """Validate the generated sample with the expected sample."""

    def validate_single_sample(index: int, output_sample: Dict, total_samples: int):
        if index < 0:
            sample_index = total_samples + index
            assert output_sample['id'] == f'{sample_index:06}'
            assert output_sample['sample'] == 3 * sample_index
        else:
            assert output_sample['id'] == f'{index:06}'
            assert output_sample['sample'] == 3 * index

    if isinstance(index, int):
        assert isinstance(output_sample, Dict)
        validate_single_sample(index, output_sample, total_samples)
    elif isinstance(index, List):
        for i, sample_idx in enumerate(index):
            validate_single_sample(sample_idx, output_sample[i], total_samples)
    elif isinstance(index, slice):
        indices = range(index.start, index.stop, index.step)
        for i, sample_idx in enumerate(indices):
            validate_single_sample(sample_idx, output_sample[i], total_samples)
    else:  # NDArray
        for i, sample_idx in enumerate(index):
            validate_single_sample(sample_idx, output_sample[i], total_samples)


@pytest.mark.parametrize(
    'share_remote_local',
    [
        True,
        # False,
    ],
)
@pytest.mark.usefixtures('mds_dataset_dir')
@pytest.mark.parametrize('index', [
    -1, 0, [17], [44, 98], [-14, -87, -55],
    slice(0, 29, 3),
    slice(-27, -99, -5),
    np.arange(10),
    np.array([3, 19, -70, -32])
])
@pytest.mark.parametrize('seed', [5566])
def test_reader_getitem(mds_dataset_dir: Any, share_remote_local: bool,
                        index: Union[int, slice, List[int], NDArray[np.int64]], seed: int):
    remote_dir, local_dir = mds_dataset_dir
    if share_remote_local:
        local_dir = remote_dir

    # Build a StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=False,
                               shuffle_seed=seed)

    # Test retrieving random sample
    sample = dataset[index]
    _validate_sample(index, sample, len(dataset))


@pytest.mark.usefixtures('mds_dataset_dir')
def test_dataset_split_instantiation(mds_dataset_dir: Any):

    splits = ['train', 'val']
    remote_dir, local_dir = mds_dataset_dir

    # Build StreamingDataset for each split
    for split in splits:
        remote_split_dir = os.path.join(remote_dir, split)
        copy_all_files(remote_dir, remote_split_dir)
        _ = StreamingDataset(local=local_dir, remote=remote_dir, split=split)


@pytest.mark.usefixtures('mds_dataset_dir')
def test_invalid_index_json_exception(local_remote_dir: Tuple[str, str]):
    local_dir, _ = local_remote_dir
    filename = 'index.json'
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    # Creates an empty file
    with open(os.path.join(local_dir, filename), 'w') as _:
        pass

    with pytest.raises(json.decoder.JSONDecodeError,
                       match=f'Index file at.*is empty or corrupted'):
        _ = StreamingDataset(local=local_dir)


@pytest.mark.usefixtures('mds_dataset_dir')
def test_empty_shards_index_json_exception(local_remote_dir: Tuple[str, str]):
    local_dir, _ = local_remote_dir
    filename = 'index.json'
    content = {'shards': [], 'version': 2}

    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    # Creates a `index.json` file and write the content to it
    with open(os.path.join(local_dir, filename), 'w') as outfile:
        json.dump(content, outfile)

    with pytest.raises(RuntimeError, match=f'Stream contains no samples: .*'):
        _ = StreamingDataset(local=local_dir)
