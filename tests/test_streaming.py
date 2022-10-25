# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import pathlib
import shutil
import time
from filecmp import dircmp
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from torch.utils.data import DataLoader

from streaming.base import Dataset, MDSWriter


@pytest.fixture
def remote_local(tmp_path: pathlib.Path) -> Tuple[str, str]:
    remote = tmp_path.joinpath('remote')
    local = tmp_path.joinpath('local')
    return str(remote), str(local)


@pytest.fixture
def compressed_remote_local(tmp_path: pathlib.Path) -> Tuple[str, str, str]:
    compressed = tmp_path.joinpath('compressed')
    remote = tmp_path.joinpath('remote')
    local = tmp_path.joinpath('local')
    return tuple(str(x) for x in [compressed, remote, local])


def get_fake_samples_and_columns_metadata(
        num_samples: int) -> Tuple[List[Dict[str, bytes]], List[str], List[str], List[Any]]:
    samples = [{
        'uid': f'{ix:06}'.encode('utf-8'),
        'data': np.int64(3 * ix).tobytes()
    } for ix in range(num_samples)]
    column_encodings = ['str', 'int']
    column_names = ['uid', 'data']
    column_sizes = [None, 8]
    return samples, column_encodings, column_names, column_sizes


def get_config_in_bytes(format: str,
                        size_limit: int,
                        column_names: List[str],
                        column_encodings: List[str],
                        column_sizes: List[str],
                        compression: Optional[str] = None,
                        hashes: Optional[List[str]] = []):
    config = {
        'version': 2,
        'format': format,
        'compression': compression,
        'hashes': hashes,
        'size_limit': size_limit,
        'column_names': column_names,
        'column_encodings': column_encodings,
        'column_sizes': column_sizes
    }
    return json.dumps(config, sort_keys=True).encode('utf-8')


def write_synthetic_streaming_dataset(
    dirname: str,
    columns: Dict[str, str],
    samples: List[Dict[str, Any]],
    size_limit: int,
    compression: Optional[str] = None,
    hashes: Optional[List[str]] = None,
) -> None:
    with MDSWriter(dirname=dirname,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for sample in samples:
            out.write(sample)


@pytest.mark.parametrize('num_samples', [1000, 10000])
@pytest.mark.parametrize('size_limit', [1 << 8, 1 << 12, 1 << 24])
def test_writer(remote_local: Tuple[str, str], num_samples: int, size_limit: int) -> None:
    dirname, _ = remote_local
    samples, column_encodings, column_names, column_sizes = get_fake_samples_and_columns_metadata(
        num_samples)
    columns = dict(zip(column_names, column_encodings))

    config_data_bytes = get_config_in_bytes('mds', size_limit, column_names, column_encodings,
                                            column_sizes)
    extra_bytes_per_shard = 4 + 4 + len(config_data_bytes)
    extra_bytes_per_sample = 4

    first_sample_body = list(samples[0].values())
    first_sample_head = np.array(
        [len(data) for data, size in zip(first_sample_body, column_sizes) if size is None],
        dtype=np.uint32)
    first_sample_bytes = len(first_sample_head.tobytes() +
                             b''.join(first_sample_body)) + extra_bytes_per_sample

    expected_samples_per_shard = (size_limit - extra_bytes_per_shard) // first_sample_bytes
    expected_num_shards = math.ceil(num_samples / expected_samples_per_shard)
    expected_num_files = expected_num_shards + 1  # the index file and compression metadata file

    write_synthetic_streaming_dataset(dirname=dirname,
                                      columns=columns,
                                      samples=samples,
                                      size_limit=size_limit)
    files = os.listdir(dirname)
    print(f'Number of files: {len(files)}')

    assert len(
        files
    ) == expected_num_files, f'Files written ({len(files)}) != expected ({expected_num_files}).'


@pytest.mark.xfail(
    reason='Fetches shard greedily. See https://mosaicml.atlassian.net/browse/CO-548')
@pytest.mark.parametrize('batch_size', [None, 1, 2])
@pytest.mark.parametrize('remote_arg', ['none', 'same', 'different'])
@pytest.mark.parametrize('shuffle', [False, True])
def test_reader(remote_local: Tuple[str, str], batch_size: int, remote_arg: str, shuffle: bool):
    num_samples = 117
    size_limit = 1 << 8
    samples, column_encodings, column_names, _ = get_fake_samples_and_columns_metadata(num_samples)
    columns = dict(zip(column_names, column_encodings))
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
                                      samples=samples,
                                      size_limit=size_limit)

    # Build Dataset
    dataset = Dataset(local=local, remote=remote, shuffle=shuffle, batch_size=batch_size)

    # Test basic sample order
    rcvd_samples = 0
    shuffle_matches = 0
    for ix, sample in enumerate(dataset):
        rcvd_samples += 1
        uid = sample['uid']
        data = sample['data']
        expected_uid = f'{ix:06}'
        expected_data = 3 * ix
        if shuffle:
            shuffle_matches += (expected_uid == uid)
        else:
            assert uid == expected_uid, f'sample ix={ix} has uid={uid}, expected {expected_uid}'
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
def test_reader_download_fail(remote_local: Tuple[str, str], missing_file: str):
    num_samples = 117
    size_limit = 1 << 8
    samples, column_encodings, column_names, _ = get_fake_samples_and_columns_metadata(num_samples)
    columns = dict(zip(column_names, column_encodings))
    remote, local = remote_local
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=samples,
                                      size_limit=size_limit)

    if missing_file == 'index':
        os.remove(os.path.join(remote, 'index.json'))
    elif missing_file == 'shard':
        os.remove(os.path.join(remote, 'shard.00000.mds'))

    # Build and iterate over a streaming Dataset
    try:
        dataset = Dataset(local=local, remote=remote, shuffle=False, download_timeout=1)
        for _ in dataset:
            pass
    except FileNotFoundError as e:
        print(f'Successfully raised error: {e}')


@pytest.mark.parametrize('created_ago', [0.5, 3])
@pytest.mark.parametrize('download_timeout', [1])
@pytest.mark.parametrize('compression', [None])
def test_reader_after_crash(remote_local: Tuple[str, str], created_ago: float,
                            download_timeout: float, compression: str) -> None:
    compression_ext = f'.{compression.split(":")[0]}' if compression is not None else ''
    num_samples = 117
    size_limit = 1 << 8
    samples, column_encodings, column_names, _ = get_fake_samples_and_columns_metadata(num_samples)
    columns = dict(zip(column_names, column_encodings))
    remote, local = remote_local
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=samples,
                                      size_limit=size_limit,
                                      compression=compression)

    if not os.path.exists(local):
        os.mkdir(local)

    shutil.copy(os.path.join(remote, f'index.json'), os.path.join(local, f'index.json.tmp'))
    shutil.copy(os.path.join(remote, f'shard.00003.mds{compression_ext}'),
                os.path.join(local, f'shard.00003.mds.tmp{compression_ext}'))
    time.sleep(created_ago)

    dataset = Dataset(local=local, remote=remote, shuffle=False, download_timeout=download_timeout)

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
def test_reader_getitem(remote_local: Tuple[str, str], share_remote_local: bool) -> None:
    num_samples = 117
    size_limit = 1 << 8
    samples, column_encodings, column_names, _ = get_fake_samples_and_columns_metadata(num_samples)
    columns = dict(zip(column_names, column_encodings))
    remote, local = remote_local
    if share_remote_local:
        local = remote
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=samples,
                                      size_limit=size_limit)

    # Build a streaming Dataset
    dataset = Dataset(local=local, remote=remote, shuffle=False)

    # Test retrieving random sample
    _ = dataset[17]


@pytest.mark.xfail(
    reason='Fetches shard greedily. See https://mosaicml.atlassian.net/browse/CO-548')
@pytest.mark.parametrize('batch_size', [1, 2, 5])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [1])
@pytest.mark.parametrize('persistent_workers', [
    False,
    True,
])
@pytest.mark.parametrize('shuffle', [False, True])
def test_dataloader_single_device(remote_local: Tuple[str, str], batch_size: int, drop_last: bool,
                                  num_workers: int, persistent_workers: bool, shuffle: bool):
    num_samples = 31
    size_limit = 1 << 6
    samples, column_encodings, column_names, _ = get_fake_samples_and_columns_metadata(num_samples)
    columns = dict(zip(column_names, column_encodings))
    remote, local = remote_local
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=samples,
                                      size_limit=size_limit)

    # Build a streaming Dataset
    dataset = Dataset(local=local, remote=remote, shuffle=shuffle, batch_size=batch_size)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=drop_last,
                            persistent_workers=persistent_workers)

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
            assert len(batch['uid']) == batch_size
        else:
            if drop_last:
                assert len(batch['uid']) == batch_size
            else:
                assert len(batch['uid']) <= batch_size

        for uid in batch['uid']:
            sample_order.append(int(uid))

    # Test dataloader length
    assert len(dataloader) == expected_num_batches
    assert rcvd_batches == expected_num_batches

    # Test that all samples arrived
    assert len(sample_order) == expected_num_samples
    if not drop_last:
        assert len(set(sample_order)) == num_samples

    # Iterate over the dataloader again to check shuffle behavior
    second_sample_order = []
    for batch_ix, batch in enumerate(dataloader):
        for uid in batch['uid']:
            second_sample_order.append(int(uid))

    assert len(sample_order) == len(second_sample_order)
    if shuffle:
        assert sample_order != second_sample_order
    else:
        assert sample_order == second_sample_order


def check_for_diff_files(dir: dircmp, compression_ext: Union[None, str]):
    """Check recursively for different files in a dircmp object.

    Local directory also has the uncompressed files, ignore it during file comparison.
    """
    if compression_ext:
        for file in dir.diff_files:
            assert not file.endswith(compression_ext)
    else:
        assert len(dir.diff_files) == 0
    for subdir in dir.subdirs:
        check_for_diff_files(subdir, compression_ext)


@pytest.mark.parametrize('compression', [None, 'gz', 'gz:5'])
def test_compression(compressed_remote_local: Tuple[str, str, str], compression: Optional[str]):
    num_samples = 31
    size_limit = 1 << 6
    shuffle = True
    compressed, remote, local = compressed_remote_local
    samples, column_encodings, column_names, _ = get_fake_samples_and_columns_metadata(num_samples)
    columns = dict(zip(column_names, column_encodings))
    compression_ext = compression.split(':')[0] if compression else None

    write_synthetic_streaming_dataset(dirname=compressed,
                                      columns=columns,
                                      samples=samples,
                                      size_limit=size_limit,
                                      compression=compression)
    write_synthetic_streaming_dataset(dirname=remote,
                                      columns=columns,
                                      samples=samples,
                                      size_limit=size_limit,
                                      compression=None)

    dataset = Dataset(local=local, remote=compressed, shuffle=shuffle)

    for _ in dataset:
        pass  # download sample

    dcmp = dircmp(remote, local)
    check_for_diff_files(dcmp, compression_ext)
