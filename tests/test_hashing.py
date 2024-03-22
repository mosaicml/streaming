# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, List, Optional

import pytest

import streaming.base.hashing as shash
from streaming.base import StreamingDataset
from tests.common.utils import convert_to_mds

logger = logging.getLogger(__name__)


def test_get_hashes():
    supported_hashes = [
        'xxh32', 'sha384', 'blake2s', 'md5', 'sha3_384', 'xxh128', 'sha1', 'blake2b', 'xxh3_128',
        'sha256', 'sha224', 'sha512', 'sha3_512', 'xxh3_64', 'xxh64', 'sha3_224', 'sha3_256'
    ]
    hashes = shash.get_hashes()
    assert isinstance(hashes, set)
    assert len(hashes) == len(supported_hashes)
    assert hashes == set(supported_hashes)


@pytest.mark.parametrize(('algo_name', 'is_supported'), [('sha384', True), ('xxh128', True),
                                                         ('', False), ('fake', False)])
def test_is_hash(algo_name: str, is_supported: bool):
    output = shash.is_hash(algo_name)
    assert output is is_supported


@pytest.mark.parametrize(('algo_name', 'data', 'expected_output'), [
    ('md5', b'hello', '5d41402abc4b2a76b9719d911017c592'),
    ('sha3_256', b'hello', '3338be694f50c5f338814986cdf0686453a888b84f424d792af4b9202398f392'),
    ('xxh3_64', b'hello', '9555e8555c62dcfd'),
])
def test_get_hash(algo_name: str, data: bytes, expected_output: str):
    output = shash.get_hash(algo_name, data)
    assert isinstance(output, str)
    assert output == expected_output


@pytest.mark.parametrize(('algo_name', 'data'), [
    ('', b'hello'),
    ('sha3', b'hello'),
])
def test_get_hash_invalid_algo(algo_name: str, data: bytes):
    with pytest.raises(ValueError):
        _ = shash.get_hash(algo_name, data)


@pytest.mark.parametrize('hashes', [None, ['sha1'], ['sha384', 'xxh128']])
@pytest.mark.parametrize('compression', [None, 'zstd:7'])
@pytest.mark.usefixtures('local_remote_dir')
def test_validate_checksum(local_remote_dir: Any, hashes: Optional[List[str]], compression: str):
    num_samples = 117
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=num_samples,
                   compression=compression,
                   hashes=hashes,
                   size_limit=1 << 8)

    # Build StreamingDataset
    hash = hashes[0] if hashes is not None else None
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               validate_hash=hash,
                               batch_size=1)

    # Append sample ID
    sample_order = []
    for sample in dataset:
        sample_order.append(sample['id'])

    # Test length
    assert len(
        dataset
    ) == num_samples, f'Got dataset length={len(dataset)} samples, expected {num_samples}'
