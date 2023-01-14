# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import pytest

import streaming.base.hashing as shash


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
