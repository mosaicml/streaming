# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from filecmp import dircmp
from typing import Any, Optional, Tuple, Union

import numpy as np
import pytest

from streaming.base import StreamingDataset
from streaming.base.compression import (Brotli, Bzip2, Gzip, Snappy, Zstandard, compress,
                                        decompress, get_compression_extension, is_compression)
from tests.common.datasets import SequenceDataset, write_mds_dataset


class TestBrotli:

    def test_constructor(self):
        brotli = Brotli()
        assert brotli.level == 11

    def test_extension(self):
        brotli = Brotli()
        assert brotli.extension == 'br'

    def test_levels(self):
        brotli = Brotli()
        levels = list(range(12))
        assert brotli.levels == levels

    @pytest.mark.parametrize('level', list(range(12)))
    @pytest.mark.parametrize(
        'data',
        [
            str.encode('Hello World 789*!' * 10, encoding='utf-8'),
            np.int64(1234).tobytes(),
            np.random.randint(0, 255, (32, 32, 3)).tobytes(),
        ],
        ids=['string', 'int', 'image'],
    )
    def test_comp_decomp(self, level: int, data: bytes):
        brotli = Brotli(level)
        output = brotli.decompress(brotli.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2, 'bigdata'])
    def test_invalid_data(self, data: Any):
        brotli = Brotli()
        with pytest.raises(TypeError):
            _ = brotli.compress(data)


class TestBzip2:

    def test_constructor(self):
        bzip2 = Bzip2()
        assert bzip2.level == 9

    def test_extension(self):
        bzip2 = Bzip2()
        assert bzip2.extension == 'bz2'

    def test_levels(self):
        bzip2 = Bzip2()
        levels = list(range(1, 10))
        assert bzip2.levels == levels

    @pytest.mark.parametrize('level', list(range(1, 10)))
    @pytest.mark.parametrize(
        'data',
        [
            str.encode('Hello World 789*!' * 10, encoding='utf-8'),
            np.int64(1234).tobytes(),
            np.random.randint(0, 255, (32, 32, 3)).tobytes(),
        ],
        ids=['string', 'int', 'image'],
    )
    def test_comp_decomp(self, level: int, data: bytes):
        bzip2 = Bzip2(level)
        output = bzip2.decompress(bzip2.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2, 'bigdata'])
    def test_invalid_data(self, data: Any):
        bzip2 = Bzip2()
        with pytest.raises(TypeError):
            _ = bzip2.compress(data)


class TestGzip:

    def test_constructor(self):
        gzip = Gzip()
        assert gzip.level == 9

    def test_extension(self):
        gzip = Gzip()
        assert gzip.extension == 'gz'

    def test_levels(self):
        gzip = Gzip()
        levels = list(range(10))
        assert gzip.levels == levels

    @pytest.mark.parametrize('level', list(range(10)))
    @pytest.mark.parametrize(
        'data',
        [
            str.encode('Hello World 789*!' * 10, encoding='utf-8'),
            np.int64(1234).tobytes(),
            np.random.randint(0, 255, (32, 32, 3)).tobytes(),
        ],
        ids=['string', 'int', 'image'],
    )
    def test_comp_decomp(self, level: int, data: bytes):
        gzip = Gzip(level)
        output = gzip.decompress(gzip.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2, 'bigdata'])
    def test_invalid_data(self, data: Any):
        gzip = Gzip()
        with pytest.raises(TypeError):
            _ = gzip.compress(data)


class TestSnappy:

    def test_extension(self):
        snappy = Snappy()
        assert snappy.extension == 'snappy'

    @pytest.mark.parametrize(
        'data',
        [
            str.encode('Hello World 789*!' * 10, encoding='utf-8'),
            np.int64(1234).tobytes(),
            np.random.randint(0, 255, (32, 32, 3)).tobytes(),
        ],
        ids=['string', 'int', 'image'],
    )
    def test_comp_decomp(self, data: bytes):
        snappy = Snappy()
        output = snappy.decompress(snappy.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2])
    def test_invalid_data(self, data: Any):
        snappy = Snappy()
        with pytest.raises(TypeError):
            _ = snappy.compress(data)


class TestZstandard:

    def test_constructor(self):
        zstd = Zstandard()
        assert zstd.level == 3

    def test_extension(self):
        zstd = Zstandard()
        assert zstd.extension == 'zstd'

    def test_levels(self):
        zstd = Zstandard()
        levels = list(range(1, 23))
        assert zstd.levels == levels

    @pytest.mark.parametrize('level', list(range(1, 23)))
    @pytest.mark.parametrize(
        'data',
        [
            str.encode('Hello World 789*!' * 10, encoding='utf-8'),
            np.int64(1234).tobytes(),
            np.random.randint(0, 255, (32, 32, 3)).tobytes(),
        ],
        ids=['string', 'int', 'image'],
    )
    def test_comp_decomp(self, level: int, data: bytes):
        zstd = Zstandard(level)
        output = zstd.decompress(zstd.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2, 'bigdata'])
    def test_invalid_data(self, data: Any):
        zstd = Zstandard()
        with pytest.raises(TypeError):
            _ = zstd.compress(data)


@pytest.mark.parametrize(('algo', 'is_valid'), [('br', True), ('br:7', True), ('bz2', True),
                                                ('gz', True), ('gz:4', True), ('snappy', True),
                                                ('zstd', True), ('xyz', False)])
def test_is_compression(algo: str, is_valid: bool):
    is_algo_valid = is_compression(algo)
    assert is_algo_valid == is_valid


@pytest.mark.parametrize('algo', ['br', 'bz2', 'gz', 'snappy', 'zstd'])
def test_success_get_compression_extension(algo: str):
    extension = get_compression_extension(algo)
    assert extension == algo


@pytest.mark.parametrize('algo', ['xyz'])
def test_invalid_compression_extension(algo: str):
    with pytest.raises(ValueError) as exc_info:
        _ = get_compression_extension(algo)
    assert exc_info.match(r'.*is not a supported compression algorithm.*')


@pytest.mark.parametrize(
    ('algo', 'data', 'expected_data'), [('br:1', b'hello', b'\x0b\x02\x80hello\x03'),
                                        (None, b'hello', b'hello')])
def test_compress(algo: Optional[str], data: bytes, expected_data: bytes):
    output = compress(algo, data)
    assert output == expected_data


def test_compress_invalid_compression_algo():
    with pytest.raises(ValueError) as exc_info:
        _ = compress('br:99', b'hello')
    assert exc_info.match(r'.*is not a supported compression algorithm.*')


@pytest.mark.parametrize(('algo', 'data', 'expected_data'),
                         [('br:1', b'\x0b\x02\x80hello\x03', b'hello'),
                          (None, b'\x0b\x02\x80hello\x03', b'\x0b\x02\x80hello\x03')])
def test_decompress(algo: Optional[str], data: bytes, expected_data: bytes):
    output = decompress(algo, data)
    assert output == expected_data


def test_decompress_invalid_compression_algo():
    with pytest.raises(ValueError) as exc_info:
        _ = decompress('gz:99', b'\x0b\x02\x80hello\x03')
    assert exc_info.match(r'.*is not a supported compression algorithm.*')


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


@pytest.mark.parametrize('compression', [None, 'br:11', 'bz2:9', 'gz:5', 'snappy', 'zstd:15'])
@pytest.mark.parametrize('num_samples', [9867])
@pytest.mark.parametrize('size_limit', [16_384])
def test_dataset_compression(compressed_remote_local: Tuple[str, str, str],
                             compression: Optional[str], num_samples: int, size_limit: int):
    shuffle = True
    compressed, remote, local = compressed_remote_local
    samples = SequenceDataset(num_samples)
    columns = dict(zip(samples.column_names, samples.column_encodings))
    compression_ext = compression.split(':')[0] if compression else None

    write_mds_dataset(dirname=compressed,
                      columns=columns,
                      samples=samples,
                      size_limit=size_limit,
                      compression=compression)
    samples = SequenceDataset(num_samples)
    write_mds_dataset(dirname=remote,
                      columns=columns,
                      samples=samples,
                      size_limit=size_limit,
                      compression=None)

    dataset = StreamingDataset(local=local, remote=compressed, shuffle=shuffle)

    for _ in dataset:
        pass  # download sample

    dcmp = dircmp(remote, local)
    check_for_diff_files(dcmp, compression_ext)
