# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest

from streaming.base.compression.compression import (Brotli, Bzip2, Gzip, Snappy, Zstandard,
                                                    compress, decompress,
                                                    get_compression_extension, is_compression)


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
    @pytest.mark.parametrize('data', [
        str.encode('Hello World 789*!' * 10, encoding='utf-8'),
        np.int64(1234).tobytes(),
        np.random.randint(0, 255, (32, 32, 3)).tobytes(),
    ])
    def test_comp_decomp(self, level: int, data: bytes):
        brotli = Brotli(level)
        output = brotli.decompress(brotli.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2, 'bigdata'])
    def test_exception(self, data: Any):
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
    @pytest.mark.parametrize('data', [
        str.encode('Hello World 789*!' * 10, encoding='utf-8'),
        np.int64(1234).tobytes(),
        np.random.randint(0, 255, (32, 32, 3)).tobytes(),
    ])
    def test_comp_decomp(self, level: int, data: bytes):
        bzip2 = Bzip2(level)
        output = bzip2.decompress(bzip2.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2, 'bigdata'])
    def test_exception(self, data: Any):
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
    @pytest.mark.parametrize('data', [
        str.encode('Hello World 789*!' * 10, encoding='utf-8'),
        np.int64(1234).tobytes(),
        np.random.randint(0, 255, (32, 32, 3)).tobytes(),
    ])
    def test_comp_decomp(self, level: int, data: bytes):
        gzip = Gzip(level)
        output = gzip.decompress(gzip.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2, 'bigdata'])
    def test_exception(self, data: Any):
        gzip = Gzip()
        with pytest.raises(TypeError):
            _ = gzip.compress(data)


class TestSnappy:

    def test_extension(self):
        snappy = Snappy()
        assert snappy.extension == 'snappy'

    @pytest.mark.parametrize('data', [
        str.encode('Hello World 789*!' * 10, encoding='utf-8'),
        np.int64(1234).tobytes(),
        np.random.randint(0, 255, (32, 32, 3)).tobytes(),
    ])
    def test_comp_decomp(self, data: bytes):
        snappy = Snappy()
        output = snappy.decompress(snappy.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2])
    def test_exception(self, data: Any):
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
    @pytest.mark.parametrize('data', [
        str.encode('Hello World 789*!' * 10, encoding='utf-8'),
        np.int64(1234).tobytes(),
        np.random.randint(0, 255, (32, 32, 3)).tobytes(),
    ])
    def test_comp_decomp(self, level: int, data: bytes):
        zstd = Zstandard(level)
        output = zstd.decompress(zstd.compress(data))
        assert output == data

    @pytest.mark.parametrize('data', [100, 1.2, 'bigdata'])
    def test_exception(self, data: Any):
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
def test_exception_get_compression_extension(algo: str):
    with pytest.raises(ValueError) as exc_info:
        _ = get_compression_extension(algo)
    assert exc_info.match(r'.*is not a supported compression algorithm.*')


def test_compress():
    data = compress('br:1', b'hello')
    expected_data = b'\x0b\x02\x80hello\x03'
    assert data == expected_data


def test_decompress():
    data = decompress('br:1', b'\x0b\x02\x80hello\x03')
    expected_data = b'hello'
    assert data == expected_data
