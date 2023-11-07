# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""List of Compression and Decompression algorithms."""

import bz2
import gzip
from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional, Set, Tuple, Type

import brotli
import snappy
import zstd
from typing_extensions import Self

__all__ = [
    'compress', 'decompress', 'get_compression_extension', 'get_compressions', 'is_compression'
]


class Compression(ABC):
    """A compression algorithm family."""

    extension: str = ''  # Filename extension.

    @classmethod
    def each(cls) -> Iterator[Tuple[str, Self]]:
        """Get each instance of this compression algorithm family.

        Returns:
            Iterator[Tuple[str, Self]]: Each level.
        """
        yield cls.extension, cls()

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress arbitrary data.

        Args:
            data (bytes): Uncompressed data.

        Returns:
            bytes: Compressed data.
        """
        raise NotImplementedError

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data compressed by this algorithm.

        Args:
            data (bytes): Compressed data.

        Returns:
            bytes: Decompressed data.
        """
        raise NotImplementedError


class LevelledCompression(Compression):
    """Compression with levels.

    Args:
        level (int, optional): Compression level. Defaults to ``None``.
    """

    levels: list = []  # Compression levels.

    def __init__(self, level: Optional[int] = None) -> None:
        raise NotImplementedError

    @classmethod
    def each(cls) -> Iterator[Tuple[str, Self]]:
        yield cls.extension, cls()
        for level in cls.levels:
            yield f'{cls.extension}:{level}', cls(level)


class Brotli(LevelledCompression):
    """Brotli compression."""

    extension = 'br'
    levels = list(range(12))

    def __init__(self, level: int = 11) -> None:
        assert level in self.levels
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return brotli.compress(data, quality=self.level)

    def decompress(self, data: bytes) -> bytes:
        return brotli.decompress(data)


class Bzip2(LevelledCompression):
    """Bzip2 compression."""

    extension = 'bz2'
    levels = list(range(1, 10))

    def __init__(self, level: int = 9) -> None:
        assert level in self.levels
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return bz2.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return bz2.decompress(data)


class Gzip(LevelledCompression):
    """Gzip compression."""

    extension = 'gz'
    levels = list(range(10))

    def __init__(self, level: int = 9) -> None:
        assert level in self.levels
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class Snappy(Compression):
    """Snappy compression."""

    extension = 'snappy'

    def compress(self, data: bytes) -> bytes:
        return snappy.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return snappy.decompress(data)


class Zstandard(LevelledCompression):
    """Zstandard compression."""

    extension = 'zstd'
    levels = list(range(1, 23))

    def __init__(self, level: int = 3) -> None:
        assert level in self.levels
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return zstd.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return zstd.decompress(data)


# Compression algorithm families (extension -> class).
_families: Dict[str, Type[Compression]] = {
    'br': Brotli,
    'bz2': Bzip2,
    'gz': Gzip,
    'snappy': Snappy,
    'zstd': Zstandard,
}


def _collect(families: Dict[str, Type[Compression]]) -> Dict[str, Compression]:
    """Get each level of each compression type and flatten into a single dict.

    Args:
        Dict[str, Type[Compression]]: Mapping of extension to class.

    Returns:
        Dict[str, Compression]: Mapping of extension:level to instance.
    """
    algos = {}
    for cls in families.values():
        for algo, obj in cls.each():
            algos[algo] = obj
    return algos


# Compression algorithms (extension:level -> instance).
_algorithms: Dict[str, Compression] = _collect(_families)


def get_compressions() -> Set[str]:
    """List supported compression algorithms.

    Returns:
        Set[str]: Compression algorithms.
    """
    return set(_algorithms)


def is_compression(algo: Optional[str]) -> bool:
    """Get whether this compression algorithm is supported.

    Args:
        algo (str, optional): Compression.

    Returns:
        bool: Whether supported.
    """
    return algo in _algorithms


def get_compression_extension(algo: str) -> str:
    """Get compressed filename extension.

    Args:
        algo (str): Compression.

    Returns:
        str: Filename extension.
    """
    if not is_compression(algo):
        raise ValueError(f'{algo} is not a supported compression algorithm.')
    obj = _algorithms[algo]
    return obj.extension


def compress(algo: Optional[str], data: bytes) -> bytes:
    """Compress arbitrary data.

    Args:
        algo (str, optional): Compression.
        data (bytes): Uncompressed data.

    Returns:
        bytes: Compressed data.
    """
    if algo is None:
        return data
    if not is_compression(algo):
        raise ValueError(f'{algo} is not a supported compression algorithm.')
    obj = _algorithms[algo]
    return obj.compress(data)


def decompress(algo: Optional[str], data: bytes) -> bytes:
    """Decompress data compressed by this algorithm.

    Args:
        algo (str, optional): Compression.
        data (bytes): Compressed data.

    Returns:
        bytes: Decompressed data.
    """
    if algo is None:
        return data
    if not is_compression(algo):
        raise ValueError(f'{algo} is not a supported compression algorithm.')
    obj = _algorithms[algo]
    return obj.decompress(data)
