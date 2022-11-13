# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Serialize samples into streaming dataset shards and index."""

import json
import os
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type

from typing_extensions import Self

from streaming.base.compression import compress, get_compression_extension, is_compression
from streaming.base.hashing import get_hash, is_hash
from streaming.base.index import get_index_basename

__all__ = ['JointWriter', 'SplitWriter']


class Writer(ABC):
    """Writes a streaming dataset.

    Args:
        dirname (str): Local dataset directory.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If ``None``, puts everything in one shard. Defaults to ``1 << 26``.
        extra_bytes_per_shard (int): Extra bytes per serialized shard (for computing shard size
            while writing). Defaults to ``0``.
        extra_bytes_per_sample (int): Extra bytes per serialized sample (for computing shard size
            while writing). Defaults to ``0``.
    """

    format: str = ''  # Name of the format (like "mds", "csv", "json", etc).

    def __init__(self,
                 dirname: str,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 extra_bytes_per_shard: int = 0,
                 extra_bytes_per_sample: int = 0) -> None:
        compression = compression or None
        if compression:
            assert is_compression(compression)

        hashes = hashes or []
        assert list(hashes) == sorted(hashes)
        for algo in hashes:
            assert is_hash(algo)

        if size_limit:
            assert 0 < size_limit
        else:
            size_limit = None

        self.dirname = dirname
        self.compression = compression
        self.hashes = hashes
        self.size_limit = size_limit
        self.extra_bytes_per_shard = extra_bytes_per_shard
        self.extra_bytes_per_sample = extra_bytes_per_sample
        self.new_samples: List[bytes]
        self.new_shard_size: int

        self.shards = []

        # Raise an exception if the directory is not empty
        if os.path.exists(dirname) and len(os.listdir(dirname)) != 0:
            raise FileExistsError(f'Directory is not empty: {dirname}')
        os.makedirs(dirname, exist_ok=True)

        self._reset_cache()

    def _reset_cache(self) -> None:
        """Reset our internal shard-building cache.

        This is called on init or after writing a shard.
        """
        self.new_samples = []
        self.new_shard_size = self.extra_bytes_per_shard

    @abstractmethod
    def encode_sample(self, sample: Dict[str, Any]) -> bytes:
        """Encode a sample dict to bytes.

        Args:
            sample (Dict[str, Any]): Sample dict.

        Returns:
            bytes: Sample encoded as bytes.
        """
        raise NotImplementedError

    def _name_next_shard(self, extension: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Get the filenames of the next shard to be created.

        Args:
            extension (str): Optional additional extension (eg, meta files).

        Returns:
            Tuple[str, str]: Pair of (decompressed, compressed) filenames.
        """
        shard = len(self.shards)
        parts = ['shard', f'{shard:05}', self.format]
        if extension:
            parts.append(extension)
        raw_basename = '.'.join(parts)
        if self.compression:
            ext = get_compression_extension(self.compression)
            parts.append(ext)
            zip_basename = '.'.join(parts)
        else:
            zip_basename = None
        return raw_basename, zip_basename

    def _hash(self, data: bytes, basename: str) -> Dict[str, Any]:
        """Generate file metadata.

        Args:
            data (bytes): The file data.
            basename (str): The file's basename.

        Returns:
            Dict[str, Any]: File metadata.
        """
        hashes = {}
        for algo in self.hashes:
            hashes[algo] = get_hash(algo, data)
        return {'basename': basename, 'bytes': len(data), 'hashes': hashes}

    def _process_file(self, raw_data: bytes, raw_basename: str,
                      zip_basename: Optional[str]) -> Tuple[dict, Optional[dict]]:
        """Process and save a shard file (hash, compress, hash, write).

        Args:
            raw_data (bytes): Uncompressed data.
            raw_basename (str): Uncompressed basename.
            zip_basename (str): Compressed basename.

        Returns:
            Dict[str, Any]: Metadata containing basename, size, and hashes.
        """
        raw_info = self._hash(raw_data, raw_basename)
        if zip_basename:
            zip_data = compress(self.compression, raw_data)
            zip_info = self._hash(zip_data, zip_basename)
            data = zip_data
            basename = zip_basename
        else:
            zip_info = None
            data = raw_data
            basename = raw_basename
        filename = os.path.join(self.dirname, basename)
        with open(filename, 'wb') as out:
            out.write(data)
        return raw_info, zip_info

    def get_config(self) -> Dict[str, Any]:
        """Get object describing shard-writing configuration.

        Returns:
            Dict[str, Any]: JSON object.
        """
        return {
            'version': 2,
            'format': self.format,
            'compression': self.compression,
            'hashes': self.hashes,
            'size_limit': self.size_limit
        }

    @abstractmethod
    def flush_shard(self) -> None:
        """Flush cached samples to storage, creating a new shard."""
        raise NotImplementedError

    def write(self, sample: Dict[str, Any]) -> None:
        """Write a sample.

        May flush an entire new shard, then caches the sample.

        Args:
            sample (Dict[str, Any]): Sample dict.
        """
        new_sample = self.encode_sample(sample)
        new_sample_size = len(new_sample) + self.extra_bytes_per_sample
        if self.size_limit and self.size_limit < self.new_shard_size + new_sample_size:
            self.flush_shard()
            self._reset_cache()
        self.new_samples.append(new_sample)
        self.new_shard_size += new_sample_size

    def _write_index(self) -> None:
        """Write the index, having written all the shards."""
        assert not self.new_samples
        filename = os.path.join(self.dirname, get_index_basename())
        obj = {
            'version': 2,
            'shards': self.shards,
        }
        with open(filename, 'w') as out:
            json.dump(obj, out, sort_keys=True)

    def finish(self) -> None:
        """Finish writing samples."""
        if self.new_samples:
            self.flush_shard()
            self._reset_cache()
        self._write_index()

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns:
            Self: This object.
        """
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """Exit context manager.

        Args:
            exc_type (Type[BaseException], optional): Exc type.
            exc (BaseException, optional): Exc.
            traceback (TracebackType, optional): Traceback.
        """
        self.finish()


class JointWriter(Writer):
    """Writes a streaming dataset with joint shards.

    Args:
        dirname (str): Local dataset directory.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Defaults to ``1 << 26``.
        extra_bytes_per_shard (int): Extra bytes per serialized shard (for computing shard size
            while writing). Defaults to ``0``.
        extra_bytes_per_sample (int): Extra bytes per serialized sample (for computing shard size
            while writing). Defaults to ``0``.
    """

    def __init__(self,
                 dirname: str,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 extra_bytes_per_shard: int = 0,
                 extra_bytes_per_sample: int = 0) -> None:
        super().__init__(dirname, compression, hashes, size_limit, extra_bytes_per_shard,
                         extra_bytes_per_sample)

    @abstractmethod
    def encode_joint_shard(self) -> bytes:
        """Encode a joint shard out of the cached samples (single file).

        Returns:
            bytes: File data.
        """
        raise NotImplementedError

    def flush_shard(self) -> None:
        raw_data_basename, zip_data_basename = self._name_next_shard()
        raw_data = self.encode_joint_shard()
        raw_data_info, zip_data_info = self._process_file(raw_data, raw_data_basename,
                                                          zip_data_basename)
        obj = {
            'samples': len(self.new_samples),
            'raw_data': raw_data_info,
            'zip_data': zip_data_info
        }
        obj.update(self.get_config())
        self.shards.append(obj)


class SplitWriter(Writer):
    """Writes a streaming dataset with split shards.

    Split shards refer to raw data (csv, json, etc.) paired with an index into it.

    Args:
        dirname (str): Local dataset directory.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Defaults to ``1 << 26``.
    """

    extra_bytes_per_shard = 0
    extra_bytes_per_sample = 0

    def __init__(self,
                 dirname: str,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26) -> None:
        super().__init__(dirname, compression, hashes, size_limit, self.extra_bytes_per_shard,
                         self.extra_bytes_per_sample)

    @abstractmethod
    def encode_split_shard(self) -> Tuple[bytes, bytes]:
        """Encode a split shard out of the cached samples (data, meta files).

        Returns:
            Tuple[bytes, bytes]: Data file, meta file.
        """
        raise NotImplementedError

    def flush_shard(self) -> None:
        raw_data_basename, zip_data_basename = self._name_next_shard()
        meta_raw_basename, meta_zip_basename = self._name_next_shard('meta')
        raw_data, raw_meta = self.encode_split_shard()
        raw_data_info, zip_data_info = self._process_file(raw_data, raw_data_basename,
                                                          zip_data_basename)
        raw_meta_info, zip_meta_info = self._process_file(raw_meta, meta_raw_basename,
                                                          meta_zip_basename)
        obj = {
            'samples': len(self.new_samples),
            'raw_data': raw_data_info,
            'zip_data': zip_data_info,
            'raw_meta': raw_meta_info,
            'zip_meta': zip_meta_info
        }
        obj.update(self.get_config())
        self.shards.append(obj)
