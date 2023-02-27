# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Serialize samples into streaming dataset shards and index."""

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from typing_extensions import Self

from streaming.base.compression import compress, get_compression_extension, is_compression
from streaming.base.hashing import get_hash, is_hash
from streaming.base.index import get_index_basename
from streaming.base.storage.upload import CloudWriter

__all__ = ['JointWriter', 'SplitWriter']

logger = logging.getLogger(__name__)


class Writer(ABC):
    """Writes a streaming dataset.

    Args:
        out (str | List[str]): Output dataset directory to save shard files.
            1. If `out` is a local directory, shard files are saved locally.
            2. If `out` is a remote directory, a random local temporary directory is created to
               cached the shard files and then the shard files are uploaded to a remote location.
               At the end, a temp directory is deleted once shards are uploaded.
            3. If `out` is a list of `(local_dir, remote_dir)`, shard files are saved in the
               `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
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
        **kwargs (Any): Additional settings for the Writer.
    """

    format: str = ''  # Name of the format (like "mds", "csv", "json", etc).

    def __init__(self,
                 *,
                 out: Union[str, List[str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 extra_bytes_per_shard: int = 0,
                 extra_bytes_per_sample: int = 0,
                 **kwargs: Any) -> None:

        compression = compression or None
        if compression:
            if not is_compression(compression):
                raise ValueError('Invalid compression: {compression}.')

        hashes = hashes or []
        if list(hashes) != sorted(hashes):
            raise ValueError('Hashes must be unique and in sorted order.')
        for algo in hashes:
            if not is_hash(algo):
                raise ValueError('Invalid hash: {algo}.')

        if size_limit:
            if size_limit < 0:
                raise ValueError('Size limit, if provided, must be greater than zero.')
        else:
            size_limit = None

        self.keep_local = keep_local
        self.compression = compression
        self.hashes = hashes
        self.size_limit = size_limit
        self.extra_bytes_per_shard = extra_bytes_per_shard
        self.extra_bytes_per_sample = extra_bytes_per_sample
        self.new_samples: List[bytes]
        self.new_shard_size: int

        self.shards = []

        self.cloud_writer = CloudWriter.get(out, keep_local, kwargs.get('progress_bar', False))
        self.local = self.cloud_writer.local
        self.remote = self.cloud_writer.remote
        # `max_workers`: The maximum number of threads that can be executed in parallel.
        # One thread is responsible for uploading one shard files to a remote location.
        self.executor = ThreadPoolExecutor(max_workers=kwargs.get('max_workers', None))

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
        filename = os.path.join(self.local, basename)
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
        if self.new_samples:
            raise RuntimeError('Internal error: not all samples have been written.')
        basename = get_index_basename()
        filename = os.path.join(self.local, basename)
        obj = {
            'version': 2,
            'shards': self.shards,
        }
        with open(filename, 'w') as out:
            json.dump(obj, out, sort_keys=True)
        self.executor.submit(self.cloud_writer.upload_file, basename)

    def finish(self) -> None:
        """Finish writing samples."""
        if self.new_samples:
            self.flush_shard()
            self._reset_cache()
        self._write_index()
        logger.debug(f'Waiting for all shard files to get uploaded to {self.remote}')
        self.executor.shutdown(wait=True, cancel_futures=False)
        if self.remote and not self.keep_local:
            shutil.rmtree(self.local)

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
        out (str | List[str]): Output dataset directory to save shard files.
            1. If `out` is a local directory, shard files are saved locally.
            2. If `out` is a remote directory, a random local temporary directory is created to
               cached the shard files and then the shard files are uploaded to a remote location.
               At the end, a temp directory is deleted once shards are uploaded.
            3. If `out` is a list of `(local_dir, remote_dir)`, shard files are saved in the
               `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
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
        **kwargs (Any): Additional settings for the Writer.
    """

    def __init__(self,
                 *,
                 out: Union[str, List[str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 extra_bytes_per_shard: int = 0,
                 extra_bytes_per_sample: int = 0,
                 **kwargs: Any) -> None:
        super().__init__(out=out,
                         keep_local=keep_local,
                         compression=compression,
                         hashes=hashes,
                         size_limit=size_limit,
                         extra_bytes_per_shard=extra_bytes_per_shard,
                         extra_bytes_per_sample=extra_bytes_per_sample,
                         **kwargs)

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

        self.executor.submit(self.cloud_writer.upload_file, zip_data_basename or raw_data_basename)


class SplitWriter(Writer):
    """Writes a streaming dataset with split shards.

    Split shards refer to raw data (csv, json, etc.) paired with an index into it.

    Args:
        out (str | List[str]): Output dataset directory to save shard files.
            1. If `out` is a local directory, shard files are saved locally.
            2. If `out` is a remote directory, a random local temporary directory is created to
               cached the shard files and then the shard files are uploaded to a remote location.
               At the end, a temp directory is deleted once shards are uploaded.
            3. If `out` is a list of `(local_dir, remote_dir)`, shard files are saved in the
               `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Defaults to ``1 << 26``.
        **kwargs (Any): Additional settings for the Writer.
    """

    extra_bytes_per_shard = 0
    extra_bytes_per_sample = 0

    def __init__(self,
                 *,
                 out: Union[str, List[str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 **kwargs: Any) -> None:
        super().__init__(out=out,
                         keep_local=keep_local,
                         compression=compression,
                         hashes=hashes,
                         size_limit=size_limit,
                         extra_bytes_per_shard=self.extra_bytes_per_shard,
                         extra_bytes_per_sample=self.extra_bytes_per_sample,
                         **kwargs)

    @abstractmethod
    def encode_split_shard(self) -> Tuple[bytes, bytes]:
        """Encode a split shard out of the cached samples (data, meta files).

        Returns:
            Tuple[bytes, bytes]: Data file, meta file.
        """
        raise NotImplementedError

    def flush_shard(self) -> None:
        raw_data_basename, zip_data_basename = self._name_next_shard()
        raw_meta_basename, zip_meta_basename = self._name_next_shard('meta')
        raw_data, raw_meta = self.encode_split_shard()
        raw_data_info, zip_data_info = self._process_file(raw_data, raw_data_basename,
                                                          zip_data_basename)
        raw_meta_info, zip_meta_info = self._process_file(raw_meta, raw_meta_basename,
                                                          zip_meta_basename)
        obj = {
            'samples': len(self.new_samples),
            'raw_data': raw_data_info,
            'zip_data': zip_data_info,
            'raw_meta': raw_meta_info,
            'zip_meta': zip_meta_info
        }
        obj.update(self.get_config())
        self.shards.append(obj)

        self.executor.submit(self.cloud_writer.upload_file, zip_data_basename or raw_data_basename)
        self.executor.submit(self.cloud_writer.upload_file, zip_meta_basename or raw_meta_basename)
