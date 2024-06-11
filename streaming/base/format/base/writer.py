# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Serialize samples into streaming dataset shards and index."""

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Future
from threading import Event
from time import sleep
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from typing_extensions import Self

from streaming.base.compression import compress, get_compression_extension, is_compression
from streaming.base.format.index import get_index_basename
from streaming.base.hashing import get_hash, is_hash
from streaming.base.storage.upload import CloudUploader
from streaming.base.util import bytes_to_int

__all__ = ['JointWriter', 'SplitWriter']

logger = logging.getLogger(__name__)


class Writer(ABC):
    """Writes a streaming dataset.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in the
               `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (Union[int, str], optional): Optional shard size limit, after which point
            to start a new shard. If ``None``, puts everything in one shard. Can specify bytes
            human-readable format as well, for example ``"100kb"`` for 100 kilobyte
            (100*1024) and so on. Defaults to ``1 << 26``.
        extra_bytes_per_shard (int): Extra bytes per serialized shard (for computing shard size
            while writing). Defaults to ``0``.
        extra_bytes_per_sample (int): Extra bytes per serialized sample (for computing shard size
            while writing). Defaults to ``0``.
        **kwargs (Any): Additional settings for the Writer.

            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            max_workers (int): Maximum number of threads used to upload output dataset files in
                parallel to a remote location. One thread is responsible for uploading one shard
                file to a remote location. Default to ``min(32, (os.cpu_count() or 1) + 4)``.
            retry (int): Number of times to retry uploading a file to a remote location.
                Default to ``2``.
            exist_ok (bool): If the local directory exists and is not empty, whether to overwrite
                the content or raise an error. `False` raises an error. `True` deletes the
                content and starts fresh. Defaults to `False`.
    """

    format: str = ''  # Name of the format (like "mds", "csv", "json", etc).

    def __init__(self,
                 *,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[Union[int, str]] = 1 << 26,
                 extra_bytes_per_shard: int = 0,
                 extra_bytes_per_sample: int = 0,
                 **kwargs: Any) -> None:

        compression = compression or None
        if compression:
            if not is_compression(compression):
                raise ValueError(f'Invalid compression: {compression}.')

        hashes = hashes or []
        if list(hashes) != sorted(hashes):
            raise ValueError('Hashes must be unique and in sorted order.')
        for algo in hashes:
            if not is_hash(algo):
                raise ValueError(f'Invalid hash: {algo}.')

        size_limit_value = None
        if size_limit:
            size_limit_value = bytes_to_int(size_limit)
            if size_limit_value < 0:
                raise ValueError(f'`size_limit` must be greater than zero, instead, ' +
                                 f'found as {size_limit_value}.')
            if size_limit_value >= 2**32:
                raise ValueError(f'`size_limit` must be less than 2**32, instead, ' +
                                 f'found as {size_limit_value}. This is because sample ' +
                                 f'byte offsets are stored with uint32.')

        # Validate keyword arguments
        invalid_kwargs = [
            arg for arg in kwargs.keys()
            if arg not in ('progress_bar', 'max_workers', 'retry', 'exist_ok')
        ]
        if invalid_kwargs:
            raise ValueError(f'Invalid Writer argument(s): {invalid_kwargs} ')

        self.keep_local = keep_local
        self.compression = compression
        self.hashes = hashes
        self.size_limit = size_limit_value
        self.extra_bytes_per_shard = extra_bytes_per_shard
        self.extra_bytes_per_sample = extra_bytes_per_sample
        self.new_samples: List[bytes]
        self.new_shard_size: int

        self.shards = []

        # Remove local directory if requested prior to creating writer
        local = os.path.expanduser(out) if isinstance(out, str) else os.path.expanduser(out[0])
        if os.path.exists(local) and kwargs.get('exist_ok', False):
            logger.warning(
                f'Directory {local} exists and is not empty; exist_ok is set to True so will remove contents.'
            )
            shutil.rmtree(local)
        self.cloud_writer = CloudUploader.get(out, keep_local, kwargs.get('progress_bar', False),
                                              kwargs.get('retry', 2))
        self.local = self.cloud_writer.local
        self.remote = self.cloud_writer.remote
        # `max_workers`: The maximum number of threads that can be executed in parallel.
        # One thread is responsible for uploading one shard file to a remote location.
        self.executor = ThreadPoolExecutor(max_workers=kwargs.get('max_workers', None))
        # Create an event to track an exception in a thread.
        self.event = Event()

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
        if self.event.is_set():
            # Shutdown the executor and cancel all the pending futures due to exception in one of
            # the threads.
            self.cancel_future_jobs()
            raise Exception('One of the threads failed. Check other traceback for more ' +
                            'details.')
        # Execute the task if there is no exception in any of the async threads.
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
        if self.event.is_set():
            # Shutdown the executor and cancel all the pending futures due to exception in one of
            # the threads.
            self.cancel_future_jobs()
            return
        basename = get_index_basename()
        filename = os.path.join(self.local, basename)
        obj = {
            'version': 2,
            'shards': self.shards,
        }
        with open(filename, 'w') as out:
            json.dump(obj, out, sort_keys=True)
        # Execute the task if there is no exception in any of the async threads.
        while self.executor._work_queue.qsize() > 0:
            logger.debug(
                f'Queue size: {self.executor._work_queue.qsize()}. Waiting for all ' +
                f'shard files to get uploaded to {self.remote} before uploading index.json')
            sleep(1)
        logger.debug(f'Queue size: {self.executor._work_queue.qsize()}. Uploading ' +
                     f'index.json to {self.remote}')
        future = self.executor.submit(self.cloud_writer.upload_file, basename)
        future.add_done_callback(self.exception_callback)

    def finish(self) -> None:
        """Finish writing samples."""
        if self.new_samples:
            self.flush_shard()
            self._reset_cache()
        self._write_index()
        logger.debug(f'Waiting for all shard files to get uploaded to {self.remote}')
        self.executor.shutdown(wait=True)
        if self.remote and not self.keep_local:
            shutil.rmtree(self.local, ignore_errors=True)

    def cancel_future_jobs(self) -> None:
        """Shutting down the executor and cancel all the pending jobs."""
        # Beginning python v3.9, ThreadPoolExecutor.shutdown() has a new parameter `cancel_futures`
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.remote and not self.keep_local:
            shutil.rmtree(self.local, ignore_errors=True)

    def exception_callback(self, future: Future) -> None:
        """Raise an exception to the caller if exception generated by one of an async thread.

        Also, set the thread event to let other threads knows about the exception in one of the
        thread.

        Args:
            future (Future): Contains the status of the task

        Raises:
            exception: re-raise an exception
        """
        exception = future.exception()
        if exception:
            # Set the event to let other pool thread know about the exception
            self.event.set()
            # re-raise the same exception
            raise exception

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
        if self.event.is_set():
            # Shutdown the executor and cancel all the pending futures due to exception in one of
            # the threads.
            self.cancel_future_jobs()
            return
        self.finish()


class JointWriter(Writer):
    """Writes a streaming dataset with joint shards.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in the
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

            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            max_workers (int): Maximum number of threads used to upload output dataset files in
                parallel to a remote location. One thread is responsible for uploading one shard
                file to a remote location. Default to ``min(32, (os.cpu_count() or 1) + 4)``.
            retry (int): Number of times to retry uploading a file to a remote location.
                Default to ``2``.
    """

    def __init__(self,
                 *,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[Union[int, str]] = 1 << 26,
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
        if self.event.is_set():
            # Shutdown the executor and cancel all the pending futures due to exception in one of
            # the threads.
            self.cancel_future_jobs()
            return

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

        # Execute the task if there is no exception in any of the async threads.
        future = self.executor.submit(self.cloud_writer.upload_file, zip_data_basename or
                                      raw_data_basename)
        future.add_done_callback(self.exception_callback)


class SplitWriter(Writer):
    """Writes a streaming dataset with split shards.

    Split shards refer to raw data (csv, json, etc.) paired with an index into it.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in the
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

            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            max_workers (int): Maximum number of threads used to upload output dataset files in
                parallel to a remote location. One thread is responsible for uploading one shard
                file to a remote location. Default to ``min(32, (os.cpu_count() or 1) + 4)``.
            retry (int): Number of times to retry uploading a file to a remote location.
                Default to ``2``.
    """

    extra_bytes_per_shard = 0
    extra_bytes_per_sample = 0

    def __init__(self,
                 *,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[Union[int, str]] = 1 << 26,
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
        if self.event.is_set():
            # Shutdown the executor and cancel all the pending futures due to exception in one of
            # the threads.
            self.cancel_future_jobs()
            return

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

        # Execute the task if there is no exception in any of the async threads.
        future = self.executor.submit(self.cloud_writer.upload_file, zip_data_basename or
                                      raw_data_basename)
        future.add_done_callback(self.exception_callback)

        # Execute the task if there is no exception in any of the async threads.
        future = self.executor.submit(self.cloud_writer.upload_file, zip_meta_basename or
                                      raw_meta_basename)
        future.add_done_callback(self.exception_callback)
