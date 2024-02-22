# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

""":class:`XSVWriter` writes samples to `.xsv` files that can be read by :class:`XSVReader`."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from streaming.base.format.base.writer import SplitWriter
from streaming.base.format.xsv.encodings import is_xsv_encoding, xsv_encode

__all__ = ['XSVWriter', 'CSVWriter', 'TSVWriter']


class XSVWriter(SplitWriter):
    r"""Writes a streaming XSV dataset.

    Args:
        columns (Dict[str, str]): Sample columns.
        separator (str): String used to separate columns.
        newline (str): Newline character inserted between samples. Defaults to ``\\n``.
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
        size_limit (Union[int, str], optional): Optional shard size limit, after which point to
            start a new shard. If None, puts everything in one shard. Can specify bytes
            human-readable format as well, for example ``"100kb"`` for 100 kilobyte
            (100*1024) and so on. Defaults to ``1 << 26``
        **kwargs (Any): Additional settings for the Writer.

            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            max_workers (int): Maximum number of threads used to upload output dataset files in
                parallel to a remote location. One thread is responsible for uploading one shard
                file to a remote location. Default to ``min(32, (os.cpu_count() or 1) + 4)``.
            exist_ok (bool): If the local directory exists and is not empty, whether to overwrite
                the content or raise an error. `False` raises an error. `True` deletes the
                content and starts fresh. Defaults to `False`.
    """

    format = 'xsv'

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 separator: str,
                 newline: str = '\n',
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
                         **kwargs)
        self.columns = columns
        self.column_names = []
        self.column_encodings = []
        for name in sorted(columns):
            encoding = columns[name]
            assert newline not in name
            assert separator not in name
            assert is_xsv_encoding(encoding)
            self.column_names.append(name)
            self.column_encodings.append(encoding)

        self.separator = separator
        self.newline = newline

    def encode_sample(self, sample: Dict[str, Any]) -> bytes:
        """Encode a sample dict to bytes.

        Args:
            sample (Dict[str, Any]): Sample dict.

        Returns:
            bytes: Sample encoded as bytes.
        """
        values = []
        for name, encoding in zip(self.column_names, self.column_encodings):
            value = xsv_encode(encoding, sample[name])
            assert self.newline not in value
            assert self.separator not in value
            values.append(value)
        text = self.separator.join(values) + self.newline
        return text.encode('utf-8')

    def get_config(self) -> Dict[str, Any]:
        """Get object describing shard-writing configuration.

        Returns:
            Dict[str, Any]: JSON object.
        """
        obj = super().get_config()
        obj.update({
            'column_names': self.column_names,
            'column_encodings': self.column_encodings,
            'separator': self.separator,
            'newline': self.newline
        })
        return obj

    def encode_split_shard(self) -> Tuple[bytes, bytes]:
        """Encode a split shard out of the cached samples (data, meta files).

        Returns:
            Tuple[bytes, bytes]: Data file, meta file.
        """
        header = self.separator.join(self.column_names) + self.newline
        header = header.encode('utf-8')
        data = b''.join([header] + self.new_samples)
        header_offset = len(header)

        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = header_offset + np.array([0] + sizes).cumsum().astype(np.uint32)
        obj = self.get_config()
        text = json.dumps(obj, sort_keys=True)
        meta = num_samples.tobytes() + offsets.tobytes() + text.encode('utf-8')

        return data, meta


class CSVWriter(XSVWriter):
    r"""Writes a streaming CSV dataset.

    Args:
        columns (Dict[str, str]): Sample columns.
        newline (str): Newline character inserted between samples. Defaults to ``\\n``.
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
            shard. If None, puts everything in one shard. Defaults to ``None``.
        **kwargs (Any): Additional settings for the Writer.

            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            max_workers (int): Maximum number of threads used to upload output dataset files in
                parallel to a remote location. One thread is responsible for uploading one shard
                file to a remote location. Default to ``min(32, (os.cpu_count() or 1) + 4)``.
    """

    format = 'csv'
    separator = ','

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 newline: str = '\n',
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 **kwargs: Any) -> None:
        super().__init__(columns=columns,
                         separator=self.separator,
                         newline=newline,
                         out=out,
                         keep_local=keep_local,
                         compression=compression,
                         hashes=hashes,
                         size_limit=size_limit,
                         **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Get object describing shard-writing configuration.

        Returns:
            Dict[str, Any]: JSON object.
        """
        obj = super().get_config()
        obj['format'] = self.format
        del obj['separator']
        return obj


class TSVWriter(XSVWriter):
    r"""Writes a streaming TSV dataset.

    Args:
        columns (Dict[str, str]): Sample columns.
        newline (str): Newline character inserted between samples. Defaults to ``\\n``.
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
            shard. If None, puts everything in one shard. Defaults to ``None``.
        **kwargs (Any): Additional settings for the Writer.

            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            max_workers (int): Maximum number of threads used to upload output dataset files in
                parallel to a remote location. One thread is responsible for uploading one shard
                file to a remote location. Default to ``min(32, (os.cpu_count() or 1) + 4)``.
    """

    format = 'tsv'
    separator = '\t'

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 newline: str = '\n',
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 **kwargs: Any) -> None:
        super().__init__(columns=columns,
                         separator=self.separator,
                         newline=newline,
                         out=out,
                         keep_local=keep_local,
                         compression=compression,
                         hashes=hashes,
                         size_limit=size_limit,
                         **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Get object describing shard-writing configuration.

        Returns:
            Dict[str, Any]: JSON object.
        """
        obj = super().get_config()
        obj['format'] = self.format
        del obj['separator']
        return obj
