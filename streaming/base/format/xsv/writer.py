# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

""":class:`XSVWriter` writes samples to `.xsv` files that can be read by :class:`XSVReader`."""

import json
from typing import Any, Dict, List, Optional, Tuple

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
        local: (str, optional): Optional local output dataset directory. If not provided, a random
            temp directory will be used. If ``remote`` is provided, this is where shards are
            cached before uploading. One or both of ``local`` and ``remote`` must be provided.
            Defaults to ``None``.
        remote: (str, optional): Optional remote output dataset directory. If not provided, no
            uploading will be done. Defaults to ``None``.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Defaults to ``None``.
        **kwargs (Any): Additional settings for the Writer.
    """

    format = 'xsv'

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 separator: str,
                 newline: str = '\n',
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 **kwargs: Any) -> None:
        super().__init__(local=local,
                         remote=remote,
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
        local: (str, optional): Optional local output dataset directory. If not provided, a random
            temp directory will be used. If ``remote`` is provided, this is where shards are
            cached before uploading. One or both of ``local`` and ``remote`` must be provided.
            Defaults to ``None``.
        remote: (str, optional): Optional remote output dataset directory. If not provided, no
            uploading will be done. Defaults to ``None``.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Defaults to ``None``.
        **kwargs (Any): Additional settings for the Writer.
    """

    format = 'csv'
    separator = ','

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 newline: str = '\n',
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 **kwargs: Any) -> None:
        super().__init__(columns=columns,
                         separator=self.separator,
                         newline=newline,
                         local=local,
                         remote=remote,
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
        local: (str, optional): Optional local output dataset directory. If not provided, a random
            temp directory will be used. If ``remote`` is provided, this is where shards are
            cached before uploading. One or both of ``local`` and ``remote`` must be provided.
            Defaults to ``None``.
        remote: (str, optional): Optional remote output dataset directory. If not provided, no
            uploading will be done. Defaults to ``None``.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Defaults to ``None``.
        **kwargs (Any): Additional settings for the Writer.
    """

    format = 'tsv'
    separator = '\t'

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 newline: str = '\n',
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 **kwargs: Any) -> None:
        super().__init__(columns=columns,
                         separator=self.separator,
                         newline=newline,
                         local=local,
                         remote=remote,
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
