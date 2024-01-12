# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Reads and decode samples from tabular formatted files such as XSV, CSV, and TSV."""

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from typing_extensions import Self

from streaming.base.format.base.reader import FileInfo, SplitReader
from streaming.base.format.xsv.encodings import xsv_decode

__all__ = ['XSVReader', 'CSVReader', 'TSVReader']


class XSVReader(SplitReader):
    """Provides random access to the samples of an XSV shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        newline (str): Newline character(s).
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        separator (str): Separator character(s).
        size_limit (Union[int, str], optional): Optional shard size limit, after
            which point to start a new shard. If None, puts everything in one shard.
            Can specify bytes in human-readable format as well, for example
            ``"100kb"`` for 100 kilobyte (100*1024) and so on.
        zip_data (FileInfo, optional): Compressed data file info.
        zip_meta (FileInfo, optional): Compressed meta file info.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        column_encodings: List[str],
        column_names: List[str],
        compression: Optional[str],
        hashes: List[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        separator: str,
        size_limit: Optional[Union[int, str]],
        zip_data: Optional[FileInfo],
        zip_meta: Optional[FileInfo],
    ) -> None:
        super().__init__(dirname, split, compression, hashes, raw_data, raw_meta, samples,
                         size_limit, zip_data, zip_meta)
        self.column_encodings = column_encodings
        self.column_names = column_names
        self.newline = newline
        self.separator = separator

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Args:
            dirname (str): Local directory containing shards.
            split (str, optional): Which dataset split to use, if any.
            obj (Dict[str, Any]): JSON object to load.

        Returns:
            Self: Loaded XSVReader.
        """
        args = deepcopy(obj)
        if args['version'] != 2:
            raise ValueError(f'Unsupported streaming data version: {args["version"]}. ' +
                             f'Expected version 2.')
        del args['version']
        if args['format'] != 'xsv':
            raise ValueError(f'Unsupported data format: {args["format"]}. ' +
                             f'Expected to be `xsv`.')
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'raw_meta', 'zip_data', 'zip_meta']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)

    def decode_sample(self, data: bytes) -> Dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        text = data.decode('utf-8')
        text = text[:-len(self.newline)]
        parts = text.split(self.separator)
        sample = {}
        for name, encoding, part in zip(self.column_names, self.column_encodings, parts):
            sample[name] = xsv_decode(encoding, part)
        return sample

    def get_sample_data(self, idx: int) -> bytes:
        """Get the raw sample data at the index.

        Args:
            idx (int): Sample index.

        Returns:
            bytes: Sample data.
        """
        meta_filename = os.path.join(self.dirname, self.split, self.raw_meta.basename)
        offset = (1 + idx) * 4
        with open(meta_filename, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
        data_filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
        with open(data_filename, 'rb', 0) as fp:
            fp.seek(begin)
            data = fp.read(end - begin)
        return data


class CSVReader(XSVReader):
    """Provides random access to the samples of a CSV shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        newline (str): Newline character(s).
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
        zip_data (FileInfo, optional): Compressed data file info.
        zip_meta (FileInfo, optional): Compressed meta file info.
    """

    separator = ','

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        column_encodings: List[str],
        column_names: List[str],
        compression: Optional[str],
        hashes: List[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: Optional[FileInfo],
        zip_meta: Optional[FileInfo],
    ) -> None:
        super().__init__(dirname, split, column_encodings, column_names, compression, hashes,
                         newline, raw_data, raw_meta, samples, self.separator, size_limit,
                         zip_data, zip_meta)

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Args:
            dirname (str): Local directory containing shards.
            split (str, optional): Which dataset split to use, if any.
            obj (Dict[str, Any]): JSON object to load.

        Returns:
            Self: Loaded CSVReader.
        """
        args = deepcopy(obj)
        if args['version'] != 2:
            raise ValueError(f'Unsupported streaming data version: {args["version"]}. ' +
                             f'Expected version 2.')
        del args['version']
        if args['format'] != 'csv':
            raise ValueError(f'Unsupported data format: {args["format"]}. ' +
                             f'Expected to be `csv`.')
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'raw_meta', 'zip_data', 'zip_meta']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)


class TSVReader(XSVReader):
    """Provides random access to the samples of an XSV shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        newline (str): Newline character(s).
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
        zip_data (FileInfo, optional): Compressed data file info.
        zip_meta (FileInfo, optional): Compressed meta file info.
    """

    separator = '\t'

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        column_encodings: List[str],
        column_names: List[str],
        compression: Optional[str],
        hashes: List[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: Optional[FileInfo],
        zip_meta: Optional[FileInfo],
    ) -> None:
        super().__init__(dirname, split, column_encodings, column_names, compression, hashes,
                         newline, raw_data, raw_meta, samples, self.separator, size_limit,
                         zip_data, zip_meta)

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Args:
            dirname (str): Local directory containing shards.
            split (str, optional): Which dataset split to use, if any.
            obj (Dict[str, Any]): JSON object to load.

        Returns:
            Self: Loaded TSVReader.
        """
        args = deepcopy(obj)
        if args['version'] != 2:
            raise ValueError(f'Unsupported streaming data version: {args["version"]}. ' +
                             f'Expected version 2.')
        del args['version']
        if args['format'] != 'tsv':
            raise ValueError(f'Unsupported data format: {args["format"]}. ' +
                             f'Expected to be `tsv`.')
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'raw_meta', 'zip_data', 'zip_meta']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)
