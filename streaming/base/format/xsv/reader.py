# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
from typing_extensions import Self

from streaming.base.format.base.reader import FileInfo, SplitReader
from streaming.base.format.xsv.encodings import xsv_decode


class XSVReader(SplitReader):
    """Provides random access to the samples of an XSV shard.

    Args:
        dirname (str): Local dataset directory.
        split (Optional[str]): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        compression (Optional[str]): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        newline (str): Newline character(s).
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        separator (str): Separator character(s).
        size_limit (Optional[int]): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
        zip_data (Optional[FileInfo]): Compressed data file info.
        zip_meta (Optional[FileInfo]): Compressed meta file info.
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
        size_limit: Optional[int],
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
        args = deepcopy(obj)
        assert args['version'] == 2
        del args['version']
        assert args['format'] == 'xsv'
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'raw_meta', 'zip_data', 'zip_meta']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)

    def decode_sample(self, data: bytes) -> Dict[str, Any]:
        text = data.decode('utf-8')
        text = text[:-len(self.newline)]
        parts = text.split(self.separator)
        sample = {}
        for name, encoding, part in zip(self.column_names, self.column_encodings, parts):
            sample[name] = xsv_decode(encoding, part)
        return sample

    def get_sample_data(self, idx: int) -> bytes:
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
        split (Optional[str]): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        compression (Optional[str]): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        newline (str): Newline character(s).
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        size_limit (Optional[int]): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
        zip_data (Optional[FileInfo]): Compressed data file info.
        zip_meta (Optional[FileInfo]): Compressed meta file info.
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
                         newline, raw_data, raw_meta, samples, self.separator, size_limit, zip_data,
                         zip_meta)

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Self:
        args = deepcopy(obj)
        assert args['version'] == 2
        del args['version']
        assert args['format'] == 'csv'
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
        split (Optional[str]): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        compression (Optional[str]): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        newline (str): Newline character(s).
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        size_limit (Optional[int]): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
        zip_data (Optional[FileInfo]): Compressed data file info.
        zip_meta (Optional[FileInfo]): Compressed meta file info.
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
                         newline, raw_data, raw_meta, samples, self.separator, size_limit, zip_data,
                         zip_meta)

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Self:
        args = deepcopy(obj)
        assert args['version'] == 2
        del args['version']
        assert args['format'] == 'tsv'
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'raw_meta', 'zip_data', 'zip_meta']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)
