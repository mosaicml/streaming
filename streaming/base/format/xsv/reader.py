from copy import deepcopy
import json
import numpy as np
import os
from typing import Any, Optional
from typing_extensions import Self

from ..base.reader import FileInfo, SplitReader
from .encodings import xsv_decode

'''
    {
      "column_encodings": [
        "int",
        "str"
      ],
      "column_names": [
        "number",
        "words"
      ],
      "compression": "zstd:7",
      "format": "csv",
      "hashes": [
        "sha1",
        "xxh3_64"
      ],
      "newline": "\n",
      "raw_data": {
        "basename": "shard.00000.csv",
        "bytes": 1048523,
        "hashes": {
          "sha1": "39f6ea99d882d3652e34fe5bd4682454664efeda",
          "xxh3_64": "ea1572efa0207ff6"
        }
      },
      "raw_meta": {
        "basename": "shard.00000.csv.meta",
        "bytes": 77486,
        "hashes": {
          "sha1": "8874e88494214b45f807098dab9e55d59b6c4aec",
          "xxh3_64": "3b1837601382af2c"
        }
      },
      "samples": 19315,
      "separator": ",",
      "size_limit": 1048576,
      "version": 2,
      "zip_data": {
        "basename": "shard.00000.csv.zstd",
        "bytes": 197040,
        "hashes": {
          "sha1": "021d288a317ae0ecacba8a1b985ee107f966710d",
          "xxh3_64": "5daa4fd69d3578e4"
        }
      },
      "zip_meta": {
        "basename": "shard.00000.csv.meta.zstd",
        "bytes": 60981,
        "hashes": {
          "sha1": "f2a35f65279fbc45e8996fa599b25290608990b2",
          "xxh3_64": "7c38dee2b3980deb"
        }
      }
    }
'''


class XSVReader(SplitReader):
    """Provides random access to the samples of an XSV shard.

    Args:
        dirname (str): Local dataset directory.
        split (Optional[str]): Which dataset split to use, if any.
        column_encodings (list[str]): Column encodings.
        column_names (list[str]): Column names.
        compression (Optional[str]): Optional compression or compression:level.
        hashes (list[str]): Optional list of hash algorithms to apply to shard files.
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
        column_encodings: list[str],
        column_names: list[str],
        compression: Optional[str],
        hashes: list[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        separator: str,
        size_limit: Optional[int],
        zip_data: Optional[FileInfo],
        zip_meta: Optional[FileInfo]
    ) -> None:
        super().__init__(dirname, split, compression, hashes, raw_data, raw_meta, samples,
                         size_limit, zip_data, zip_meta)
        self.column_encodings = column_encodings
        self.column_names = column_names
        self.newline = newline
        self.separator = separator

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
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

    def _decode_sample(self, data: bytes) -> dict[str, Any]:
        text = data.decode('utf-8')
        text = text[:-len(self.newline)]
        parts = text.split(self.separator)
        sample = {}
        for name, encoding, part in zip(self.column_names, self.column_encodings, parts):
            sample[name] = xsv_decode(encoding, part)
        return sample

    def _get_sample_data(self, idx: int) -> bytes:
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
        column_encodings (list[str]): Column encodings.
        column_names (list[str]): Column names.
        compression (Optional[str]): Optional compression or compression:level.
        hashes (list[str]): Optional list of hash algorithms to apply to shard files.
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
        column_encodings: list[str],
        column_names: list[str],
        compression: Optional[str],
        hashes: list[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: Optional[FileInfo],
        zip_meta: Optional[FileInfo]
    ) -> None:
        super().__init__(dirname, split, column_encodings, column_names, compression, hashes,
                         newline, raw_data, raw_meta, samples, self.separator, size_limit, zip_data,
                         zip_meta)

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
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
        column_encodings (list[str]): Column encodings.
        column_names (list[str]): Column names.
        compression (Optional[str]): Optional compression or compression:level.
        hashes (list[str]): Optional list of hash algorithms to apply to shard files.
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
        column_encodings: list[str],
        column_names: list[str],
        compression: Optional[str],
        hashes: list[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: Optional[FileInfo],
        zip_meta: Optional[FileInfo]
    ) -> None:
        super().__init__(dirname, split, column_encodings, column_names, compression, hashes,
                         newline, raw_data, raw_meta, samples, self.separator, size_limit, zip_data,
                         zip_meta)

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
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
