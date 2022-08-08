from copy import deepcopy
import json
import numpy as np
import os
from typing import Any, Optional
from typing_extensions import Self

from ..base.reader import FileInfo, SplitReader


'''
    {
      "columns": {
        "number": "int",
        "words": "str"
      },
      "compression": "zstd:7",
      "format": "json",
      "hashes": [
        "sha1",
        "xxh3_64"
      ],
      "newline": "\n",
      "raw_data": {
        "basename": "shard.00000.json",
        "bytes": 1048546,
        "hashes": {
          "sha1": "bfb6509ba6f041726943ce529b36a1cb74e33957",
          "xxh3_64": "0eb102a981b299eb"
        }
      },
      "raw_meta": {
        "basename": "shard.00000.json.meta",
        "bytes": 53590,
        "hashes": {
          "sha1": "15ae80e002fe625b0b18f1a45058532ee867fa9b",
          "xxh3_64": "7b113f574a422ac1"
        }
      },
      "samples": 13352,
      "size_limit": 1048576,
      "version": 2,
      "zip_data": {
        "basename": "shard.00000.json.zstd",
        "bytes": 149268,
        "hashes": {
          "sha1": "7d45c600a71066ca8d43dbbaa2ffce50a91b735e",
          "xxh3_64": "3d338d4826d4b5ac"
        }
      },
      "zip_meta": {
        "basename": "shard.00000.json.meta.zstd",
        "bytes": 42180,
        "hashes": {
          "sha1": "f64477cca5d27fc3a0301eeb4452ef7310cbf670",
          "xxh3_64": "6e2b364f4e78670d"
        }
      }
    }
'''


class JSONReader(SplitReader):
    """Provides random access to the samples of a JSON shard.

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
        zip_data (FileInfo): Compressed data file info.
        zip_meta (FileInfo): Compressed meta file info.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        columns: dict[str, str],
        compression: Optional[str],
        hashes: list[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: FileInfo,
        zip_meta: FileInfo
    ) -> None:
        super().__init__(dirname, split, compression, hashes, raw_data, raw_meta, samples,
                         size_limit, zip_data, zip_meta)
        self.columns = columns
        self.newline = newline

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
        args = deepcopy(obj)
        assert args['version'] == 2
        del args['version']
        assert args['format'] == 'json'
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'raw_meta', 'zip_data', 'zip_meta']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)

    def _decode_sample(self, data: bytes) -> dict[str, Any]:
        text = data.decode('utf-8')
        return json.loads(text)

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
