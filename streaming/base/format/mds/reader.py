from copy import deepcopy
import numpy as np
import os
from typing import Any, Optional
from typing_extensions import Self

from ..base.reader import FileInfo, JointReader
from .encodings import mds_decode


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
      "column_sizes": [
        8,
        null
      ],
      "compression": "zstd:7",
      "format": "mds",
      "hashes": [
        "sha1",
        "xxh3_64"
      ],
      "raw_data": {
        "basename": "shard.00000.mds",
        "bytes": 1048544,
        "hashes": {
          "sha1": "8d0634d3836110b00ae435bbbabd1739f3bbeeac",
          "xxh3_64": "2c54988514bca807"
        }
      },
      "samples": 16621,
      "size_limit": 1048576,
      "version": 2,
      "zip_data": {
        "basename": "shard.00000.mds.zstd",
        "bytes": 228795,
        "hashes": {
          "sha1": "2fb5ece19aabc91c2d6d6c126d614ab291abe24a",
          "xxh3_64": "fe6e78d7c73d9e79"
        }
      }
    }
'''


class MDSReader(JointReader):
    """Provides random access to the samples of an MDS shard.

    Args:
        dirname (str): Local dataset directory.
        split (Optional[str]): Which dataset split to use, if any.
        column_encodings (list[str]): Column encodings.
        column_names (list[str]): Column names.
        column_sizes (list[Optional[int]]): Column fixed sizes, if any.
        compression (Optional[str]): Optional compression or compression:level.
        hashes (list[str]): Optional list of hash algorithms to apply to shard files.
        raw_data (FileInfo): Uncompressed data file info.
        samples (int): Number of samples in this shard.
        size_limit (Optional[int]): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
        zip_data (Optional[FileInfo]): Compressed data file info.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        column_encodings: list[str],
        column_names: list[str],
        column_sizes: list[Optional[int]],
        compression: Optional[str],
        hashes: list[str],
        raw_data: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: Optional[FileInfo]
    ) -> None:
        super().__init__(dirname, split, compression, hashes, raw_data, samples, size_limit,
                         zip_data)
        self.column_encodings = column_encodings
        self.column_names = column_names
        self.column_sizes = column_sizes

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
        args = deepcopy(obj)
        assert args['version'] == 2
        del args['version']
        assert args['format'] == 'mds'
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'zip_data']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)

    def _decode_sample(self, data: bytes) -> dict[str, Any]:
        sizes = []
        idx = 0
        for key, size in zip(self.column_names, self.column_sizes):
            if size:
                sizes.append(size)
            else:
                size, = np.frombuffer(data[idx:idx + 4], np.uint32)
                sizes.append(size)
                idx += 4
        sample = {}
        for key, encoding, size in zip(self.column_names, self.column_encodings, sizes):
            value = data[idx:idx + size]
            sample[key] = mds_decode(encoding, value)
            idx += size
        return sample

    def _get_sample_data(self, idx: int) -> bytes:
        filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
        offset = (1 + idx) * 4
        with open(filename, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
            fp.seek(begin)
            data = fp.read(end - begin)
        return data
