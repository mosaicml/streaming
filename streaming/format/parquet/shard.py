# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Parquet shard reading."""

import os
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Set

from pyarrow import parquet as pq
from typing_extensions import Self

from streaming.format.mds.shard import MDSShard
from streaming.format.mds.writer import MDSWriter
from streaming.format.shard import FileInfo


# TODO: This approach is close, but wrong.
class ParquetShard(MDSShard):
    """Provides random access to the samples of a Parquet shard (via MDS internally).

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        column_sizes (List[Optional[int]]): Column fixed sizes, if any.
        raq_parquet (FileInfo): Non-compressed Parquet file info.
        raw_data (FileInfo): Uncompressed data file info.
        samples (int): Number of samples in this shard.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        column_encodings: List[str],
        column_names: List[str],
        column_sizes: List[Optional[int]],
        raw_parquet: FileInfo,
        raw_data: FileInfo,
        samples: int,
    ) -> None:
        super().__init__(dirname=dirname,
                         split=split,
                         column_encodings=column_encodings,
                         column_names=column_names,
                         column_sizes=column_sizes,
                         compression=None,
                         hashes=[],
                         raw_data=raw_data,
                         samples=samples,
                         size_limit=None,
                         zip_data=None)
        self.raw_parquet = raw_parquet
        self.file_pairs.append((raw_parquet, None))

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Args:
            dirname (str): Local directory containing shards.
            split (str, optional): Which dataset split to use, if any.
            obj (Dict[str, Any]): JSON object to load.

        Returns:
            Self: Loaded ParquetShard.
        """
        args = deepcopy(obj)

        if args['version'] != 2:
            raise ValueError(f'Unsupported streaming data version: {args["version"]}. ' +
                             f'Expected version 2.')
        del args['version']

        if args['format'] != 'parquet':
            raise ValueError(f'Unsupported data format: {args["format"]}. ' +
                             f'Expected to be `parquet`.')
        del args['format']

        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_parquet', 'raw_data', 'zip_data']:
            arg = args.get(key)
            if arg:
                args[key] = FileInfo(**arg)

        return cls(**args)

    def set_up_local(self, listing: Set[str], safe_keep_zip: bool, safe_keep_parquet: bool) -> int:
        """Bring what shard files are present to a consistent state, returning whether present.

        Args:
            listing (Set[str]): The listing of all files under dirname/[split/]. This is listed
                once and then saved because there could potentially be very many shard files.
            safe_keep_zip (bool): Whether to keep or drop the zip form after decompression, if
                applicable, safely taking into account whether this directory is the official copy.
            safe_keep_parquet (bool): Whether to keep or drop the Parquet form after MDS
                conversion, if applicable, safely taking into account whether this directory is the
                official copy.

        Returns:
            int: This shard's current contribution to cache usage after normalization.
        """
        parquet_filename = os.path.join(self.dirname, self.split, self.raw_parquet.basename)
        mds_filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
        if os.path.exists(mds_filename):
            if os.path.exists(parquet_filename):
                if safe_keep_parquet:
                    # Present: keep both (because of safe_keep_parquet).
                    size = os.stat(mds_filename).st_size + os.stat(parquet_filename).st_size
                else:
                    # Present: keep MDS, drop Parquet (because of saftfe_keep_parquet).
                    os.remove(parquet_filename)
                    size = os.stat(mds_filename).st_size
            else:
                if safe_keep_parquet:
                    # Normalize to missing, because safe_keep_parquet requires that we keep the
                    # Parquet.
                    os.remove(mds_filename)
                    size = 0
                else:
                    # Present: have MDS, don't have or want Parquet.
                    size = os.stat(mds_filename).st_size
        else:
            if os.path.exists(parquet_filename):
                # Present: Parquet hasn't been converted to MDS yet and we don't have time to here.
                size = os.stat(parquet_filename).st_size
            else:
                # Missing: both Parquet and MDS are not there.
                size = 0
        return size

    def get_column(self, val: Any) -> str:
        """Get the MDS column encoding of one field.

        Args:
            val (Any): The field.

        Returns:
            str: Its corresponding MDS encoding.
        """
        if isinstance(val, int):
            return 'int'
        elif isinstance(val, str):
            return 'str'
        else:
            raise ValueError('Unsupported column type: {type(val)}.')

    def get_columns(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Get the MDS columns given one sample.

        Args:
            sample (Dict[str, Any]): Mapping of column name to value.

        Returns:
            Dict[str, str]: Mapping of column name to MDS encoding.
        """
        col_names = sorted(sample)
        col_encs = []
        for name in col_names:
            val = sample[name]
            enc = self.get_column(val)
            col_encs.append(enc)
        return dict(zip(col_names, col_encs))

    def prepare(self, safe_keep_zip: bool, safe_keep_parquet: bool) -> int:
        """Prepare this shard for fast random access by converting to MDS.

        Args:
            safe_keep_zip (bool): Whether to keep or drop the zip form after decompression, if
                applicable, safely taking into account whether this directory is the official copy.
            safe_keep_parquet (bool): Whether to keep or drop the Parquet form after MDS
                conversion, if applicable, safely taking into account whether this directory is the
                official copy.

        Returns:
            int: Change in cache usage in bytes resulting from Parquet to MDS conversion.
        """
        # Read the samples from Parquet.
        parquet_filename = os.path.join(self.dirname, self.split, self.raw_parquet.basename)
        table = pq.read_table(parquet_filename)
        samples = table.to_pylist()

        # Write the samples to MDS.
        columns = dict(zip(self.column_names, self.column_encodings))
        with TemporaryDirectory() as temp_dir:
            with MDSWriter(columns=columns, out=temp_dir, size_limit=None) as out:
                for sample in samples:
                    out.write(sample)
            temp_mds_filename = os.path.join(temp_dir, 'shard.00000.mds')
            mds_filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
            os.rename(temp_mds_filename, mds_filename)
        delta = os.stat(mds_filename).st_size

        # Maybe drop the original Parquet.
        if not safe_keep_parquet:
            os.remove(parquet_filename)
            delta -= os.stat(parquet_filename).st_size

        return delta
