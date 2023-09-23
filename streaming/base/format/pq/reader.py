# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""PQReader reads Parquet shards for StreamingDataset (via MDS internally)."""

import os
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Set

from pyarrow import parquet as pq
from typing_extensions import Self

from streaming.base.format.base.reader import FileInfo
from streaming.base.format.mds.reader import MDSReader
from streaming.base.format.mds.writer import MDSWriter


class PQReader(MDSReader):
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
            Self: Loaded PQReader.
        """
        args = deepcopy(obj)

        if args['version'] != 2:
            raise ValueError(f'Unsupported streaming data version: {args["version"]}. ' +
                             f'Expected version 2.')
        del args['version']

        if args['format'] != 'pq':
            raise ValueError(f'Unsupported data format: {args["format"]}. ' +
                             f'Expected to be `pq`.')
        del args['format']

        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_parquet', 'raw_data', 'zip_data']:
            arg = args.get(key)
            if arg:
                args[key] = FileInfo(**arg)

        return cls(**args)

    def set_up_local(self, listing: Set[str], safe_keep_zip: bool) -> int:
        """Bring what shard files are present to a consistent state, returning whether present.

        Args:
            listing (Set[str]): The listing of all files under dirname/[split/]. This is listed
                once and then saved because there could potentially be very many shard files.
            safe_keep_zip (bool): Whether to keep zip files when decompressing. Possible when
                compression was used. Necessary when local is the remote or there is no remote.

        Returns:
            int: Shard cache usage.
        """
        pq_filename = os.path.join(self.dirname, self.split, self.raw_parquet.basename)
        mds_filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
        if os.path.exists(mds_filename):
            if os.path.exists(pq_filename):
                if safe_keep_zip:
                    # Present: keep both (because of safe_keep_zip).
                    usage = os.stat(mds_filename).st_size + os.stat(pq_filename).st_size
                else:
                    # Present: keep MDS, drop PQ (because of safe_keep_zip).
                    os.remove(pq_filename)
                    usage = os.stat(mds_filename).st_size
            else:
                if safe_keep_zip:
                    # Normalize to missing, because safe_keep_zip requires that we keep the PQ.
                    os.remove(mds_filename)
                    usage = 0
                else:
                    # Present: have MDS, don't have or want PQ.
                    usage = os.stat(mds_filename).st_size
        else:
            if os.path.exists(pq_filename):
                # Present: PQ hasn't been converted to MDS yet and we don't have time to here.
                usage = os.stat(pq_filename).st_size
            else:
                # Missing: both PQ and MDS are not there.
                usage = 0
        return usage

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

    def prepare(self, safe_keep_zip: bool) -> int:
        """Prepare a Parquet shard for fast random access by converting to MDS.

        Args:
            safe_keep_zip (bool): Whether to keep Parquet shards, or drop post-conversion.

        Returns:
            int: Change in cache usage in bytes due to PQ -> MDS conversion.
        """
        pq_filename = os.path.join(self.dirname, self.split, self.raw_parquet.basename)
        table = pq.read_table(pq_filename)
        samples = table.to_pylist()
        columns = self.get_columns(samples[0])
        with TemporaryDirectory() as temp_dir:
            with MDSWriter(columns=columns, out=temp_dir, size_limit=None) as out:
                for sample in samples:
                    out.write(sample)
            temp_mds_filename = os.path.join(temp_dir, 'shard.00000.mds')
            mds_filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
            os.rename(temp_mds_filename, mds_filename)
            delta = os.stat(mds_filename).st_size
        if not safe_keep_zip:
            delta -= os.stat(pq_filename).st_size
            os.remove(pq_filename)
        return delta
