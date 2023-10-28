# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Individual dataset writer for every format."""

from typing import Any, Dict, Optional, Union

from streaming.format import FileInfo, Reader
from streaming.format.delta import index_delta
from streaming.format.index import get_index_basename
from streaming.format.json import JSONReader, JSONWriter
from streaming.format.lance import index_lance
from streaming.format.mds import MDSReader, MDSWriter
from streaming.format.parquet import index_parquet
from streaming.format.xsv import (CSVReader, CSVWriter, TSVReader, TSVWriter, XSVReader,
                                       XSVWriter)

__all__ = [
    'CSVWriter', 'FileInfo', 'JSONWriter', 'MDSWriter', 'Reader', 'TSVWriter', 'XSVWriter',
    'get_index_basename', 'index_backend', 'index_delta', 'index_lance', 'index_parquet',
    'reader_from_json'
]

_readers = {
    'csv': CSVReader,
    'json': JSONReader,
    'mds': MDSReader,
    'tsv': TSVReader,
    'xsv': XSVReader
}


def reader_from_json(dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Reader:
    """Initialize the reader from JSON object.

    Args:
        dirname (str): Local directory containing shards.
        split (str, optional): Which dataset split to use, if any.
        obj (Dict[str, Any]): JSON object to load.

    Returns:
        Reader: Loaded Reader of `format` type
    """
    assert obj['version'] == 2
    cls = _readers[obj['format']]
    return cls.from_json(dirname, split, obj)


def index_backend(backend: str,
                  local: str,
                  remote: Optional[str] = None,
                  split: Optional[str] = None,
                  version: Optional[int] = None,
                  num_procs: Optional[int] = 0,
                  download_timeout: Union[float, str] = '1m',
                  max_file_bytes: Optional[Union[int, str]] = '200mb',
                  same_schema: bool = True,
                  columns: Optional[Dict[str, Any]] = None,
                  show_progress: bool = True) -> Dict[str, Any]:
    """Index a local and/or remote third-party dataset directory for use by Streaming.

    Args:
        backend (str): What dataset/database system serves this entire dataset, whose files we
            convert, wrap, or both as Streaming shards. Must be one of ``delta`` (Delta table),
            ``lance`` (Lance dataset), or ``parquet`` (Parquet dataset) (if ``streaming``, the
            index is created at dataset write time).
        local (str): Where the dataset is cached on the local filesystem.
        remote (str, optional): Where the dataset is downloaded from. Defaults to ``None``.
        split (str, optional): Which dataset split to use. Defaults to ``None``.
        version (int, optional): Dataset snapshot version (used by ``delta`` and ``lance``
            datasets). If not provided, takes the latest version. Defaults to ``None``.
        num_procs (int, optional): Parallelism for downloading/processing of third-party dataset
            files. ``None`` means single-process. ``0`` means <number of CPUs> processes. Positive
            integer means use that number of processes. Defaults to ``0``.
        download_timeout (Union[float, str]): For each Parquet file. Defaults to ``2m``.
        max_file_bytes (Union[int, str], optional): File size limit, above which we raise an error.
            This is a performance guard rail, as choppiness increases linearly with shard size. The
            sweet spot is typically around 32mb. Defaults to ``200mb``.
        same_schema (bool): Whether to require that all the dataset shards have exactly the same
            MDS schema. Applicable to indexless Parquet datasets. This is a correctness guard rail,
            preventingh non-dataset shards from sneaking into our dataset. Streaming for its part
            is fine with shards being "incompatible"; assumes client will handle it. Defaults to
            ``True``.
        columns (Dict[str, str], optional): For field names and types specified here, override the
            inferred schema to configure it manually. Defaults to ``None``.
        show_progress (bool): Show progress bar for download/processing. Defaults to ``True``.

    Returns:
        Dict[str, Any]: StreamingDataset index configuration to stream this Parquet dataset.
    """
    if backend == 'delta':
        return index_delta(local=local,
                           remote=remote,
                           split=split,
                           version=version,
                           num_threads=num_procs,
                           download_timeout=download_timeout,
                           max_file_bytes=max_file_bytes,
                           columns=columns,
                           show_progress=show_progress)
    elif backend == 'lance':
        return index_lance(local=local,
                           remote=remote,
                           split=split,
                           version=version,
                           download_timeout=download_timeout,
                           max_file_bytes=max_file_bytes,
                           columns=columns)
    elif backend == 'parquet':
        return index_parquet(local=local,
                             remote=remote,
                             split=split,
                             num_procs=num_procs,
                             download_timeout=download_timeout,
                             max_file_bytes=max_file_bytes,
                             same_schema=same_schema,
                             columns=columns,
                             show_progress=show_progress)
    else:
        raise ValueError('Unsupported backend: {backend}.')
