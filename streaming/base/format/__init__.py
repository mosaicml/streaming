# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Individual dataset writer for every format."""

from typing import Any, Dict, Optional

from streaming.base.format.base import FileInfo, Reader
from streaming.base.format.delta import index_delta
from streaming.base.format.index import get_index_basename
from streaming.base.format.json import JSONReader, JSONWriter
from streaming.base.format.lance import index_lance
from streaming.base.format.mds import MDSReader, MDSWriter
from streaming.base.format.parquet import index_parquet
from streaming.base.format.xsv import (CSVReader, CSVWriter, TSVReader, TSVWriter, XSVReader,
                                       XSVWriter)

__all__ = [
    'CSVWriter', 'FileInfo', 'JSONWriter', 'MDSWriter', 'Reader', 'TSVWriter', 'XSVWriter',
    'get_index_basename', 'index_delta', 'index_lance', 'index_parquet', 'reader_from_json'
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
