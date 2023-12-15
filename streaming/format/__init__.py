# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming serialization format, consisting of an index and multiple types of shards."""

from typing import Any, Dict, Optional

from streaming.format.index import get_index_basename
from streaming.format.json import JSONReader, JSONWriter
from streaming.format.mds import MDSReader, MDSWriter
from streaming.format.reader import FileInfo, Reader
from streaming.format.xsv import CSVReader, CSVWriter, TSVReader, TSVWriter, XSVReader, XSVWriter

__all__ = [
    'CSVWriter', 'FileInfo', 'get_index_basename', 'JSONWriter', 'MDSWriter', 'Reader',
    'reader_from_json', 'TSVWriter', 'XSVWriter'
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
