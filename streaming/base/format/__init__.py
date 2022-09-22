# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Individual dataset writer for every format."""

from typing import Any, Dict, Optional

from streaming.base.format.base.reader import Reader
from streaming.base.format.json import JSONReader, JSONWriter
from streaming.base.format.mds import MDSReader, MDSWriter
from streaming.base.format.xsv import (CSVReader, CSVWriter, TSVReader, TSVWriter, XSVReader,
                                       XSVWriter)

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


__all__ = ['CSVWriter', 'JSONWriter', 'MDSWriter', 'reader_from_json', 'TSVWriter', 'XSVWriter']
