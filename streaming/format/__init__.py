# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming serialization format, consisting of an index and multiple types of shards."""

from typing import Any, Dict, Optional

from streaming.format.index import get_index_basename
from streaming.format.json import JSONShard, JSONWriter
from streaming.format.mds import MDSShard, MDSWriter
from streaming.format.shard import FileInfo, Shard
from streaming.format.xsv import CSVShard, CSVWriter, TSVShard, TSVWriter, XSVShard, XSVWriter

__all__ = [
    'CSVWriter', 'FileInfo', 'get_index_basename', 'JSONWriter', 'MDSWriter', 'Shard',
    'shard_from_json', 'TSVWriter', 'XSVWriter'
]

_shards = {
    'csv': CSVShard,
    'json': JSONShard,
    'mds': MDSShard,
    'tsv': TSVShard,
    'xsv': XSVShard,
}


def shard_from_json(dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Shard:
    """Create a shard from a JSON config.

    Args:
        dirname (str): Local directory containing shards.
        split (str, optional): Which dataset split to use, if any.
        obj (Dict[str, Any]): JSON object to load.

    Returns:
        Shard: The loaded Shard.
    """
    assert obj['version'] == 2
    cls = _shards[obj['format']]
    return cls.from_json(dirname, split, obj)
