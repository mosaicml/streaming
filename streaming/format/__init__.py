# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming serialization format, consisting of an index and multiple types of shards."""

from typing import Any, Dict, Optional

from streaming.format.index import get_index_basename
from streaming.format.jsonl import JSONLShard, JSONLWriter
from streaming.format.mds import MDSShard, MDSWriter
from streaming.format.shard import FileInfo, Shard
from streaming.format.xsv import CSVShard, CSVWriter, TSVShard, TSVWriter, XSVShard, XSVWriter

__all__ = [
    'CSVWriter', 'FileInfo', 'get_index_basename', 'JSONLWriter', 'MDSWriter', 'Shard',
    'shard_from_json', 'TSVWriter', 'XSVWriter'
]

# Mapping of shard metadata dict "format" field to what type of Shard it is.
_shards = {
    'csv': CSVShard,
    'jsonl': JSONLShard,
    'mds': MDSShard,
    'tsv': TSVShard,
    'xsv': XSVShard,
}


def _get_shard_class(format_name: str) -> Shard:
    """Get the associated Shard class given a Shard format name.

    Args:
        format_name (str): Shard format name.
    """
    # JSONL shards were originally called JSON shards (while containing JSONL).
    if format_name == 'json':
        format_name = 'jsonl'
    return _shards[format_name]


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
    cls = _get_shard_class(obj['format'])
    return cls.from_json(dirname, split, obj)
