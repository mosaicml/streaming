# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming serialization format, consisting of an index and multiple types of shards."""

from typing import Any, Dict

from streaming.format.index import get_index_basename
from streaming.format.jsonl import JSONLShard, JSONLWriter
from streaming.format.mds import MDSShard, MDSWriter
from streaming.format.shard import Shard
from streaming.format.xsv import CSVShard, CSVWriter, TSVShard, TSVWriter, XSVShard, XSVWriter
from streaming.stream.dir_conf import StreamDirConf

__all__ = [
    'CSVWriter', 'get_index_basename', 'JSONLWriter', 'MDSWriter', 'shard_from_json', 'TSVWriter',
    'XSVWriter', 'Shard'
]

# Mapping of shard metadata dict "format" field to what type of Shard it is.
#
# Pedantic historical note: JSONL shards were originally called JSON shards, because while they
# were JSONL in practice, the line delimiter was/is customizable.
shard_formats = {
    'csv': CSVShard,
    'json': JSONLShard,
    'jsonl': JSONLShard,
    'mds': MDSShard,
    'tsv': TSVShard,
    'xsv': XSVShard,
}


def shard_from_json(stream: StreamDirConf, obj: Dict[str, Any]) -> Shard:
    """Create a shard from a JSON config.

    Args:
        stream (StreamDirConf): The Stream that we are a part of.
        obj (Dict[str, Any]): JSON object to load.

    Returns:
        Shard: The loaded Shard.
    """
    fmt = obj.get('format')
    if not fmt:
        raise ValueError(f'Shard JSON config is missing format field.')

    cls = shard_formats.get(fmt)
    if not cls:
        raise ValueError(f'Got shard format {fmt}, but our supported formats are ' +
                         f'{sorted(shard_formats)}.')

    return cls.from_json(stream, obj)
