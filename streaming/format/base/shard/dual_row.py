# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming shard abstract base classes."""

from typing import Optional

from streaming.format.base.file import ShardFile
from streaming.format.base.shard.base import WriterConf
from streaming.format.base.shard.row import RowShard
from streaming.format.base.stream_conf import StreamConf

__all__ = ['DualRowShard']


class DualRowShard(RowShard):
    """A RowShard that is stored as a pair of data and metadata files.

    Args:
        writer_conf (WriterConf, optional): Keyword arguments used when writing this shard.
            This metadata is kept just for informational purposes. Defaults to ``None``.
        stream (StreamConf): Link back up to the Stream that owns this shard, from which
            we get arguments which are shared across all shards like remote/local paths. Optional
            to avoid a chicken and egg problem, but required by most methods. Defaults to ``None``.
        num_samples (int): Number of samples in this shard.
        data_file (ShardFile): The file containing shard data.
        meta_file (ShardFile): The file containing shard metadata.
    """

    def __init__(
        self,
        *,
        writer_conf: Optional[WriterConf] = None,
        stream: StreamConf,
        num_samples: int,
        data_file: ShardFile,
        meta_file: ShardFile,
    ) -> None:
        super().__init__(
            writer_conf=writer_conf,
            stream=stream,
            num_samples=num_samples,
        )
        self.files += [data_file, meta_file]
        self.data_file = data_file
        self.meta_file = meta_file
