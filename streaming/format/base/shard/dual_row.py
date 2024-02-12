# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming shard abstract base classes."""

from typing import Any, Optional

from streaming.format.base.file import ShardFile
from streaming.format.base.shard.row import RowShard
from streaming.stream.dir_conf import StreamDirConf

__all__ = ['DualRowShard']


class DualRowShard(RowShard):
    """A RowShard that is stored as a pair of data and metadata files.

    Args:
        conf (Any, optional): JSON shard config. Defaults to ``None``.
        stream (StreamDirConf): Link back up to the Stream that owns this shard, from which
            we get arguments which are shared across all shards like remote/local paths. Optional
            to avoid a chicken and egg problem, but required by most methods. Defaults to ``None``.
        num_samples (int): Number of samples in this shard.
        data_file (ShardFile): The file containing shard data.
        meta_file (ShardFile): The file containing shard metadata.
    """

    def __init__(
        self,
        *,
        conf: Optional[Any] = None,
        stream: StreamDirConf,
        num_samples: int,
        data_file: ShardFile,
        meta_file: ShardFile,
    ) -> None:
        super().__init__(
            conf=conf,
            stream=stream,
            num_samples=num_samples,
            files=[data_file, meta_file],
        )
        self.data_file = data_file
        self.meta_file = meta_file