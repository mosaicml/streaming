# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming shard abstract base classes."""

from typing import Optional

from streaming.format.base.file import ShardFile
from streaming.format.base.shard.base import WriterConf
from streaming.format.base.shard.row import RowShard
from streaming.format.base.stream_conf import StreamDirConf

__all__ = ['MonoRowShard']


class MonoRowShard(RowShard):
    """A RowShard that is stored as a single file.

    Args:
        writer_conf (WriterConf, optional): Keyword arguments used when writing this shard. This
            metadata is kept just for informational purposes. Defaults to ``None``.
        stream (StreamDirConf): Link back up to the Stream that owns this shard, from which
            we get arguments which are shared across all shards like remote/local paths. Optional
            to avoid a chicken and egg problem, but required by most methods. Defaults to ``None``.
        num_samples (int): Number of samples in this shard.
        file (ShardFile): The file containing shard data and metadata.
    """

    def __init__(
        self,
        *,
        writer_conf: Optional[WriterConf] = None,
        stream: StreamDirConf,
        num_samples: int,
        file: ShardFile,
    ) -> None:
        super().__init__(
            writer_conf=writer_conf,
            stream=stream,
            num_samples=num_samples,
        )
        self.files.append(file)
        self.file = file
