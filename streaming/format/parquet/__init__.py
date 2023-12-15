# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Parquet shards."""

from streaming.format.parquet.indexing import index_parquet
from streaming.format.parquet.reader import ParquetShard

__all__ = ['index_parquet', 'ParquetShard']
