# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Collection of dataset base classes all using the same streaming shard serialization format."""

from streaming.base.dataset.local_iterable import LocalIterableDataset
from streaming.base.dataset.local_map import LocalMapDataset
from streaming.base.dataset.streaming import StreamingDataset

__all__ = ['LocalIterableDataset', 'LocalMapDataset', 'StreamingDataset']
