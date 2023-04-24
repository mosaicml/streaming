# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Mapping of global index to shard and relative index."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class Spanner:
    """Given a list of shards, construct a mapping of global index to shard and relative index.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples in each shard.
        span_size (int): Size of the divisions of the sample space. Defaults to ``1 << 10``.
    """

    def __init__(self, shard_sizes: NDArray[np.int64], span_size: int = 1 << 10) -> None:
        self.shard_sizes = shard_sizes
        self.span_size = span_size

        self.size = size = sum(shard_sizes)
        self.shard_stops = shard_sizes.cumsum()
        self.shard_starts = self.shard_stops - shard_sizes

        overflow = size % span_size
        if overflow:
            self.shard_sizes[-1] += span_size - overflow

        sample_shards = np.repeat(np.arange(len(shard_sizes)), self.shard_sizes)
        sample_shards = sample_shards.reshape(-1, span_size)
        span_lowest_shards = sample_shards.min(1)
        span_highest_shards = sample_shards.max(1)

        self.spans = []
        for low, high in zip(span_lowest_shards, span_highest_shards):
            shards = np.arange(low, high + 1)
            self.spans.append(shards)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        """Map global index to shard and relative index.

        Args:
            index (int): Global index.

        Returns:
            Tuple[int, int]: Span and relative index.
        """
        if not (0 <= index < self.size):
            raise ValueError(f'Invalid index: 0 <= {index} < {self.size}')

        span = index // self.span_size
        for shard in self.spans[span]:
            shard_start = self.shard_starts[shard]
            shard_stop = self.shard_stops[shard]
            if shard_start <= index < shard_stop:
                return shard, index - shard_start

        raise RuntimeError('Internal error: shards were indexed incorrectly')
