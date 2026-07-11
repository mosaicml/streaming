# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Mapping of global sample index to shard and relative sample index."""

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
        self.num_samples = sum(shard_sizes)
        self.shard_bounds = np.concatenate([np.zeros(1, np.int64), shard_sizes.cumsum()])

        overflow = self.num_samples % span_size
        underflow = span_size - overflow if overflow else 0
        self.shard_sizes[-1] += underflow

        n_shards = len(shard_sizes)
        current_shard = 0
        current_position_in_shard = 0

        span_lowest_shards = []
        span_highest_shards = []

        while current_shard < n_shards:
            span_min_shard = current_shard
            span_max_shard = current_shard

            remaining_span_size = span_size
            while remaining_span_size > 0 and current_shard < n_shards:
                available_in_current_shard = shard_sizes[current_shard] - current_position_in_shard

                if remaining_span_size >= available_in_current_shard:
                    remaining_span_size -= available_in_current_shard
                    current_shard += 1
                    current_position_in_shard = 0
                else:
                    current_position_in_shard += remaining_span_size
                    remaining_span_size = 0

                if current_shard < n_shards:
                    span_max_shard = current_shard

            span_lowest_shards.append(span_min_shard)
            span_highest_shards.append(span_max_shard)

        self.spans = []
        for low, high in zip(span_lowest_shards, span_highest_shards):
            shards = np.arange(low, high + 1)
            self.spans.append(shards)

        self.shard_sizes[-1] -= underflow

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Map global sample index to shard and relative sample index.

        Args:
            index (int): Global sample index.

        Returns:
            Tuple[int, int]: Shard and relative sample index.
        """
        if not (0 <= index < self.num_samples):
            raise IndexError(f'Invalid sample index `{index}`: 0 <= {index} < {self.num_samples}')

        span = index // self.span_size
        for shard in self.spans[span]:
            shard_start = self.shard_bounds[shard]
            shard_stop = self.shard_bounds[shard + 1]
            if shard_start <= index < shard_stop:
                return shard, int(index - shard_start.item())  # pyright: ignore

        raise RuntimeError('Internal error: shards were indexed incorrectly')
