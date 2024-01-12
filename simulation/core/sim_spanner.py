# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Mapping of global sample index to shard index for simulation purposes."""

from typing import Tuple

from streaming.base.spanner import Spanner


class SimulationSpanner(Spanner):
    """Given a list of shards, construct a mapping of global index to shard index.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples in each shard.
        span_size (int): Size of the divisions of the sample space. Defaults to ``1 << 10``.
    """

    def __getitem__(self, index: int) -> Tuple[int, int]:
        """Map global sample index to shard index only.

        Args:
            index (int): Global sample index.

        Returns:
            int: Shard index of sample.
        """
        if not (0 <= index < self.num_samples):
            raise ValueError(f'Invalid sample index `{index}`: 0 <= {index} < {self.num_samples}')

        span = index // self.span_size
        for shard in self.spans[span]:
            shard_start = self.shard_bounds[shard]
            shard_stop = self.shard_bounds[shard + 1]
            if shard_start <= index < shard_stop:
                return shard

        raise RuntimeError('Internal error: shards were indexed incorrectly')
