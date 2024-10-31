# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shuffling algorithm that naively shuffles all-to-all.

Useful for single-node training on small data, where you want the most random shuffle possible.

Statistically, this algorithm will result in all nodes downloading all shards, with those downloads
all happening at the start of the epoch, bringing training to a crawl.
"""

import numpy as np
from numpy.typing import NDArray


def get_shuffle_naive(shard_sizes: NDArray[np.int64],
                      num_canonical_nodes: int,
                      seed: int,
                      epoch: int,
                      block_size: int = 1 << 18) -> NDArray[np.int64]:
    """Get the shuffled global ordering of samples for an epoch.

    The assignment of shards to nodes is fixed across epochs, but each grouping
    of shards is processed concurrently in a different order by each node's
    workers each epoch.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples contained in each shard, in order.
        num_canonical_nodes (int): Number of canonical nodes.
        seed (int): Base random seed, which is held constant over an entire training run.
        epoch (int): Current epoch, which is added to the seed to get a different deterministic
            shuffle each epoch.
        block_size (int): Unit of shuffle (ignored because we shuffle all samples together).
            Defaults to ``1 << 18``.

    Returns:
        NDArray[np.int64]: 1:1 mapping of sample ID to shuffled sample ID.
    """
    rng = np.random.default_rng(seed + epoch)
    return rng.permutation(sum(shard_sizes))
