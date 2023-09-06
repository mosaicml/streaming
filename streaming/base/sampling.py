# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Functionality relating to sampling."""

import numpy as np
from numpy.typing import NDArray


def get_sampling(samples_per_shard: NDArray[np.int64], choose: int, granularity: int, seed: int,
                 epoch: int, use_epoch: bool) -> NDArray[np.int64]:
    """Get how many samples to draw from each shard of the given stream.

    Args:
        samples_per_shard (NDArray[np.int64]): Array of underlying shard sizes.
        choose (int): How many samples to draw in total over all shards.
        granularity (int): How many samples to draw at a time from the same shard.
        seed (int): Seed for shuffling sampling granules.
        epoch (int): Which epoch we are sampling for.
        use_epoch (bool): Whether to factor epoch into the base seed, or use the same seed across
            epochs.

    Returns:
        NDArray[np.int64]: Array of ephemeral samples chosen per shard.
    """
    # Handle whole integer repeat case.
    num_samples = sum(samples_per_shard)
    if not choose % num_samples:
        return samples_per_shard * choose // num_samples

    # Fractional repeat case.

    # Get the ordering by which we will exhaust the shards.
    pairs = []  # List of (shard ID, samples to draw).
    for shard_id, shard_samples in enumerate(samples_per_shard):
        num_granules = (shard_samples + granularity - 1) // granularity
        shard_ids = np.full(num_granules, shard_id)
        counts = np.full(num_granules, granularity)
        counts[-1] = shard_samples % granularity
        pair = shard_ids, counts
        pairs.append(pair)
    shard_ids, counts = zip(*pairs)
    shard_ids = np.concatenate(shard_ids)
    counts = np.concatenate(counts)
    num_granules = len(shard_ids)
    epoch_seed = seed + epoch if use_epoch else seed
    rng = np.random.default_rng(epoch_seed)
    ordering = rng.permutation(num_granules)

    # Collect choose per shard.
    choose_per_shard = samples_per_shard * (choose // num_samples)
    choose %= num_samples
    for index in ordering:
        shard_id = shard_ids[index]
        count = counts[index]
        count = min(choose, int(count))
        choose_per_shard[shard_id] += count
        choose -= count
        if not choose:
            break

    return choose_per_shard
