# Copyright 2022-2024 MosaicML Streaming authors
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
    if choose < 0:
        raise ValueError(f'`choose` must be a non-negative integer, but got: {choose}.')

    if granularity <= 0:
        raise ValueError(f'`granularity` must be a positive integer, but got: {granularity}.')

    if seed < 0:
        raise ValueError(f'`seed` must be a non-negative integer, but got: {seed}.')

    if epoch < 0:
        raise ValueError('`epoch` must be a non-negative integer, but got: {epoch}.')

    # Handle whole integer repeat case.
    num_samples = sum(samples_per_shard)
    if not choose % num_samples:
        return samples_per_shard * choose // num_samples

    # Fractional repeat case:

    # Get the number of shards.
    num_shards = len(samples_per_shard)

    # Get how many times you have to pick each shard to draw all its samples.
    picks_per_shard = (samples_per_shard + granularity - 1) // granularity
    num_picks = sum(picks_per_shard)

    # Get the shard ID of each of those picks.
    shard_ids = np.arange(num_shards)
    pick_shard_ids = np.repeat(shard_ids, picks_per_shard)

    # Get the size in samples of each of those picks.
    pick_samples_per_shard = np.full(num_shards, granularity)
    samples_per_pick = np.repeat(pick_samples_per_shard, picks_per_shard)
    shard_last_pick_indices = np.cumsum(picks_per_shard) - 1
    shard_last_pick_sizes = samples_per_shard - (picks_per_shard - 1) * granularity
    samples_per_pick[shard_last_pick_indices] = shard_last_pick_sizes

    # Deterministically shuffle the picks.
    epoch_seed = seed + epoch if use_epoch else seed
    rng = np.random.default_rng(epoch_seed)
    pick_ordering = rng.permutation(num_picks)

    # Add up the picks until we have enough chosen samples to get choose per shard.
    choose_per_shard = samples_per_shard * (choose // num_samples)
    choose %= num_samples
    for pick_id in pick_ordering:
        shard_id = pick_shard_ids[pick_id]
        num_picked = int(samples_per_pick[pick_id])
        num_picked = min(choose, num_picked)
        choose_per_shard[shard_id] += num_picked
        choose -= num_picked
        if not choose:
            break

    return choose_per_shard
