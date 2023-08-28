# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Functionality relating to sampling."""

import numpy as np
from numpy.typing import NDArray


def get_shard_sampling(samples_per_shard: NDArray[np.int64], choose: int,
                       visits_per_shard: int) -> NDArray[np.int64]:
    """Get how many samples to draw from each shard of the given stream.

    Args:
        samples_per_shard (NDArray[np.int64]): Array of underlying shard sizes.
        choose (int): How many samples to draw in total over all shards.
        visits_per_shard (int): A slider to trade off shard efficiency vs shard diversity. When you
            are sampling few samples from many shards, do you draw all the samples from the same
            shard in order to save downloading, or do you balance it out over a bunch of them?

    Returns:
        NDArray[np.int64]: Array of ephemeral samples chosen per shard.
    """
    # Handle whole integer repeat case.
    samples = sum(samples_per_shard)
    if not choose % samples:
        return samples_per_shard * choose // samples

    # Fractional repeat case.

    # Get the ordering by which we will exhaust the shards.
    num_shards = len(samples_per_shard)
    visit_ids = np.arange(num_shards * visits_per_shard)
    np.random.shuffle(visit_ids)

    # Get sample size of each visit of each shard.
    x = np.arange(visits_per_shard)
    x = np.expand_dims(x, 0)
    samples_per_shard_rows = np.expand_dims(samples_per_shard, 1)
    begins = samples_per_shard_rows * x // visits_per_shard
    ends = samples_per_shard_rows * (x + 1) // visits_per_shard
    samples_per_visit = (ends - begins).flatten()

    # Start choose per shard with the full repeats.
    choose_per_shard = samples_per_shard * (choose // samples)
    choose %= samples

    # Walk visits, adding to choose per shard for the last partial reeat.
    for visit_id in visit_ids:
        shard_id = visit_id // visits_per_shard
        visit_samples = min(int(samples_per_visit[visit_id]), choose)
        choose_per_shard[shard_id] += visit_samples
        choose -= visit_samples
        if not choose:
            break

    return choose_per_shard
