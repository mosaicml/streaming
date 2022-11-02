# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Slice, shuffle, and dice epochs of samples across worker partitions."""

from typing import List

import numpy as np
from numpy.typing import NDArray


class _Shard(object):
    """Shard ID paired with its sample IDs.

    Args:
        index (int): Shard ID.
        smaples (NDArray[np.int64]): Sample IDs
    """

    def __init__(self, index: int, samples: NDArray[np.int64]) -> None:
        self.index = index
        self.samples = samples


def _create_shards(sizes: NDArray[np.int64]) -> List[_Shard]:
    """Get the sample ID range for each shard.

    Args:
        sizes (NDArray[np.int64]): Number of samples for each shard.

    Returns:
        List[_Shard]: List of shard objects.
    """
    shards = []
    ends = sizes.cumsum()
    begins = ends - sizes
    for shard, (begin, end) in enumerate(zip(begins, ends)):
        shard = _Shard(shard, np.arange(begin, end))
        shards.append(shard)
    return shards


def _shards_to_samples(shards: List[_Shard]) -> NDArray[np.int64]:
    """Collect the sample IDs of the given shards into a single array.

    Args:
        shards (List[_Shard]): The given shards.

    Returns:
        NDArray[np.int64]: Their sample IDs.
    """
    for shard in shards:
        if len(shard.samples):
            arrs = [shard.samples for shard in shards]
            return np.concatenate(arrs)
    return np.array([], np.int64)


def _break_into_balanced_parts(shards: List[_Shard], num_parts: int) -> List[List[_Shard]]:
    """Divide the given shards into partitions (groupings of shards).

    Warning: don't use `shards` after this, as its memory is recycled into the returned partitions
    for performance reasons.

    Args:
        shards (List[_Shard]): List of shards to partition.
        num_parts (int): Number of groupings to divide shards into.

    Returns:
        List[List[_Shard]]: Partitions of shards.
    """
    total_samples = sum([len(x.samples) for x in shards])
    lists = []
    shard_index = 0
    samples_so_far = 0
    part_sizes = [total_samples // num_parts] * num_parts
    part_ends = np.array(part_sizes).cumsum()
    for part_end in part_ends:
        new_shards = []
        while True:
            if shard_index == len(shards):
                break

            shard = shards[shard_index]
            samples_this_shard = len(shard.samples)
            if part_end < samples_so_far + samples_this_shard:
                if samples_so_far < part_end:
                    split = part_end - samples_so_far
                    new_shard = _Shard(shard.index, shard.samples[:split])
                    new_shards.append(new_shard)
                    shards[shard_index].samples = shard.samples[split:]
                    samples_so_far += split
                break

            new_shards.append(shard)
            shard_index += 1
            samples_so_far += samples_this_shard

        lists.append(new_shards)
        new_shards = []
    return lists


def get_shuffle(shard_sizes: NDArray[np.int64], shuffle: bool, seed: int, world_size: int,
                epoch: int) -> NDArray[np.int64]:
    """Get the global ordering of samples for an epoch.

    Approximate shuffling:
    - Shuffles shards and samples within shards, but not samples across shards.
    - This is due to the scattering effect of additional sessions with changed workers/world size
      on the shards needed by each worker for its samples, which could easily overwhelm i/o.
    - It's also low compute enough to be able to shuffle large datasets in pure numpy in reasonable
      time.
    - Shuffle quality is not expected to be a practical concern outside of the very low data with
      low device regime. We are targeting at-scale.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples contained in each shard, in order.
        shuffle (bool): Whether to approximately randomize the sample ordering (see notes), while
            still using the same shards across epochs as long as workers/world size is the same.
        seed (int): Base random seed, which is held constant over an entire training run.
        world_size (int): Canonical world size for which this shuffle is optimized.
        epoch (int): Current epoch, which is added to the seed to get a different deterministic
            shuffle each epoch.

    Returns:
        NDArray[np.int64]: World size interleaved sequences of sample IDs giving the epoch.
    """
    # Initiailze the sample ID range for each shard.
    shards = _create_shards(shard_sizes)

    # Do the initial fixed scattering of shards over the sample space.
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(shards)  # pyright: ignore
        for shard in shards:
            rng.shuffle(shard.samples)

    # Shuffle uniquely for the current epoch within each canonical rank.
    parts = _break_into_balanced_parts(shards, world_size)
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        for shards in parts:
            rng.shuffle(shards)  # pyright: ignore
            for shard in shards:
                rng.shuffle(shard.samples)

    arrs = list(map(_shards_to_samples, parts))
    arr = np.stack(arrs)
    return arr.transpose().flatten()
