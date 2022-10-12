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


def _drop_first_samples(shards: List[_Shard], drop: int) -> None:
    """Drop the given number of samples from the front of a partition (modifies in-place).

    Args:
        shards (List[_Shard]): Partition of shards.
        drop (int): Number of samples to drop from the front of the partition.
    """
    new_shards = []
    samples_so_far = 0
    for shard in shards:
        samples_this_shard = len(shard.samples)
        if drop < samples_so_far:
            new_shards.append(shard)
        elif drop < samples_so_far + samples_this_shard:
            split = samples_so_far + samples_this_shard - drop
            new_shard = _Shard(shard.index, shard.samples[-split:])
            new_shards.append(new_shard)
        else:
            pass
        samples_so_far += samples_this_shard
    shards.clear()
    shards.extend(new_shards)


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
    for part in range(num_parts):
        part_end = total_samples * (part + 1) // num_parts
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


def _concat_parts(lists: List[List[_Shard]]) -> List[_Shard]:
    """Join the given lists of shards into one global list of shards (undo a partitioning).

    Args:
        lists (List[List[_Shard]]): The input partitions.

    Returns:
        List[_Shard]: The list of all shards.
    """
    ret = []
    for shards in lists:
        ret += shards
    return ret


def get_epoch(sizes: NDArray[np.int64], shuffle: bool, seed: int, epoch: int,
              sessions: List[NDArray[np.int64]]):
    """Get this epoch's ordering of samples for each worker.

    Mid-epoch resumption:
    - Deterministic shuffle given seed and epoch.
    - Drops already-processed samples according to sessions.

    Approximate shuffling:
    - Shuffles shards and samples within shards, but not samples across shards.
    - This is due to the scattering effect of additional sessions with changed workers/world size
      on the shards needed by each worker for its samples, which could easily overwhelm i/o.
    - It's also low compute enough to be able to shufflle large datasets in pure numpy in
      reasonable time.
    - Shufflle quality is not expected to be a practical concern outside of the very low data with
      low device regime. We are more targeting scalability.

    Args:
        sizes (List[int]): Number of samples contained in each shard, in order.
        shuffle (bool): Whether to approximately randomize the sample ordering (see notes), while
            still using the same shards across epochs as long as workers/world size is the same.
        seed (int): Base random seed, which is held constant over an entire training run.
        epoch (int): Current epoch, which is added to the seed to get a different deterministic
            shuffle each epoch.
        sessions (List[NDArray[np.int64]]): Partial epoch progress, in the form of an array per
            partial epoch training session. Each array contains the samples seen per worker during
            that session, which can be zero. The list will always contain at least one array.

    Returns:
        List[NDArray[np.int64]]: Sequence of sample IDs for each worker.
    """
    if not sessions:
        raise RuntimeError('There must be at least one training session')

    # Initialize fixed and per-epoch PRNGs.
    static_rng = np.random.default_rng(seed)
    epoch_rng = np.random.default_rng(seed + epoch)

    # Initialize the sample ID range for each shard.
    shards = _create_shards(sizes)

    # Do the original fixed scattering of shards over the sample space.
    if shuffle:
        static_rng.shuffle(shards)  # pyright: ignore
        for shard in shards:
            static_rng.shuffle(shard.samples)

    # Reproduce this partially completed epoch's training history.
    for samples_per_part in sessions:
        num_parts = len(samples_per_part)
        parts = _break_into_balanced_parts(shards, num_parts)
        for shards, num_samples in zip(parts, samples_per_part):
            if shuffle:
                epoch_rng.shuffle(shards)  # pyright: ignore
                for shard in shards:
                    epoch_rng.shuffle(shard.samples)
            _drop_first_samples(shards, num_samples)
        shards = _concat_parts(parts)  # Result is ignored on last session.

    # Return the array of sample IDs for each partition.
    return list(map(_shards_to_samples, parts))  # pyright: ignore
