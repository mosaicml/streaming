# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Slice, shuffle, and dice epochs of samples across worker partitions."""

from typing import List, Tuple

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


def _partition(shards: List[_Shard], num_parts: int) -> List[List[_Shard]]:
    """Divide the given shards into partitions (groupings of shards).

    Warning: don't use `shards` after this, as its memory is recycled into the returned partitions
    for performance reasons.

    Args:
        shards (List[_Shard]): List of shards to partition.
        num_parts (int): Number of groupings to divide shards into.

    Returns:
        List[List[_Shard]]: Partitions of shards.
    """
    total_samples = sum(len(x.samples) for x in shards)
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


def get_shuffle_slow(shard_sizes: NDArray[np.int64], num_canonical_nodes: int, seed: int,
                     epoch: int) -> NDArray[np.int64]:
    """Get the shuffled global ordering of samples for an epoch.

    The assignment of shards to nodes is fixed across epochs, but each grouping of shards is
    processed concurrently in a different order by each node's workers each epoch.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples contained in each shard, in order.
        num_canonical_nodes (int): Number of canonical nodes.
        seed (int): Base random seed, which is held constant over an entire training run.
        epoch (int): Current epoch, which is added to the seed to get a different deterministic
            shuffle each epoch.

    Returns:
        NDArray[np.int64]: 1:1 mapping of sample ID to shuffled sample ID.
    """
    # Initiailze the sample ID range for each shard.
    shards = _create_shards(shard_sizes)

    # Do the initial fixed scattering of shards over the sample space.
    fixed_rng = np.random.default_rng(seed)
    fixed_rng.shuffle(shards)  # pyright: ignore
    for shard in shards:
        fixed_rng.shuffle(shard.samples)

    # Shuffle uniquely for the current epoch within each canonical rank.
    parts = _partition(shards, num_canonical_nodes)
    epoch_rng = np.random.default_rng(seed + epoch)
    for shards in parts:
        epoch_rng.shuffle(shards)  # pyright: ignore
        for shard in shards:
            epoch_rng.shuffle(shard.samples)

    # Flatten the shard spans to their sample IDs, then concatenate those into a global list.
    arrs = list(map(_shards_to_samples, parts))
    return np.concatenate(arrs)


def _divide_spans(spans: List[Tuple[int, int]], num_samples: int, num_parts: int) -> \
        Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Divide the spans into discrete, equal sized partitions.

    Don't use ``spans`` after this, as it is modified in-place for performance reasons.

    Args:
        spans (List[Tuple[int, int]]): List of spans to partition.
        num_samples (int): Total number of samples across all spans.
        num_parts (int): Number of groupings to divide spans into.

    Returns:
        Tuple[List[Tuple, int, int]], List[Tuple[int, int]]]: Spans and super spans.
    """
    begin_part = 0
    span_index = 0
    samples_so_far = 0

    out_spans = []
    super_spans = []

    for part in range(num_parts):
        part_end = num_samples * (part + 1) // num_parts

        while True:
            if span_index == len(spans):
                break

            span = spans[span_index]
            samples_this_span = span[1] - span[0]
            if part_end < samples_so_far + samples_this_span:
                if samples_so_far < part_end:
                    split = part_end - samples_so_far
                    new_span = span[0], span[0] + split
                    out_spans.append(new_span)
                    spans[span_index] = span[0] + split, span[1]
                    samples_so_far += split
                break

            out_spans.append(span)
            span_index += 1
            samples_so_far += samples_this_span

        super_span = begin_part, len(out_spans)
        super_spans.append(super_span)
        begin_part = len(out_spans)

    return out_spans, super_spans


def get_shuffle_med(shard_sizes: NDArray[np.int64], num_canonical_nodes: int, seed: int,
                    epoch: int) -> NDArray[np.int64]:
    """Get the shuffled global ordering of samples for an epoch.

    The assignment of shards to nodes is fixed across epochs, but each grouping of shards is
    processed concurrently in a different order by each node's workers each epoch.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples contained in each shard, in order.
        num_canonical_nodes (int): Number of canonical nodes.
        seed (int): Base random seed, which is held constant over an entire training run.
        epoch (int): Current epoch, which is added to the seed to get a different deterministic
            shuffle each epoch.

    Returns:
        NDArray[np.int64]: 1:1 mapping of sample ID to shuffled sample ID.
    """
    # Create each shard's sample ID span (begin, end excl).
    spans = []
    num_samples = 0
    for shard_size in shard_sizes:
        span = num_samples, num_samples + shard_size
        spans.append(span)
        num_samples += shard_size

    # Generate the initial ordering of shards, which is fixed over an entire training run.
    run_rng = np.random.default_rng(seed)
    run_rng.shuffle(spans)

    # Break the shard spans at canonical node boundaries.
    spans, super_spans = _divide_spans(spans, num_samples, num_canonical_nodes)

    # Shuffle the span ordering within each canonical node uniquely to this epoch.
    epoch_rng = np.random.default_rng(seed + epoch)
    for begin, end in super_spans:
        part = spans[begin:end]
        epoch_rng.shuffle(part)  # pyright: ignore
        spans[begin:end] = part

    # Populate the global sample ID mapping, shuffling within each span.
    ids = np.empty(num_samples, np.int64)
    offset = 0
    for begin, end in spans:
        span_size = end - begin
        ids[offset:offset + span_size] = np.arange(begin, end)
        epoch_rng.shuffle(ids[offset:offset + span_size])
        offset += span_size

    return ids


get_shuffle = get_shuffle_med
