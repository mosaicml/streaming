# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shuffling algorithm that shuffles in fixed-size blocks.

These units are presumably larger or much larger than single shards, leading to better shuffledness
at the cost of having to download more shards to make progress.
"""

import numpy as np
from numpy.typing import NDArray

from streaming.base.shuffle.py1s import divide_spans


def get_shuffle_py1b(shard_sizes: NDArray[np.int64],
                     num_canonical_nodes: int,
                     seed: int,
                     epoch: int,
                     block_size: int = 1 << 18) -> NDArray[np.int64]:
    """Get the shuffled global ordering of samples for an epoch.

    The assignment of shards to nodes is fixed across epochs, but each grouping of shards is
    processed concurrently in a different order by each node's workers each epoch.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples contained in each shard, in order.
        num_canonical_nodes (int): Number of canonical nodes.
        seed (int): Base random seed, which is held constant over an entire training run.
        epoch (int): Current epoch, which is added to the seed to get a different deterministic
            shuffle each epoch.
        block_size (int): Unit of shuffle. Defaults to ``1 << 18``.

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
    # Because the ordering of shards is fixed the downloaded shards from the first epoch
    # can be persisted and used for subsequent epochs in each node as well.
    run_rng = np.random.default_rng(seed)
    run_rng.shuffle(spans)

    # Break the shard spans at canonical node boundaries.
    spans, super_spans = divide_spans(spans, num_samples, num_canonical_nodes)

    # Shuffle the span ordering within each canonical node uniquely to this epoch.
    epoch_rng = np.random.default_rng(seed + epoch)
    for begin, end in super_spans:
        part = spans[begin:end]
        epoch_rng.shuffle(part)  # pyright: ignore
        spans[begin:end] = part

    # Populate the global sample ID mapping, shuffling within each block within each super-span.
    ids = np.empty(num_samples, np.int64)
    offset = 0
    # Loop over each canonical node.
    for super_begin, super_end in super_spans:
        # The super_offset is the offset of the first sample in the canonical node.
        super_offset = offset
        # Loop over each span contained in the canonical node.
        for begin, end in spans[super_begin:super_end]:
            span_size = end - begin
            ids[offset:offset + span_size] = np.arange(begin, end)
            offset += span_size
        # Shuffle within each block, but don't shuffle past the canonical node boundary.
        for start in range(super_offset, offset, block_size):
            stop = min(start + block_size, offset)
            epoch_rng.shuffle(ids[start:stop])

    return ids
