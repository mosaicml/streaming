# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shuffling algorithm that shuffles in fixed-size blocks.

These units are presumably larger or much larger than single shards, leading to better shuffledness
at the cost of having to download more shards to make progress.
"""

import numpy as np
from numpy.typing import NDArray

from streaming.base.shuffle.py1s import divide_spans


def get_shuffle_py1br(shard_sizes: NDArray[np.int64],
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
        block_size (int): Unit of shuffle. For py1br shuffling method, the block size is chosen
            uniformly at random in the range (0.75*block_size, 1.25*block_size).
            Defaults to ``1 << 18``.

    Returns:
        NDArray[np.int64]: 1:1 mapping of sample ID to shuffled sample ID.
    """
    # Create each shard's sample ID span (start, stop excl).
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
    spans, node_spans = divide_spans(spans, num_samples, num_canonical_nodes)

    # Shuffle the span ordering within each canonical node uniquely to this epoch.
    epoch_rng = np.random.default_rng(seed + epoch)
    for node_start_span, node_stop_span in node_spans:
        node_span = spans[node_start_span:node_stop_span]
        epoch_rng.shuffle(node_span)  # pyright: ignore
        spans[node_start_span:node_stop_span] = node_span

    # Populate the global sample ID mapping, shuffling within each block within each node.
    ids = np.empty(num_samples, np.int64)
    node_stop_sample = 0
    stagger = epoch_rng.integers(0, int(0.75 * block_size), (num_canonical_nodes,))
    for node, (node_start_span, node_stop_span) in enumerate(node_spans):
        node_start_sample = node_stop_sample

        # Populate sample IDs given the span ordering for this node.
        for span_start_sample, span_stop_sample in spans[node_start_span:node_stop_span]:
            span_size = span_stop_sample - span_start_sample
            ids[node_stop_sample:node_stop_sample + span_size] = \
                np.arange(span_start_sample, span_stop_sample)
            node_stop_sample += span_size

        # Get randomized and staggered block ranges for the current node.
        block_staggered_ranges = []
        blocks_end = node_start_sample
        node_stagger = stagger[node]
        while blocks_end < node_stop_sample:
            rand_block_size = epoch_rng.integers(int(0.75 * block_size), int(1.25 * block_size))
            # We don't want the block to start before the first sample of the node.
            staggered_block_start = max(blocks_end - node_stagger, node_start_sample)
            # We don't want the block to stop after the last sample of the node.
            staggered_block_stop = min(blocks_end + rand_block_size - node_stagger,
                                       node_stop_sample)
            block_staggered_ranges.append((staggered_block_start, staggered_block_stop))
            blocks_end += staggered_block_stop - staggered_block_start

        # Shuffle within each staggered, randomized block.
        for block_start, block_stop in block_staggered_ranges:
            epoch_rng.shuffle(ids[block_start:block_stop])

    return ids
