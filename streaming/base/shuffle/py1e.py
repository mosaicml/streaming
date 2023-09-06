# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shuffling algorithm that shuffles by randomly placing shard samples in expanded ranges.

This algorithm has more balanced downloading and a lower minimum cache limit than ``py1b`` and
``py1br``, but also slightly lower shuffle quality. The maximum range the samples from each shard
can cover is determined by ``shuffle_block_size``.
"""

import numpy as np
from numpy.typing import NDArray

from streaming.base.shuffle.py1s import divide_spans


def get_shuffle_py1e(shard_sizes: NDArray[np.int64],
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
        block_size (int): Unit of shuffle, used to set the std and clip length for the gaussian
            noise to be added to each shard. Defaults to ``1 << 18``.

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
    # The super_spans are the indices of spans that correspond to each canonical node.
    spans, super_spans = divide_spans(spans, num_samples, num_canonical_nodes)

    # Shuffle the span ordering within each canonical node uniquely to this epoch.
    epoch_rng = np.random.default_rng(seed + epoch)
    for begin, end in super_spans:
        # Retrieve the spans (shard parts) associated with this canonical node.
        part = spans[begin:end]
        epoch_rng.shuffle(part)  # pyright: ignore
        spans[begin:end] = part

    # Populate the global sample ID mapping, shuffling within each span.
    ids = np.empty(num_samples, np.int64)
    offset = 0
    # Iterate through each canonical node's spans because we don't want samples crossing canonical node boundaries
    for cn_begin, cn_end in super_spans:
        cn_spans = spans[cn_begin:cn_end]
        cn_span_sizes = np.array([end - begin for begin, end in cn_spans])
        num_cn_samples = cn_span_sizes.sum()
        # The spans of a canonical node are shuffled, so they have sample ids that are
        # not contiguous. We need to get the correct sample ids for the current canonical node.
        cn_samples = np.empty(num_cn_samples)
        samples_inserted = 0
        for begin, end in cn_spans:
            # Inserting span samples into cn_samples array.
            cn_span_samples = np.arange(begin, end)
            epoch_rng.shuffle(cn_span_samples)
            cn_samples[samples_inserted:samples_inserted + (end - begin)] = cn_span_samples
            samples_inserted += (end - begin)

        # Iterate over each span and shift sample indices by randomly sampled shifts from uniform distribution.
        cn_sample_offset = 0
        sample_positions = np.arange(num_cn_samples).astype(np.float64)
        for span_size in cn_span_sizes:

            # The maximum range on each side of the span is (block_size - span_size) / 2.
            # This ensures that the span samples are only found in a range of max possible size block_size.
            cutoff = (block_size - span_size) / 2

            # Make sure the lower bound of the range doesn't cross the start of the canonical node.
            lower_bound = max(-cutoff, -cn_sample_offset)
            # Make sure the upper bound of the range doesn't cross the end of the canonical node.
            upper_bound = min(cutoff, num_cn_samples - cn_sample_offset - span_size)
            # Sample shifts from a uniform distribution with the bounds calculated above.
            shifts = epoch_rng.uniform(low=lower_bound, high=upper_bound, size=span_size)

            # Add shifts to shard sample indices.
            sample_positions[cn_sample_offset:cn_sample_offset + span_size] += shifts

            # Update sample offset for the next shard.
            cn_sample_offset += span_size

        # Get incides that would sort the sample_positions array.
        sort_indices = np.argsort(sample_positions)

        # Apply the sorting to the samples for our canonical node.
        cn_samples = cn_samples[sort_indices]

        # Assign the newly shuffled samples to the global ids array.
        ids[offset:offset + num_cn_samples] = cn_samples

        offset += num_cn_samples

    return ids
