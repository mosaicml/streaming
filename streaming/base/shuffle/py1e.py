# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shuffling algorithm that shuffles intra-shard in one place.

This algorithm is roughly twice as fast as algorithm ``py2s``, and ever so slightly biased.

Bias in this case merely refers to how we assign samples when we split shards at canonical node
boundaries, which is non-random in this algorithm. In practice, we found this does not matter to
convergence, while making us faster.
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
    # also get max shard size to calculate the
    spans = []
    num_samples = 0
    max_shard_size = 0
    for shard_size in shard_sizes:
        span = num_samples, num_samples + shard_size
        spans.append(span)
        num_samples += shard_size
        if shard_size > max_shard_size:
            max_shard_size = shard_size

    # Generate the initial ordering of shards, which is fixed over an entire training run.
    run_rng = np.random.default_rng(seed)
    run_rng.shuffle(spans)

    # Break the shard spans at canonical node boundaries.
    # super_spans are the indices of spans that correspond to each canonical node
    spans, super_spans = divide_spans(spans, num_samples, num_canonical_nodes)

    # Shuffle the span ordering within each canonical node uniquely to this epoch.
    epoch_rng = np.random.default_rng(seed + epoch)
    for begin, end in super_spans:
        # retrieving the spans (shard parts) associated with this canonical node
        part = spans[begin:end]
        epoch_rng.shuffle(part)  # pyright: ignore
        spans[begin:end] = part

    # Populate the global sample ID mapping, shuffling within each span.
    ids = np.empty(num_samples, np.int64)
    offset = 0
    # iterate through each canonical node's spans because we don't want samples crossing canonical node boundaries
    for cn_begin, cn_end in super_spans:
        cn_spans = spans[cn_begin:cn_end]
        cn_span_sizes = np.array([end - begin for begin, end in cn_spans])
        num_cn_samples = cn_span_sizes.sum()
        # the spans of a canonical node are shuffled, so they have sample ids that are
        # not contiguous. need to get the correct sample ids for the current canonical node
        cn_samples = np.empty(num_cn_samples)
        samples_inserted = 0
        for begin, end in cn_spans:
            # insert span samples into cn_samples array
            cn_span_samples = np.arange(begin, end)
            epoch_rng.shuffle(cn_span_samples)
            cn_samples[samples_inserted:samples_inserted + (end - begin)] = cn_span_samples
            samples_inserted += (end - begin)

        # iterate over each span and shift sample indices by gaussian noise
        cn_sample_offset = 0
        shifted_samples = np.arange(num_cn_samples).astype(np.float64)
        for span_size in cn_span_sizes:

            # cutoff is (block_size - span_size)/2, so the span samples
            # are only found in a range of maximum possible size block_size
            cutoff = (block_size - span_size) / 2

            # make sure the lower bound doesn't cross the start of the canonical node
            lower_bound = max(-cutoff, -cn_sample_offset)
            # make sure the upper bound doesn't cross the end of the canonical node
            upper_bound = min(cutoff, num_cn_samples - cn_sample_offset - span_size)
            # sample shifts from uniform distribution
            shifts = epoch_rng.uniform(low=lower_bound, high=upper_bound, size=span_size)

            # add shifts to shard samples
            shifted_samples[cn_sample_offset:cn_sample_offset + span_size] += shifts

            # update offset for next shard
            cn_sample_offset += span_size

        # get incides that would sort the shifted_samples array
        sort_indices = np.argsort(shifted_samples)

        # apply the sorting to the samples for our canonical node
        cn_samples = cn_samples[sort_indices]

        # assign the gaussian "shuffled" samples to the global ids array
        ids[offset:offset + num_cn_samples] = cn_samples

        offset += num_cn_samples

    return ids
