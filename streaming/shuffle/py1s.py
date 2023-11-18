# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shuffling algorithm that shuffles intra-shard in one place.

This algorithm is roughly twice as fast as algorithm ``py2s``, and ever so slightly biased.

Bias in this case merely refers to how we assign samples when we split shards at canonical node
boundaries, which is non-random in this algorithm. In practice, we found this does not matter to
convergence, while making us faster.
"""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def divide_spans(spans: List[Tuple[int, int]], num_samples: int, num_parts: int) -> \
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
        # note that the size of a part (canonical node) is num_samples // num_parts.
        part_end = num_samples * (part + 1) // num_parts

        # loop over spans until we've filled up our part (canonical node) completely
        while True:
            if span_index == len(spans):
                break

            # input spans are the shard spans. these can be unequally sized and may cross
            # part (canonical node) boundaries.
            span = spans[span_index]
            # spans are (begin, end excl)
            samples_this_span = span[1] - span[0]
            # check if the shard span contains more samples than the part (canonical node) can fit
            if part_end < samples_so_far + samples_this_span:
                # if there is space left in the part, split the span
                if samples_so_far < part_end:
                    split = part_end - samples_so_far
                    # create a span, filling up with as many samples as possible from shard span
                    new_span = span[0], span[0] + split
                    out_spans.append(new_span)
                    # modify the old shard span to reflect that it's been split
                    spans[span_index] = span[0] + split, span[1]
                    samples_so_far += split
                break

            out_spans.append(span)
            span_index += 1
            samples_so_far += samples_this_span

        # super spans are tell us which new spans belong to each part (canonical node)
        # as a tuple of (begin span index, end span index excl)
        super_span = begin_part, len(out_spans)
        super_spans.append(super_span)
        begin_part = len(out_spans)

    return out_spans, super_spans


def get_shuffle_py1s(shard_sizes: NDArray[np.int64],
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
        block_size (int): Unit of shuffle (ignored, because we shuffle on the basis of shards).
            Defaults to ``1 << 18``.

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
    spans, super_spans = divide_spans(spans, num_samples, num_canonical_nodes)

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
