# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Helper methods to get the shard attributes."""

from math import ceil
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from streaming.base import distributed as dist

__all__ = ['get_index_basename', 'Index']


def get_index_basename() -> str:
    """Get the canonical index file basename.

    Returns:
        str: Index basename.
    """
    return 'index.json'


class Index(object):
    """An index of sample ranges (corresponding to shards).

    Maps global sample IDs to their shards and offsets.

    Args:
        samples_per_shard (NDArray[np.int64]): Size of each shard, in samples.
    """

    def __init__(self, samples_per_shard: NDArray[np.int64]) -> None:
        self.samples_per_shard = samples_per_shard

        self.total_samples = sum(samples_per_shard)
        self.shard_offsets = samples_per_shard.cumsum() - samples_per_shard

        # Make a lookup table of sample to shard, stored in the form of equal-sized spans of sample
        # IDs that map to at most two adjacent shards, keeping the dividing sample ID.
        if len(samples_per_shard[:-1]):
            self.slot_size = min(samples_per_shard[:-1])
        else:
            self.slot_size = samples_per_shard[-1]
        self.slot_size = self.slot_size or 1  # For the edge case of empty shards.
        self.num_slots = (self.total_samples + self.slot_size - 1) // self.slot_size
        shard_ends = samples_per_shard.cumsum()
        shard = 0
        slots = []
        for slot in range(self.num_slots):
            slot_end = (slot + 1) * self.slot_size
            if shard_ends[shard] < slot_end:
                div = shard_ends[shard]
                slots.append((shard, div))
                shard += 1
            else:
                div = slot_end
                slots.append((shard, div))
        self.slots = np.array(slots)

    def find_sample(self, idx: int) -> Tuple[int, int]:
        """Get the shard and offset where a sample will be found.

        Args:
            idx (int): Global sample index.

        Returns:
            Tuple[int, int]: Shard and sample index within that shard.
        """
        if not (0 <= idx < self.total_samples):
            raise ValueError(f'Invalid sample index: 0 <= {idx} < {self.total_samples}')
        slot_idx = min(idx // self.slot_size, len(self.slots) - 1)
        shard, div = self.slots[slot_idx]
        if div <= idx:
            shard += 1
        offset = idx - self.shard_offsets[shard]
        return shard, offset  # pyright: ignore

    def get_samples_per_device(self) -> int:
        """Get the per-device dataset size (i.e., IterableDataset.__len__).

        Returns:
            int: Per-device dataset size.
        """
        return ceil(self.total_samples / dist.get_world_size())
