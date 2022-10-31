# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming DataLoader."""

from typing import Any, Dict

from torch.utils.data import DataLoader

from streaming.base.dataset import Dataset
from streaming.base.world import World

class StreamingDataLoader(DataLoader):

    def count_and_yield_samples(self, batch: Any):
        self._num_samples_yielded += batch[0].shape[0] # TODO: Does this batch[0] work in general? Dataloaders might have different yield styles
        return batch

    def __iter__(self):
        self._num_samples_yielded = 0
        for batch in super().__iter__():
            yield self.count_and_yield_samples(batch)

    @property
    def num_samples_yielded(self):
        return self._num_samples_yielded if hasattr(self, '_num_samples_yielded') else 0

    def state_dict(self):
        if isinstance(self.dataset, Dataset):
            world = World()
            return self.dataset.state_dict(self.num_samples_yielded * world.num_ranks)
        return {}

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        if isinstance(self.dataset, Dataset):
            return self.dataset.load_state_dict(obj)
