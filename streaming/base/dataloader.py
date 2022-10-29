# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming DataLoader."""

from typing import Any, Dict

from torch.utils.data import DataLoader

from streaming.base.dataset import Dataset

class StreamingDataLoader(DataLoader):

    def state_dict(self):
        if isinstance(self.dataset, Dataset):
            return self.dataset.state_dict(self._iterator._num_yielded)
        return {}

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        if isinstance(self.dataset, Dataset):
            return self.dataset.load_state_dict(obj)