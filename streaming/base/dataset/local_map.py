# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A non-streaming pytorch map Dataset."""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset

from streaming.base.format import reader_from_json
from streaming.base.index import Index

__all__ = ['LocalDataset']


class LocalMapDataset(Dataset):
    """A streaming dataset whose shards reside locally as a pytorch Dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
    """

    def __init__(self, local: str, split: Optional[str] = None):
        split = split or ''

        self.local = local
        self.split = split

        filename = os.path.join(local, split, 'index.json')  # pyright: ignore
        obj = json.load(open(filename))
        assert obj['version'] == 2

        self.shards = []
        for info in obj['shards']:
            shard = reader_from_json(local, split, info)
            self.shards.append(shard)

        shard_sizes = np.array([x.samples for x in self.shards])
        self.index = Index(shard_sizes)

    def __len__(self) -> int:
        """Get the length as an IterableDataset.

        Returns:
            int: Dataset length.
        """
        return self.index.total_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get sample by global index.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard, index_in_shard = self.index.find_sample(index)
        reader = self.shards[shard]
        return reader[index_in_shard]
