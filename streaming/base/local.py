# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Local Dataset."""

import json
import os
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from streaming.base.format import reader_from_json
from streaming.base.index import Index

__all__ = ['LocalDataset']


class LocalDataset(Dataset):
    """The dataset resides locally in a machine.

    Args:
        dirname (str): Local dataset directory where the dataset is present.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
    """

    def __init__(self, dirname: str, split: Optional[str] = None):
        split = split or ''

        self.dirname = dirname
        self.split = split

        filename = os.path.join(dirname, split, 'index.json')
        obj = json.load(open(filename))
        assert obj['version'] == 2

        self.shards = []
        for info in obj['shards']:
            shard = reader_from_json(dirname, split, info)
            self.shards.append(shard)

        shard_sizes = list(map(lambda x: x.samples, self.shards))
        self.index = Index(shard_sizes)

    def __len__(self) -> int:
        """Get the length as an IterableDataset.

        Returns:
            int: Dataset length.
        """
        return self.index.total_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by global index.

        Args:
            idx (int): Sample index.

        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard_idx, idx_in_shard = self.index.find_sample(idx)
        shard = self.shards[shard_idx]
        return shard[idx_in_shard]
