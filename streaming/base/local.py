# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A non-streaming pytorch map Dataset."""

import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset

from streaming.base.format import reader_from_json
from streaming.base.index import Index, get_index_basename
from streaming.base.util import set_mp_start_method

__all__ = ['LocalDataset']


class LocalDataset(Dataset):
    """A streaming dataset whose shards reside locally as a pytorch Dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
    """

    def __init__(self, local: str, split: Optional[str] = None):
        split = split or ''

        self.local = local
        self.split = split
        set_mp_start_method(sys.platform)

        filename = os.path.join(local, split, get_index_basename())  # pyright: ignore
        obj = json.load(open(filename))
        if obj['version'] != 2:
            raise ValueError('Unsupported version')

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

    def __getitem__(self, sample_id: int) -> Dict[str, Any]:
        """Get sample by global sample ID.

        Args:
            sample_id (int): Sample ID.

        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard_id, index_in_shard = self.index.find_sample(sample_id)
        shard = self.shards[shard_id]
        return shard[index_in_shard]
