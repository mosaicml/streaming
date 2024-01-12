# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A non-streaming pytorch map Dataset."""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset

from streaming.base.array import Array
from streaming.base.format import get_index_basename, reader_from_json
from streaming.base.spanner import Spanner

__all__ = ['LocalDataset']


class LocalDataset(Array, Dataset):
    """A streaming dataset whose shards reside locally as a pytorch Dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
    """

    def __init__(self, local: str, split: Optional[str] = None):
        split = split or ''

        self.local = local
        self.split = split

        filename = os.path.join(local, split, get_index_basename())  # pyright: ignore
        obj = json.load(open(filename))
        if obj['version'] != 2:
            raise ValueError(f'Unsupported streaming data version: {obj["version"]}. ' +
                             f'Expected version 2.')

        self.shards = []
        for info in obj['shards']:
            shard = reader_from_json(local, split, info)
            self.shards.append(shard)
        self.num_samples = sum([shard.samples for shard in self.shards])

        shard_sizes = np.array([x.samples for x in self.shards])
        self.spanner = Spanner(shard_sizes)

    def __len__(self) -> int:
        """Get the length as a PyTorch Dataset.

        Returns:
            int: Dataset length.
        """
        return self.num_samples

    @property
    def size(self) -> int:
        """Get the size of the dataset in samples.

        Returns:
            int: Number of samples.
        """
        return self.num_samples

    def get_item(self, sample_id: int) -> Dict[str, Any]:
        """Get sample by global sample ID.

        Args:
            sample_id (int): Sample ID.

        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard_id, index_in_shard = self.spanner[sample_id]
        shard = self.shards[shard_id]
        return shard[index_in_shard]
