# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A non-streaming pytorch map Dataset."""

from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset

from streaming.array import Array
from streaming.spanner import Spanner
from streaming.stream.base import Stream

__all__ = ['LocalDataset']


class LocalDataset(Array, Dataset):
    """A streaming dataset whose shards reside locally as a PyTorch Dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
    """

    def __init__(self, local: str, split: Optional[str] = None):
        self.stream = Stream(local=local, split=split)
        self.stream.download_index()
        self.shards = self.stream.load_index()
        shard_sizes = np.array([shard.num_samples for shard in self.stream.shards], np.int64)
        self.spanner = Spanner(shard_sizes)
        self.num_samples = sum(shard_sizes)

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
