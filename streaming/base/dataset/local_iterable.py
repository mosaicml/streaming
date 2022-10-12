# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A non-streaming pytorch IterableDataset."""

import json
import os
from typing import Any, Dict, Iterator, Optional

import numpy as np
from torch.utils.data import IterableDataset

from streaming.base.format import reader_from_json
from streaming.base.index import Index


class LocalIterableDataset(IterableDataset):
    """A streaming dataset whose shards reside locally as a pytorch IterableDataset.

    Args:
        local (str): Local dataset directory where the dataset is present.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to shuffle the samples while iterating. Defaults to ``True``.
    """

    def __init__(self, local: str, split: Optional[str] = None, shuffle: bool = True):
        self.local = local
        self.split = split or ''
        self.shuffle = shuffle

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
        return self.index.get_samples_per_device()

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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        part = self.index.get_partition()
        todos = np.arange(part.min_sample_id, part.max_sample_id)
        if self.shuffle:
            np.random.shuffle(todos)
        for index in todos:
            yield self[index]
