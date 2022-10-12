# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A non-streaming pytorch IterableDataset with mid-epoch resumption."""

import json
import os
from typing import Any, Dict, Iterator, Optional

import numpy as np
from torch.utils.data import IterableDataset

from streaming.base.cursor import Cursor
from streaming.base.format import reader_from_json
from streaming.base.index import Index
from streaming.base.shuffle import get_epoch
from streaming.base.world import World


class LocalResumableDataset(IterableDataset):
    """A resumable streaming dataset whose shards reside locally as a pytorch IterableDataset.

    Args:
        local (str): Local dataset directory where the dataset is present.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to shuffle the samples while iterating. Defaults to ``True``.
        seed (int): Base random seed, used for shuffling.
    """

    def __init__(self,
                 local: str,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 seed: int = 42):
        self.local = local
        self.split = split or ''
        self.shuffle = shuffle
        self.seed = seed

        filename = os.path.join(local, split, 'index.json')  # pyright: ignore
        obj = json.load(open(filename))
        assert obj['version'] == 2

        self.shards = []
        for info in obj['shards']:
            shard = reader_from_json(local, split, info)
            self.shards.append(shard)

        self.shard_sizes = np.array([x.samples for x in self.shards])
        self.index = Index(self.shard_sizes)

        self.cursor = Cursor(self.split)

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
        world = World()
        self.cursor.push_session(world)

        sequences = get_epoch(self.shard_sizes, self.shuffle, self.seed, self.cursor.epoch,
                              self.cursor.sessions)
        todos = sequences[world.worker]

        for index in todos:
            self.cursor.step_sample(world)
            yield self[index]

        self.cursor.pop_sessions(world)
        self.cursor.step_epoch(world)

    def state_dict(self) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        Returns:
            Dict[str, Any]: The state.
        """
        return self.cursor.state_dict()

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        Args:
            obj (Dict[str, Any]): The state.
        """
        self.cursor.load_state_dict(obj)
