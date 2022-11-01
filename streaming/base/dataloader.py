# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming DataLoader."""

from typing import Any, Dict, Iterator, Optional

from torch import Tensor
from torch.utils.data import DataLoader

from streaming.base.dataset import Dataset
from streaming.base.world import World


class StreamingDataLoader(DataLoader):
    """A streaming data loader.

    Provides an additional checkpointing/resumption interface, for which it tracks the number of
    samples seen by the model this rank.

    Args:
        *args: Llist arguments.
        **kwargs: Keyword arguments.
    """

    def __init__(self, *args, **kwargs) -> None:  # pyright: ignore
        super().__init__(*args, **kwargs)
        self.num_samples_yielded = 0

    def __iter__(self) -> Iterator[Any]:
        """Iterate over this DataLoader, yielding batches.

        Also tracks the number of samples seen this rank.

        Returns:
            Iterator[Any]: Each batch.
        """
        self.num_samples_yielded = 0
        for batch in super().__iter__():
            if isinstance(batch, Tensor):
                count = len(batch)
            else:
                count = len(batch[0])
            self.num_samples_yielded += count
            yield batch

    def state_dict(self) -> Optional[Dict[str, Any]]:
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Args:
            samples_in_epoch (int): The number of samples processed so far in the current epoch.

        Returns:
            Optional[Dict[str, Any]]: The state, if a streaming dataset.
        """
        if isinstance(self.dataset, Dataset):
            world = World()
            return self.dataset.state_dict(self.num_samples_yielded * world.num_ranks)
        return None

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        Args:
            obj (Dict[str, Any]): The state.
        """
        if isinstance(self.dataset, Dataset):
            return self.dataset.load_state_dict(obj)
