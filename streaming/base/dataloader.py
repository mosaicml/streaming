# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming DataLoader."""

from typing import Any, Dict, Iterator, Optional

from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BatchEncoding, BatchFeature

from streaming.base.dataset import StreamingDataset
from streaming.base.world import World


class StreamingDataLoader(DataLoader):
    """A streaming data loader.

    Provides an additional checkpoint/resumption interface, for which it tracks the number of
    samples seen by the model this rank.

    Args:
        *args: List arguments.
        **kwargs: Keyword arguments.
    """

    def __init__(self, *args, **kwargs) -> None:  # pyright: ignore
        super().__init__(*args, **kwargs)
        self.num_samples_yielded = 0

    def _get_batch_size(self, batch: Any) -> int:
        """Get the number of samples in a batch.

        Args:
            batch (Any): The batch.

        Returns:
            int: Number of samples.
        """
        if isinstance(batch, (dict, BatchEncoding, BatchFeature)):
            for value in batch.values():
                return len(value)
            raise ValueError('Batch is empty')
        elif isinstance(batch, Tensor):
            return len(batch)
        else:
            return len(batch[0])

    def __iter__(self) -> Iterator[Any]:
        """Iterate over this DataLoader, yielding batches.

        Also tracks the number of samples seen this rank.

        Returns:
            Iterator[Any]: Each batch.
        """
        self.num_samples_yielded = 0
        for batch in super().__iter__():
            self.num_samples_yielded += self._get_batch_size(batch)
            yield batch

    def state_dict(self) -> Optional[Dict[str, Any]]:
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Args:
            samples_in_epoch (int): The number of samples processed so far in the current epoch.

        Returns:
            Optional[Dict[str, Any]]: The state, if a streaming dataset.
        """
        if isinstance(self.dataset, StreamingDataset):
            world = World.detect()
            num_samples = self.num_samples_yielded * world.num_ranks
            if self.dataset.replication is not None:
                # Check if we are using `replication`. If we are, then we need to adjust the
                # `num_samples_yielded` to reflect the fact that sample ids are shared across
                # `replication` consecutive devices. For example, if `replication` is 2, then the
                # number of samples seen is half the number of samples yielded, since every pair
                # of devices shares sample ids. So the index into the sample partition is halved.
                num_samples = num_samples // self.dataset.replication
            return self.dataset.state_dict(num_samples, False)
        return None

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        Args:
            obj (Dict[str, Any]): The state.
        """
        if isinstance(self.dataset, StreamingDataset):
            self.dataset.load_state_dict(obj)

    def __del__(self) -> None:
        """Terminate the workers during cleanup."""
        if self._iterator is not None:
            self._iterator._shutdown_workers()  # type: ignore [reportGeneralTypeIssues]
