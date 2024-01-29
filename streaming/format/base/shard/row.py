# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming shard abstract base classes."""

from abc import abstractmethod
from typing import Any, Dict

from streaming.format.base.shard.base import Shard

__all__ = ['RowShard']


class RowShard(Shard):
    """A Shard whose samples are stored contiguously, not columnar."""

    @abstractmethod
    def get_sample_data(self, index: int) -> bytes:
        """Get the raw sample data at the index.

        Args:
            index (int): Sample index.

        Returns:
            bytes: Sample data.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_sample(self, data: bytes) -> Dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        raise NotImplementedError

    def get_item(self, index: int) -> Dict[str, Any]:
        """Get the sample at the index.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        data = self.get_sample_data(index)
        return self.decode_sample(data)
