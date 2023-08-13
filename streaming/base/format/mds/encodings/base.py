# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Store data in MDS."""

from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = ['Encoding']


class Encoding(ABC):
    """Encodes and decodes between objects of a certain type and raw bytes."""

    size: Optional[int] = None  # Fixed size in bytes of encoded data (None if variable size).

    @abstractmethod
    def encode(self, obj: Any) -> bytes:
        """Encode the given data from the original object to bytes.

        Args:
            obj (Any): Decoded object.

        Returns:
            bytes: Encoded data.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, data: bytes) -> Any:
        """Decode the given data from bytes to the original object.

        Args:
            data (bytes): Encoded data.

        Returns:
            Any: Decoded object.
        """
        raise NotImplementedError

    @staticmethod
    def _validate(data: Any, expected_type: Any) -> None:
        if not isinstance(data, expected_type):
            raise AttributeError(
                f'data should be of type {expected_type}, but instead, found as {type(data)}')
