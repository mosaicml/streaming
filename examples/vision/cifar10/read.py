# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR-10 classification streaming dataset.

It is one of the most widely used datasets for machine learning research. Please refer to the
`CIFAR-10 Dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more details.
"""

from typing import Any, Dict

from streaming.vision import StreamingVisionDataset

__all__ = ['StreamingCIFAR10']


class StreamingCIFAR10(StreamingVisionDataset):
    """Implementation of the CIFAR-10 dataset using StreamingDataset.

    Args:
        **kwargs (Dict[str, Any]): Keyword arguments.
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)
