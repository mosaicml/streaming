# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet classification streaming dataset.

The most widely used dataset for Image Classification algorithms. Please refer to the `ImageNet
2012 Classification Dataset <http://image-net.org/>`_ for more details.
"""

from typing import Any, Dict

from streaming.modality.vision import StreamingVisionDataset

__all__ = ['StreamingImageNet']


class StreamingImageNet(StreamingVisionDataset):
    """Implementation of the ImageNet dataset using StreamingDataset.

    Args:
        **kwargs (Dict[str, Any]): Keyword arguments.
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)
