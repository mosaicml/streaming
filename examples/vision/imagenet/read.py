# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet classification streaming dataset.

The most widely used dataset for Image Classification algorithms. Please refer to the `ImageNet
2012 Classification Dataset <http://image-net.org/>`_ for more details.
"""

from streaming.vision.base import StreamingVisionDataset

__all__ = ['StreamingImageNet']


class StreamingImageNet(StreamingVisionDataset):
    """Implementation of the ImageNet dataset using StreamingDataset.

    No custom work is needed.
    """
