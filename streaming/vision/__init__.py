# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported CV datasets."""

from streaming.vision.ade20k import StreamingADE20K as StreamingADE20K
from streaming.vision.cifar10 import StreamingCIFAR10 as StreamingCIFAR10
from streaming.vision.coco import StreamingCOCO as StreamingCOCO
from streaming.vision.imagenet import StreamingImageNet as StreamingImageNet

__all__ = ['StreamingADE20K', 'StreamingCIFAR10', 'StreamingCOCO', 'StreamingImageNet']
