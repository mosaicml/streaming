# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported CV datasets."""

from joshua.vision.ade20k import StreamingADE20K as StreamingADE20K
from joshua.vision.cifar10 import StreamingCIFAR10 as StreamingCIFAR10
from joshua.vision.coco import StreamingCOCO as StreamingCOCO
from joshua.vision.imagenet import StreamingImageNet as StreamingImageNet

__all__ = ['StreamingADE20K', 'StreamingCIFAR10', 'StreamingCOCO', 'StreamingImageNet']
