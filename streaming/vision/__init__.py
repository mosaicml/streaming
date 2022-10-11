# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported CV datasets."""

from streaming.vision.ade20k import ADE20K as ADE20K
from streaming.vision.cifar10 import CIFAR10 as CIFAR10
from streaming.vision.cifar10 import LocalResumableCIFAR10 as LocalResumableCIFAR10
from streaming.vision.coco import COCO as COCO
from streaming.vision.imagenet import ImageNet as ImageNet

__all__ = ['ADE20K', 'CIFAR10', 'COCO', 'ImageNet']
