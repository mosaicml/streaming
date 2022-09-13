# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from streaming.vision.base import VisionDataset
from streaming.vision.transforms import (PadToSize, PhotometricDistoration, RandomCropPair,
                                         RandomHFlipPair, RandomResizePair)

IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class ADE20k(VisionDataset):
    """
    Implementation of the ADE20k dataset using streaming Dataset.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        shuffle (bool, optional): Whether to shuffle the train samples in this dataset. Default: ``True``.
        transforms (callable, optional): A function/transforms that takes in an image and a label and returns the transformed versions of both. Default: ``None``.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default: ``None``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: ``None``.
        prefetch (int, optional): Target number of samples remaining to prefetch while iterating. Default: ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when decompressing downloaded shards. If set to None, keep iff remote == local. Default: ``None``.
        retry (int, optional): Number of download re-attempts before giving up. Default: ``2``.
        timeout (float, optional): Number of seconds to wait for a shard to download before raising an exception. Default: ``60``.
        hash (str, optional): Hash or checksum algorithm to use to validate shards. Default: ``None``.
        batch_size (int, optional): Batch size that will be used on each device's DataLoader. Default: ``None``.
        min_resize_scale (float, optional): Minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float, optional): Maximum value the samples can be rescaled. Default: ``2.0``.
        base_size (int, optional): Initial size of the image and target before other augmentations. Default: ``512``.
        final_size (int, optional): Final size of the image and target. Default: ``512``.
    """

    def __init__(self,
                 local: str,
                 split: str,
                 remote: Optional[str] = None,
                 shuffle: bool = True,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 prefetch: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 retry: int = 2,
                 timeout: float = 60,
                 hash: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 min_resize_scale: float = 0.5,
                 max_resize_scale: float = 2.0,
                 base_size: int = 512,
                 final_size: int = 512):

        # Validation
        if base_size <= 0:
            raise ValueError('base_size must be positive.')
        if min_resize_scale <= 0:
            raise ValueError('min_resize_scale must be positive')
        if max_resize_scale <= 0:
            raise ValueError('max_resize_scale must be positive')
        if max_resize_scale < min_resize_scale:
            raise ValueError('max_resize_scale cannot be less than min_resize_scale')
        if final_size <= 0:
            raise ValueError('final_size must be positive')

        super().__init__(local, split, remote, shuffle, transforms, transform, target_transform,
                         prefetch, keep_zip, retry, timeout, hash, batch_size)

        r_mean, g_mean, b_mean = IMAGENET_CHANNEL_MEAN

        if split == 'train':
            if not self.transform:
                self.transform = torch.nn.Sequential(
                    RandomResizePair(min_scale=min_resize_scale,
                                     max_scale=max_resize_scale,
                                     base_size=(base_size, base_size)),
                    RandomCropPair(
                        crop_size=(final_size, final_size),
                        class_max_percent=0.75,
                        num_retry=10,
                    ),
                    RandomHFlipPair(),
                    # Photometric distoration values come from mmsegmentation:
                    # https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L861
                    PhotometricDistoration(brightness=32. / 255,
                                           contrast=0.5,
                                           saturation=0.5,
                                           hue=18. / 255),
                    PadToSize(size=(final_size, final_size),
                              fill=(int(r_mean), int(g_mean), int(b_mean))))
            if not self.target_transform:
                self.target_transform = torch.nn.Sequential(
                    RandomResizePair(min_scale=min_resize_scale,
                                     max_scale=max_resize_scale,
                                     base_size=(base_size, base_size)),
                    RandomCropPair(
                        crop_size=(final_size, final_size),
                        class_max_percent=0.75,
                        num_retry=10,
                    ), RandomHFlipPair(), PadToSize(size=(final_size, final_size), fill=0))
        else:
            if not self.transform:
                self.transform = T.Resize(size=(final_size, final_size),
                                          interpolation=TF.InterpolationMode.BILINEAR)
            if not self.target_transform:
                self.target_transform = T.Resize(size=(final_size, final_size),
                                                 interpolation=TF.InterpolationMode.NEAREST)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        obj = super().__getitem__(idx)
        x = obj['image']
        y = obj['annotation']
        if self.transforms:
            x, y = self.transforms((x, y))
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
