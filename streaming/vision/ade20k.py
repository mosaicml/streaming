# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms

from streaming.base import Dataset
from streaming.vision.transform import (PadToSize, PhotometricDistoration, RandomCropPair,
                                        RandomHFlipPair, RandomResizePair)

IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class ADE20k(Dataset):
    """
    Implementation of the ADE20k dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        base_size (int): initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): the minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): the maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): the final size of the image and target. Default: ``512``.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    # def decode_uid(self, data: bytes) -> str:
    #     return data.decode('utf-8')

    # def decode_image(self, data: bytes) -> Image.Image:
    #     return Image.open(BytesIO(data))

    # def decode_annotation(self, data: bytes) -> Image.Image:
    #     return Image.open(BytesIO(data))

    # def __init__(self,
    #              remote: str,
    #              local: str,
    #              split: str,
    #              shuffle: bool,
    #              base_size: int = 512,
    #              min_resize_scale: float = 0.5,
    #              max_resize_scale: float = 2.0,
    #              final_size: int = 512,
    #              batch_size: Optional[int] = None):

    # TODO: Rearrange the args
    def __init__(
            self,
            local: str,
            remote: Optional[str] = None,
            split: Optional[str] = None,
            shuffle: bool = True,
            prefetch: Optional[int] = 100_000,  #TODO
            keep_zip: Optional[bool] = None,  #TODO
            retry: int = 2,
            timeout: float = 60,
            hash: Optional[str] = None,
            batch_size: Optional[int] = None,
            base_size: int = 512,
            min_resize_scale: float = 0.5,
            max_resize_scale: float = 2.0,
            final_size: int = 512):

        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")
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

        # Build StreamingDataset
        # decoders = {
        #     'image': self.decode_image,
        #     'annotation': self.decode_annotation,
        # }
        # super().__init__(remote=os.path.join(remote, split),
        #                  local=os.path.join(local, split),
        #                  shuffle=shuffle,
        #                  decoders=decoders,
        #                  batch_size=batch_size)
        super().__init__(local, remote, split, shuffle, prefetch, keep_zip, retry, timeout, hash,
                         batch_size)

        # Define custom transforms
        if split == 'train':
            self.both_transform = torch.nn.Sequential(
                RandomResizePair(min_scale=min_resize_scale,
                                 max_scale=max_resize_scale,
                                 base_size=(base_size, base_size)),
                RandomCropPair(
                    crop_size=(final_size, final_size),
                    class_max_percent=0.75,
                    num_retry=10,
                ),
                RandomHFlipPair(),
            )

            # Photometric distoration values come from mmsegmentation:
            # https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L861
            r_mean, g_mean, b_mean = IMAGENET_CHANNEL_MEAN
            self.image_transform = torch.nn.Sequential(
                PhotometricDistoration(brightness=32. / 255,
                                       contrast=0.5,
                                       saturation=0.5,
                                       hue=18. / 255),
                PadToSize(size=(final_size, final_size),
                          fill=(int(r_mean), int(g_mean), int(b_mean))))

            self.annotation_transform = PadToSize(size=(final_size, final_size), fill=0)
        else:
            self.both_transform = None
            self.image_transform = transforms.Resize(size=(final_size, final_size),
                                                     interpolation=TF.InterpolationMode.BILINEAR)
            self.annotation_transform = transforms.Resize(
                size=(final_size, final_size), interpolation=TF.InterpolationMode.NEAREST)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        obj = super().__getitem__(idx)
        x = obj['image']
        y = obj['annotation']
        if self.both_transform:
            x, y = self.both_transform((x, y))
        if self.image_transform:
            x = self.image_transform(x)
        if self.annotation_transform:
            y = self.annotation_transform(y)
        return x, y
