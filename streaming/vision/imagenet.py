# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet classification streaming dataset.

The most widely used dataset for Image Classification algorithms. Please refer to the `ImageNet
2012 Classification Dataset <http://image-net.org/>`_ for more details.
"""

from streaming.vision.base import ImageClassDataset

__all__ = ['ImageNet']


class ImageNet(ImageClassDataset):
    """Implementation of the ImageNet dataset using streaming Dataset.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored.
            Defaults to ``None``.
        split (str, optional): The dataset split to use, either 'train' or 'val'. Defaults to
            ``None``.
        shuffle (bool, optional): Whether to iterate over the samples in randomized order. Defaults to
            ``True``.
        transform (callable, optional): A function/transform that takes in an image and returns a
            transformed version. Defaults to ``None``.
        target_transform (callable, optional): A function/transform that takes in the target and
            transforms it. Defaults to ``None``.
        prefetch (int, optional): Target number of samples remaining to prefetch while iterating.
            Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when decompressing
            downloaded shards. If set to None, keep iff remote is local. Defaults to ``None``.
        retry (int, optional): Number of download re-attempts before giving up. Defaults to ``2``.
        timeout (float, optional): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        hash (str, optional): Hash or checksum algorithm to use to validate shards. Defaults to
            ``None``.
        batch_size (int, optional): Hint the batch size that will be used on each device's DataLoader.
            Defaults to ``None``.
    """
