# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from streaming.vision.base import ImageClassDataset


class ImageNet(ImageClassDataset):
    """Streaming ImageNet.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``None``.
        split (str, optional): The dataset split to use, either 'train' or 'val'. Default:
            ``None``.
        shuffle (bool, optional): Whether to shuffle the train samples in this dataset. Default:
            ``True``.
        transform (callable, optional): A function/transform that takes in an image and returns a
            transformed version. Default: ``None``.
        target_transform (callable, optional): A function/transform that takes in the target and
            transforms it. Default: ``None``.
        prefetch (int, optional): Target number of samples remaining to prefetch while iterating.
            Default: ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when decompressing
            downloaded shards. If set to None, keep iff remote is local. Default: ``None``.
        retry (int, optional): Number of download re-attempts before giving up. Default: ``2``.
        timeout (float, optional): Number of seconds to wait for a shard to download before raising
            an exception. Default: ``60``.
        hash (str, optional): Hash or checksum algorithm to use to validate shards. Default:
            ``None``.
        batch_size (int, optional): Batch size that will be used on each device's DataLoader.
            Default: ``None``.
    """
