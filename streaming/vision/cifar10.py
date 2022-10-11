# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR-10 classification streaming dataset.

It is one of the most widely used datasets for machine learning research. Please refer to the
`CIFAR-10 Dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more details.
"""

from streaming.vision.base import ImageClassDataset, LocalResumableImageClassDataset


class CIFAR10(ImageClassDataset):
    """Implementation of the CIFAR-10 dataset using streaming Dataset.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored.
            Defaults to ``None``.
        split (str, optional): The dataset split to use, either 'train' or 'val'. Defaults to
            ``None``.
        shuffle (bool, optional): Whether to shuffle the train samples in this dataset. Defaults to
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
        batch_size (int, optional): Batch size that will be used on each device's DataLoader.
            Defaults to ``None``.
    """


class LocalResumableCIFAR10(LocalResumableImageClassDataset):
    """Implementation of the CIFAR-10 dataset using streaming Dataset.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str, optional): The dataset split to use, either 'train' or 'val'. Defaults to
            ``None``.
        shuffle (bool, optional): Whether to shuffle the train samples in this dataset. Defaults to
            ``True``.
        transform (callable, optional): A function/transform that takes in an image and returns a
            transformed version. Defaults to ``None``.
        target_transform (callable, optional): A function/transform that takes in the target and
            transforms it. Defaults to ``None``.
    """
