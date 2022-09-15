# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Tuple

from streaming.base import Dataset


class ADE20k(Dataset):
    """Implementation of the ADE20k dataset using streaming Dataset.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``None``.
        split (str, optional): The dataset split to use, either 'train' or 'val'. Default:
            ``None``.
        shuffle (bool, optional): Whether to shuffle the train samples in this dataset. Default:
            ``True``.
        both_transforms (callable, optional): A function/transforms that takes in an image and a
            label and returns the transformed versions of both. Default: ``None``.
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

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 both_transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 prefetch: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 retry: int = 2,
                 timeout: float = 60,
                 hash: Optional[str] = None,
                 batch_size: Optional[int] = None):
        super().__init__(local, remote, split, shuffle, prefetch, keep_zip, retry, timeout, hash,
                         batch_size)
        self.both_transforms = both_transforms
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        obj = super().__getitem__(idx)
        x = obj['x']
        y = obj['y']
        if self.both_transforms:
            x, y = self.both_transforms((x, y))
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
