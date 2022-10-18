# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K Semantic segmentation and scene parsing dataset.

Please refer to the `ADE20K dataset <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ for more details about this
dataset.
"""

from typing import Any, Callable, Optional, Tuple

from streaming.base import StreamingDataset


class ADE20K(StreamingDataset):
    """Implementation of the ADE20K dataset using StreamingDataset.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored.
            Defaults to ``None``.
        split (str, optional): The dataset split to use, either 'train' or 'val'. Defaults to
            ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``True``.
        prefetch (int, optional): Target number of samples remaining to prefetch while iterating.
            Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when decompressing
            downloaded shards. If set to None, keep iff remote is local. Defaults to ``None``.
        retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        timeout (float):Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        hash (str, optional): Hash or checksum algorithm to use to validate shards. Defaults to
            ``None``.
        batch_size (int, optional): Hint the batch size that will be used on each device's DataLoader.
            Defaults to ``None``.
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
            Tuple[Any, Any]: Sample data and label.
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
