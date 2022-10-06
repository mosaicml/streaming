# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""COCO (Common Objects in Context) dataset.

COCO is a large-scale object detection, segmentation, and captioning dataset. Please refer to the
`COCO dataset <https://cocodataset.org>`_ for more details.
"""

from typing import Any, Callable, Optional

from streaming.base import Dataset


class COCO(Dataset):
    """Implementation of the COCO dataset using streaming Dataset.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored.
            Defaults to ``None``.
        split (str, optional): The dataset split to use, either 'train' or 'val'. Defaults to
            ``None``.
        shuffle (bool, optional): Whether to shuffle the train samples in this dataset. Defaults to
            ``True``.
        transform (callable, optional): A function/transform that takes in an image and bboxes and
            returns a transformed version. Defaults to ``None``.
        prefetch (int, optional): Target number of samples remaining to prefetch while iterating.
            Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when decompressing
            downloaded shards. If set to None, keep iff remote == local. Defaults to ``None``.
        retry (int, optional): Number of download re-attempts before giving up. Defaults to ``2``.
        timeout (float, optional): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        hash (str, optional): Hash or checksum algorithm to use to validate shards. Defaults to
            ``None``.
        batch_size (int, optional): Batch size that will be used on each device's DataLoader.
            Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 transform: Optional[Callable] = None,
                 prefetch: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 retry: int = 2,
                 timeout: float = 60,
                 hash: Optional[str] = None,
                 batch_size: Optional[int] = None):
        super().__init__(local, remote, split, shuffle, prefetch, keep_zip, retry, timeout, hash,
                         batch_size)

        self.transform = transform

    def __getitem__(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        x = super().__getitem__(idx)
        img = x['img'].convert('RGB')
        img_id = x['img_id']
        htot = x['htot']
        wtot = x['wtot']
        bbox_sizes = x['bbox_sizes']
        bbox_labels = x['bbox_labels']
        if self.transform:
            img, (htot,
                  wtot), bbox_sizes, bbox_labels = self.transform(img, (htot, wtot), bbox_sizes,
                                                                  bbox_labels)
        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels
