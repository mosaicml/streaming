# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Base Class for making Computer Vision datasets which are compatible with :class:`Dataset`."""

from typing import Any, Callable, Optional, Tuple

from torchvision.transforms.functional import to_tensor

from streaming.base import Dataset

__all__ = ['VisionDataset', 'ImageClassDataset']


class StandardTransform(object):
    """Individual input and output transforms called jointly, following torchvision.

    Args:
        transform (Callable, optional): Input transform. Defaults to ``None``.
        target_transform (Callable, optional): Output transform. Defaults to ``None``.
    """

    def __init__(self,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, x: Any, y: Any) -> Tuple[Any, Any]:
        """Apply the transforms to input and output.

        Args:
            x (Any): Input.
            y (Any): Output.

        Returns:
            Tuple[Any, Any]: Transformed input and output.
        """
        if self.transform:
            x = self.transform(x)
        else:
            x = to_tensor(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class VisionDataset(Dataset):
    """Base Class for creating a Vision streaming datasets.

    Args:
        local (str): Local filesystem directory where dataset is cached during operation.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored.
            Defaults to ``None``.
        split (str, optional): The dataset split to use, either 'train' or 'val'. Defaults to
            ``None``.
        shuffle (bool, optional): Whether to shuffle the train samples in this dataset. Defaults to
            ``True``.
        transforms (callable, optional): A function/transforms that takes in an image and a label
            and returns the transformed versions of both. Default to ``None``.
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

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 prefetch: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = True,
                 retry: int = 2,
                 timeout: float = 60,
                 hash: Optional[str] = None,
                 batch_size: Optional[int] = None) -> None:
        super().__init__(local, remote, split, shuffle, prefetch, keep_zip, retry, timeout, hash,
                         batch_size)

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                'Only transforms or transform/target_transform can be passed as an argument')

        self.transform = transform
        self.target_transform = target_transform
        if not has_transforms:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        obj = super().__getitem__(idx)
        x = obj['x']
        y = obj['y']
        return self.transforms(x, y)


class ImageClassDataset(VisionDataset):
    """Base Class for creating an Image Classification streaming datasets.

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

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 prefetch: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = True,
                 retry: int = 2,
                 timeout: float = 60,
                 hash: Optional[str] = None,
                 batch_size: Optional[int] = None) -> None:
        super().__init__(local, remote, split, shuffle, None, transform, target_transform,
                         prefetch, keep_zip, retry, timeout, hash, batch_size)
