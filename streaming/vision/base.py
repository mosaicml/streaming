# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Base classes for computer vision :class:`StreamingDataset`s."""

from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import to_tensor

from streaming.base import StreamingDataset

__all__ = ['StreamingVisionDataset', 'StreamingImageClassDataset']


class StandardTransform:
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


class StreamingVisionDataset(StreamingDataset, VisionDataset):
    """A streaming, iterable, torchvision VisionDataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to
            ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        transforms (callable, optional): A function/transforms that takes in an image and a label
            and returns the transformed versions of both. Defaults to ``None``.
        transform (callable, optional): A function/transform that takes in an image and returns a
            transformed version. Defaults to ``None``.
        target_transform (callable, optional): A function/transform that takes in the target and
            transforms it. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            Defaults to ``None``, which is interpreted as the number of nodes of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 predownload: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 shuffle_seed: int = 9176,
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None):
        super().__init__(local, remote, split, shuffle, predownload, keep_zip, download_retry,
                         download_timeout, validate_hash, shuffle_seed, num_canonical_nodes,
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


class StreamingImageClassDataset(StreamingVisionDataset):
    """A streaming, iterable, image classification dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to
            ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        transform (callable, optional): A function/transform that takes in an image and returns a
            transformed version. Defaults to ``None``.
        target_transform (callable, optional): A function/transform that takes in the target and
            transforms it. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            Defaults to ``None``, which is interpreted as the number of nodes of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 predownload: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 shuffle_seed: int = 9176,
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None):
        transforms = None
        super().__init__(local, remote, split, shuffle, transforms, transform, target_transform,
                         predownload, keep_zip, download_retry, download_timeout, validate_hash,
                         shuffle_seed, num_canonical_nodes, batch_size)
