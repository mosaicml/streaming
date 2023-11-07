# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Base classes for computer vision :class:`StreamingDataset`s."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from streaming import MDSWriter, StreamingDataset

__all__ = ['StreamingVisionDataset']


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
    """A streaming, iterable, torchvision VisionDataset."""

    def __init__(self,
                 *,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs: Dict[str, Any]) -> None:
        StreamingDataset.__init__(self, **kwargs)

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

    def get_item(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        obj = super().get_item(idx)
        x = obj['x']
        y = obj['y']
        return self.transforms(x, y)


def convert_image_class_dataset(dataset: Dataset,
                                out_root: str,
                                split: Optional[str] = None,
                                compression: Optional[str] = None,
                                hashes: Optional[List[str]] = None,
                                size_limit: int = 1 << 24,
                                progress_bar: bool = True,
                                leave: bool = False,
                                encoding: str = 'pil') -> None:
    """Convert an image classification Dataset.

    Args:
        dataset (Dataset): The dataset object to convert.
        out_root (str): Output directory where shards are cached by split.
        remote (str, optional): Remote dataset directory where shards are uploaded by split.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        compression (str, optional): Optional compression. Defaults to ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int): Uncompressed shard size limit, at which point it flushes the shard and
            starts a new one. Defaults to ``1 << 26``.
        progress_bar (bool): Whether to display a progress bar while converting.
            Defaults to ``True``.
        leave (bool): Whether to leave the progress bar in the console when done. Defaults to
            ``False``.
        encoding (str): MDS encoding to use for the image data. Defaults to ``pil``.
    """
    split = split or ''
    columns = {
        'i': 'int',
        'x': encoding,
        'y': 'int',
    }
    hashes = hashes or []
    indices = np.random.permutation(len(dataset)).tolist()  # pyright: ignore
    if progress_bar:
        indices = tqdm(indices, leave=leave)

    out_split_dir = os.path.join(out_root, split)

    with MDSWriter(out=out_split_dir,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit,
                   progress_bar=progress_bar) as out:
        for i in indices:
            x, y = dataset[i]
            out.write({
                'i': i,
                'x': x,
                'y': y,
            })
