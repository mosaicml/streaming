# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions to convert CV datasets."""

import os
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from streaming.base import MDSWriter


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
