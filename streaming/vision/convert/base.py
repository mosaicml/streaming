import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Optional, Union

from ...base.mds.writer import MDSCreator


def get_list_arg(text: str) -> List[str]:
    """Pass a list as a commandline flag.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: Splits, if any.
    """
    return text.split(',') if text else []


def convert_img_clf_dataset(
    dataset: Dataset,
    root: str,
    split: Optional[str] = None,
    compression: Optional[str] = None,
    hashes: Optional[List[str]] = None,
    limit: int = 1 << 24,
    progbar: bool = True,
    leave: bool = False,
    encoding: str = 'pil'
) -> None:
    """
    Convert an image classification Dataset to MDS.

    Args:
        dataset (Dataset): The dataset object to convert.
        root (str): Local dataset directory where shards are cached by split.
        split (Optional[str], default: None): Which dataset split to use, if any.
        compression (Optional[str], default: None): Optional compression.
        hashes (Optional[List[str]], default: None): List of hash or checksum algorithms to use to
            validate shards.
        limit (int, default: 1 << 26): Uncompressed shard size limit, at which point it flushes
            the shard and starts a new one.
        progbar (bool, default: True): Whether to display a progress bar while converting.
        leave (bool, default: False): Whether to leave the progress bar in the console when done.
        encoding (str, default: pil): MDS encoding to use for the image data.
    """
    split = split or ''
    fields = {
        'i': 'int',
        'x': encoding,
        'y': 'int',
    }
    hashes = hashes or []
    indices = np.random.permutation(len(dataset))  # pyright: ignore
    if progbar:
        indices = tqdm(indices, leave=leave)
    split_dir = os.path.join(root, split)
    with MDSCreator(split_dir, fields, compression, hashes, limit) as out:
        for i in indices:
            x, y = dataset[i]
            out.write({
                'i': i,
                'x': x,
                'y': y,
            })
