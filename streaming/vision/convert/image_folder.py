# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from ...base import MDSWriter
from .base import get_list_arg


def parse_args() -> Namespace:
    """Parse commandline arguments.

    Args:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in', type=str, default='/datasets/imagenet_files/')
    args.add_argument('--out', type=str, default='/datasets/mds/imagenet/')
    args.add_argument('--splits', type=str, default='train,val')
    args.add_argument('--compression', type=str, default='')
    args.add_argument('--hashes', type=str, default='sha1,xxh64')
    args.add_argument('--size_limit', type=int, default=1 << 26)
    args.add_argument('--progbar', type=int, default=1)
    args.add_argument('--leave', type=int, default=0)
    args.add_argument('--validate', type=int, default=1)
    args.add_argument('--extensions', type=str, default='jpeg')
    return args.parse_args()


def check_extensions(filenames: List[str], extensions: Set[str]) -> None:
    """Validate filename extensions.

    Args:
        filenames (List[str]): List of files.
        extensions (Set[str]): Acceptable extensions.
    """
    for f in filenames:
        idx = f.rindex('.')
        ext = f[idx + 1:]
        assert ext.lower() in extensions


def get_classes(filenames: List[str],
                class_names: Optional[List[str]] = None) -> Tuple[List[int], List[str]]:
    """Get the classes for a dataset split of sample image filenames.

    Args:
        filenames (List[str]): Files, in the format "root/split/class/sample.jpeg".
        class_names (Optional[List[str]], default: None): List of class names from the other splits
            that we must macth.

    Returns:
        Tuple[List[int], List[str]]: Class ID per sample, and the list of unique class names.
    """
    classes = []
    dirname2class = {}
    for f in filenames:
        d = f.split(os.path.sep)[-2]
        c = dirname2class.get(d)
        if c is None:
            c = len(dirname2class)
            dirname2class[d] = c
        classes.append(c)
    new_class_names = sorted(dirname2class)
    if class_names is not None:
        assert class_names == new_class_names
    return classes, new_class_names


def main(args: Namespace) -> None:
    """Main: create streaming CIFAR10 dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    splits = get_list_arg(args.splits)
    columns = {'i': 'int', 'x': 'jpeg', 'y': 'int'}
    hashes = get_list_arg(args.hashes)
    extensions = set(get_list_arg(args.extensions))
    class_names = None
    for split in splits:
        pattern = os.path.join(getattr(args, 'in'), split, '*', '*')
        filenames = sorted(glob(pattern))
        check_extensions(filenames, extensions)
        classes, class_names = get_classes(filenames, class_names)
        indices = np.random.permutation(len(filenames))
        if args.progbar:
            indices = tqdm(indices, leave=args.leave)
        split_dir = os.path.join(args.out, split)
        with MDSWriter(split_dir, columns, args.compression, hashes, args.size_limit) as out:
            for i in indices:
                if args.validate:
                    x = Image.open(filenames[i])
                x = open(filenames[i], 'rb').read()
                y = classes[i]
                out.write({
                    'i': i,
                    'x': x,
                    'y': y,
                })


if __name__ == '__main__':
    main(parse_args())
