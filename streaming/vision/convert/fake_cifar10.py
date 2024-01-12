# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""fake cifar 10 module."""
from argparse import ArgumentParser, Namespace

import numpy as np
from PIL import Image

from streaming.vision.convert.base import convert_image_class_dataset


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Args:
        Namespace: command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--out', type=str, default='/tmp/fake_cifar10/')
    args.add_argument('--num_train', type=int, default=8 * 8 * 16384)
    args.add_argument('--num_val', type=int, default=8 * 8 * 2048)
    return args.parse_args()


def make_split(root: str, split: str, count: int) -> None:
    """Process one split of the fake CIFAR10 dataset.

    Args:
        root (str): Output root directory.
        split (str): Dataset split name.
        count (int): Number of samples to create.
    """
    x = np.random.randint(0, 255, (count, 32, 32, 3), np.uint8)
    y = np.random.randint(0, 9, count)
    dataset = list(zip(map(Image.fromarray, x), y))
    convert_image_class_dataset(dataset=dataset, out_root=root, split=split)  # pyright: ignore


def main(args: Namespace) -> None:
    """Main: create streaming fake CIFAR10 dataset.

    Args:
        args (Namespace): command-line arguments.
    """
    make_split(args.out, 'train', args.num_train)
    make_split(args.out, 'val', args.num_val)


if __name__ == '__main__':
    main(parse_args())
