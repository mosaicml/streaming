# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace

from torchvision.datasets import CIFAR10

from streaming.base.util import get_list_arg
from streaming.vision.convert.base import convert_image_class_dataset


def parse_args() -> Namespace:
    """Parse commandline arguments.

    Args:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--splits', type=str, default='train,val')
    args.add_argument('--compression', type=str, default='')
    args.add_argument('--hashes', type=str, default='sha1,xxh64')
    args.add_argument('--limit', type=int, default=1 << 20)
    args.add_argument('--progbar', type=int, default=1)
    args.add_argument('--leave', type=int, default=0)
    return args.parse_args()


def main(args: Namespace) -> None:
    """Main: create streaming CIFAR10 dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    splits = get_list_arg(args.splits)
    hashes = get_list_arg(args.hashes)
    for split in splits:
        dataset = CIFAR10(root=args.in_root, train=(split == 'train'), download=True)
        convert_image_class_dataset(dataset, args.out_root, split, args.compression, hashes,
                                    args.limit, args.progbar, args.leave, 'pil')


if __name__ == '__main__':
    main(parse_args())
