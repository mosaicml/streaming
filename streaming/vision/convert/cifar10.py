# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR10 streaming dataset conversion scripts."""

from argparse import ArgumentParser, Namespace

from torchvision.datasets import CIFAR10

from streaming.base.util import get_list_arg
from streaming.vision.convert.base import convert_image_class_dataset


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Args:
        Namespace: command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--in_root',
        type=str,
        required=True,
        help='Directory path of the input dataset',
    )
    args.add_argument(
        '--out_root',
        type=str,
        required=True,
        help='Directory path to store the output dataset',
    )
    args.add_argument(
        '--splits',
        type=str,
        default='train,val',
        help='Split to use. Default: train,val',
    )
    args.add_argument(
        '--compression',
        type=str,
        default='',
        help='Compression algorithm to use. Default: None',
    )
    args.add_argument(
        '--hashes',
        type=str,
        default='sha1,xxh64',
        help='Hashing algorithms to apply to shard files. Default: sha1,xxh64',
    )
    args.add_argument(
        '--size_limit',
        type=int,
        default=1 << 20,
        help='Shard size limit, after which point to start a new shard. Default: 1 << 20',
    )
    args.add_argument(
        '--progbar',
        type=int,
        default=1,
        help='tqdm progress bar. Default: 1 (True)',
    )
    args.add_argument(
        '--leave',
        type=int,
        default=0,
        help='Keeps all traces of the progressbar upon termination of iteration. Default: 0 ' +
        '(False)',
    )
    return args.parse_args()


def main(args: Namespace) -> None:
    """Main: create streaming CIFAR10 dataset.

    Args:
        args (Namespace): command-line arguments.
    """
    splits = get_list_arg(args.splits)
    hashes = get_list_arg(args.hashes)
    for split in splits:
        dataset = CIFAR10(root=args.in_root, train=(split == 'train'), download=True)
        convert_image_class_dataset(dataset, args.out_root, split, args.compression, hashes,
                                    args.size_limit, args.progbar, args.leave, 'pil')


if __name__ == '__main__':
    main(parse_args())
