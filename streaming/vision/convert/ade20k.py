# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K streaming dataset conversion scripts."""

import os
import random
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from streaming.base import MDSWriter
from streaming.base.util import get_list_arg


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
        default=1 << 22,
        help='Shard size limit, after which point to start a new shard. Default: 1 << 22',
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


def get(in_root: str, split: str, shuffle: bool) -> List[Tuple[str, str, str]]:
    """Collect the samples for this dataset split.

    Args:
        in_root (str): Input dataset directory.
        split (str): Split name.
        shuffle (bool): Whether to shuffle the samples before writing.

    Returns:
        List of samples of (uid, image_filename, annotation_filename).
    """
    # Get uids
    if split not in ('train', 'val'):
        raise ValueError(f"Split must be one of 'train', 'val', not {split}")
    subdir = 'training' if split == 'train' else 'validation'
    split_images_in_dir = os.path.join(in_root, 'images', subdir)
    if not os.path.exists(split_images_in_dir):
        raise FileNotFoundError(f'Images path does not exist: {split_images_in_dir}')
    split_annotations_in_dir = os.path.join(in_root, 'annotations', subdir)
    if not os.path.exists(split_annotations_in_dir):
        raise FileNotFoundError(f'Annotations path does not exist: {split_annotations_in_dir}')
    image_glob_pattern = os.path.join(split_images_in_dir, f'ADE_{split}_*.jpg')
    images = sorted(glob(image_glob_pattern))
    uids = [s.strip('.jpg')[-8:] for s in images]

    # Remove some known corrupted uids from 'train' split
    if split == 'train':
        corrupted_uids = ['00003020', '00001701', '00013508', '00008455']
        uids = [uid for uid in uids if uid not in corrupted_uids]

    # Create samples
    samples = [(uid, os.path.join(split_images_in_dir, f'ADE_{split}_{uid}.jpg'),
                os.path.join(split_annotations_in_dir, f'ADE_{split}_{uid}.png')) for uid in uids]

    # Optionally shuffle samples at dataset creation for extra randomness
    if shuffle:
        random.shuffle(samples)

    return samples


def each(samples: Iterable[Tuple[str, str, str]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        samples (list): List of samples of (uid, image_filename, annotation_filename).

    Yields:
        Sample dicts.
    """
    for (uid, image_file, annotation_file) in samples:
        uid = uid.encode('utf-8')
        image = open(image_file, 'rb').read()
        annotation = open(annotation_file, 'rb').read()
        yield {
            'uid': uid,
            'x': image,
            'y': annotation,
        }


def main(args: Namespace) -> None:
    """Main: create streaming ADE20K dataset.

    Args:
        args (Namespace): command-line arguments.
    """
    fields = {'uid': 'bytes', 'x': 'jpeg', 'y': 'png'}

    for (split, expected_num_samples, shuffle) in [
        ('train', 20206, True),
        ('val', 2000, False),
    ]:
        # Get samples
        samples = get(in_root=args.in_root, split=split, shuffle=shuffle)
        if len(samples) != expected_num_samples:
            raise ValueError(f'Number of samples in a dataset doesn\'t match. Expected ' +
                             f'{expected_num_samples}, but got {len(samples)}')

        split_images_out_dir = os.path.join(args.out_root, split)
        hashes = get_list_arg(args.hashes)

        if args.progbar:
            samples = tqdm(samples, leave=args.leave)

        with MDSWriter(split_images_out_dir, fields, args.compression, hashes,
                       args.size_limit) as out:
            for sample in each(samples):
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
