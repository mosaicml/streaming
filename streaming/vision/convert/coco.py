# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""COCO 2017 streaming dataset conversion scripts."""

import json
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
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
        help='Location of the input dataset',
    )
    args.add_argument(
        '--out_root',
        type=str,
        required=True,
        help='Location to store the output dataset',
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
        default=1 << 25,
        help='Shard size limit, after which point to start a new shard. Default: 1 << 25',
    )
    args.add_argument(
        '--progbar',
        type=int,
        default=1,
        help='tqdm progress bar. Default: 1 (Act as True)',
    )
    args.add_argument(
        '--leave',
        type=int,
        default=0,
        help='Keeps all traces of the progressbar upon termination of iteration. Default: 0 ' +
        '(Act as False)',
    )
    return args.parse_args()


class _COCODetection(Dataset):
    """PyTorch Dataset for the COCO dataset.

    Args:
        img_folder (str): Path to a COCO folder.
        annotate_file (str): Path to a file that contains image id, annotations (e.g., bounding
            boxes and object classes) etc.
    """

    def __init__(self, img_folder: str, annotate_file: str):
        self.img_folder = img_folder
        self.annotate_file = annotate_file

        # Start processing annotation
        with open(annotate_file) as fin:
            self.data = json.load(fin)

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        # 0 stands for the background
        cnt = 0
        self.label_info[cnt] = 'background'
        for cat in self.data['categories']:
            cnt += 1
            self.label_map[cat['id']] = cnt
            self.label_info[cnt] = cat['name']

        # build inference for images
        for img in self.data['images']:
            img_id = img['id']
            img_name = img['file_name']
            img_size = (img['height'], img['width'])
            if img_id in self.images:
                raise Exception('duplicated image record')
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data['annotations']:
            img_id = bboxes['image_id']
            bbox = bboxes['bbox']
            bbox_label = self.label_map[bboxes['category_id']]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())

    #@property
    def labelnum(self):
        return len(self.label_info)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)

        img = Image.open(img_path).convert('RGB')

        htot, wtot = img_data[1]
        bbox_sizes = []
        bbox_labels = []

        for (l, t, w, h), bbox_label in img_data[2]:
            r = l + w
            b = t + h
            bbox_size = (l / wtot, t / htot, r / wtot, b / htot)
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes)
        bbox_labels = torch.tensor(bbox_labels)

        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels


def each(dataset: _COCODetection, shuffle: bool) -> Iterable[Dict[str, bytes]]:
    """Generator over each dataset sample.

    Args:
        dataset (COCODetection): COCO detection dataset.
        shuffle (bool): Whether to shuffle the samples.

    Yields:
        Sample dicts.
    """
    if shuffle:
        indices = np.random.permutation(len(dataset))
    else:
        indices = np.arange(len(dataset))
    for idx in indices:
        _, img_id, (htot, wtot), bbox_sizes, bbox_labels = dataset[idx]

        img_id = dataset.img_keys[idx]
        img_data = dataset.images[img_id]
        img_basename = img_data[0]
        img_filename = os.path.join(dataset.img_folder, img_basename)
        img_bytes = open(img_filename, 'rb').read()

        yield {
            'img': img_bytes,
            'img_id': img_id,
            'htot': htot,
            'wtot': wtot,
            'bbox_sizes': bbox_sizes,  # (_, 4) float32.
            'bbox_labels': bbox_labels,  # int64.
        }


def main(args: Namespace) -> None:
    """Main: create COCO streaming dataset.

    Args:
        args (Namespace): command-line arguments.
    """
    fields = {
        'img': 'jpeg',
        'img_id': 'int',
        'htot': 'int',
        'wtot': 'int',
        'bbox_sizes': 'pkl',
        'bbox_labels': 'pkl'
    }

    for (split, expected_num_samples, shuffle) in [
        ('train', 117266, True),
        ('val', 4952, False),
    ]:
        split_out_dir = os.path.join(args.out_root, split)

        split_images_in_dir = os.path.join(args.in_root, f'{split}2017')
        if not os.path.exists(split_images_in_dir):
            raise FileNotFoundError(f'Images path does not exist: {split_images_in_dir}')
        split_annotations_in_file = os.path.join(args.in_root, 'annotations',
                                                 f'instances_{split}2017.json')
        if not os.path.exists(split_annotations_in_file):
            raise FileNotFoundError(
                f'Annotations file does not exist: {split_annotations_in_file}')
        dataset = _COCODetection(split_images_in_dir, split_annotations_in_file)

        if len(dataset) != expected_num_samples:
            raise ValueError(f'Number of samples in a dataset doesn\'t match. Expected ' +
                             f'{expected_num_samples}, but got {len(dataset)}')

        hashes = get_list_arg(args.hashes)

        if args.progbar:
            dataset = tqdm(each(dataset, shuffle), leave=args.leave, total=len(dataset))
        else:
            dataset = each(dataset, shuffle)

        with MDSWriter(split_out_dir, fields, args.compression, hashes, args.size_limit) as out:
            for sample in dataset:
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
