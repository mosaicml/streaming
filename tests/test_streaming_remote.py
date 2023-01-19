# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
import time
from typing import Any, Dict, Optional, Tuple

import pytest

from streaming.base import StreamingDataset
from streaming.text import StreamingC4
from streaming.vision import StreamingADE20K, StreamingCIFAR10, StreamingCOCO, StreamingImageNet


def get_dataset(name: str,
                local: str,
                split: str,
                shuffle: bool,
                batch_size: Optional[int],
                other_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[int, StreamingDataset]:
    other_kwargs = {} if other_kwargs is None else other_kwargs
    dataset_map = {
        'ade20k': {
            'remote': 's3://mosaicml-internal-dataset-ade20k/mds/2/',
            'num_samples': {
                'train': 20206,
                'val': 2000,
            },
            'class': StreamingADE20K,
            'kwargs': {},
        },
        'imagenet1k': {
            'remote': 's3://mosaicml-internal-dataset-imagenet1k/mds/2/',
            'num_samples': {
                'train': 1281167,
                'val': 50000,
            },
            'class': StreamingImageNet,
            'kwargs': {},
        },
        'coco': {
            'remote': 's3://mosaicml-internal-dataset-coco/mds/2/',
            'num_samples': {
                'train': 117266,
                'val': 4952,
            },
            'class': StreamingCOCO,
            'kwargs': {},
        },
        'c4': {
            'remote': 's3://mosaicml-internal-dataset-c4/mds/2/',
            'num_samples': {
                'train': 364868892,
                'val': 364608,
            },
            'class': StreamingC4,
            'kwargs': {
                'tokenizer_name': 'bert-base-uncased',
                'max_seq_len': 512,
                'group_method': 'truncate'
            },
        },
        'cifar10': {
            'remote': 's3://mosaicml-internal-dataset-cifar10/mds/2/',
            'num_samples': {
                'train': 50000,
                'val': 10000,
            },
            'class': StreamingCIFAR10,
            'kwargs': {},
        },
        'test_streaming_upload': {
            'remote': 's3://streaming-upload-test-bucket/',
            'num_samples': {
                'all': 0,
            },
            'class': StreamingDataset,
            'kwargs': {},
        }
    }
    if name not in dataset_map and split not in dataset_map[name]['num_samples'][split]:
        raise ValueError('Could not load dataset with name={name} and split={split}')

    d = dataset_map[name]
    expected_samples = d['num_samples'][split]
    remote = d['remote']
    kwargs = {**d['kwargs'], **other_kwargs}
    dataset = d['class'](local=local,
                         remote=remote,
                         split=split,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         **kwargs)
    return (expected_samples, dataset)


@pytest.mark.remote
@pytest.mark.parametrize('name', [
    'ade20k',
    'imagenet1k',
    'coco',
    'cifar10',
    'c4',
])
@pytest.mark.parametrize('split', ['val'])
def test_streaming_remote_dataset(tmp_path: pathlib.Path, name: str, split: str) -> None:
    # Build StreamingDataset
    build_start = time.time()
    expected_samples, dataset = get_dataset(name=name,
                                            local=str(tmp_path),
                                            split=split,
                                            shuffle=False,
                                            batch_size=None)
    build_end = time.time()
    build_dur = build_end - build_start
    print('Built dataset')

    # Test basic iteration
    rcvd_samples = 0
    iter_start = time.time()
    for _ in dataset:
        rcvd_samples += 1

        if (rcvd_samples % 1000 == 0):
            print(f'samples read: {rcvd_samples}')

    iter_end = time.time()
    iter_dur = iter_end - iter_start
    samples_per_sec = rcvd_samples / iter_dur

    # Print debug info
    print(
        f'build_dur={build_dur:.2f}s, iter_dur={iter_dur:.2f}, samples_per_sec={samples_per_sec:.2f}'
    )

    # Test all samples arrived
    assert rcvd_samples == expected_samples
