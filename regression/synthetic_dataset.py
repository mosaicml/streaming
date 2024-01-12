# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create a dataset and save it to a directory."""

import ast
import shutil
import sys
import urllib.parse
from argparse import ArgumentParser, Namespace
from typing import Any, Sequence

import numpy as np
import torch
from utils import delete_gcs, delete_oci, delete_s3, get_kwargs, get_writer_params

from streaming.base import MDSWriter

_DATASET_MAP = {
    'sequencedataset': 'SequenceDataset',
    'numberandsaydataset': 'NumberAndSayDataset',
    'imagedataset': 'ImageDataset',
}


class SequenceDataset:
    """A Sequence dataset with incremental ID and a value with a multiple of 3.

    Args:
        num_samples (int): number of samples. Defaults to 100.
        column_names list[str]: A list of features' and target name. Defaults to ['id', 'sample'].
        offset: Offset to start the sequence from. Defaults to 0.
    """

    def __init__(
        self,
        num_samples: int = 100,
        column_names: list[str] = ['id', 'sample'],
        offset: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.column_encodings = ['str', 'int']
        self.column_sizes = [None, 8]
        self.column_names = column_names
        self.offset = offset
        self._index = 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < self.num_samples:
            return {
                self.column_names[0]: f'{index:06}',
                self.column_names[1]: (3 * index) + self.offset,
            }
        raise IndexError('Index out of bound')

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        if self._index >= self.num_samples:
            raise StopIteration
        id = f'{self._index:06}'
        data = (3 * self._index) + self.offset
        self._index += 1
        return {
            self.column_names[0]: id,
            self.column_names[1]: data,
        }

    def get_sample_in_bytes(self, index: int) -> dict[str, Any]:
        sample = self.__getitem__(index)
        sample[self.column_names[0]] = sample[self.column_names[0]].encode('utf-8')
        sample[self.column_names[1]] = np.int64(sample[self.column_names[1]]).tobytes()
        return sample


class NumberAndSayDataset:
    """Generate a synthetic number-saying dataset.

    Converting a numbers from digits to words, for example, number 123 would spell as
    `one hundred twenty three`. The numbers are generated randomly and it supports a number
    up-to positive/negative approximately 99 Millions.

    Args:
        num_samples (int): number of samples. Defaults to 100.
        column_names list[str]: A list of features' and target name. Defaults to ['number',
            'words'].
        seed (int): seed value for deterministic randomness.
    """

    ones = (
        'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen ' +
        'fifteen sixteen seventeen eighteen nineteen').split()

    tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()

    def __init__(self,
                 num_samples: int = 100,
                 column_names: list[str] = ['number', 'words'],
                 seed: int = 987) -> None:
        self.num_samples = num_samples
        self.column_encodings = ['int', 'str']
        self.column_sizes = [8, None]
        self.column_names = column_names
        self._index = 0
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _say(self, i: int) -> list[str]:
        if i < 0:
            return ['negative'] + self._say(-i)
        elif i <= 19:
            return [self.ones[i]]
        elif i < 100:
            return [self.tens[i // 10 - 2]] + ([self.ones[i % 10]] if i % 10 else [])
        elif i < 1_000:
            return [self.ones[i // 100], 'hundred'] + (self._say(i % 100) if i % 100 else [])
        elif i < 1_000_000:
            return self._say(i // 1_000) + ['thousand'
                                           ] + (self._say(i % 1_000) if i % 1_000 else [])
        elif i < 1_000_000_000:
            return self._say(
                i // 1_000_000) + ['million'] + (self._say(i % 1_000_000) if i % 1_000_000 else [])
        else:
            assert False

    def _get_number(self) -> int:
        sign = (np.random.random() < 0.8) * 2 - 1
        mag = 10**np.random.uniform(1, 4) - 10
        return sign * int(mag**2)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        if self._index >= self.num_samples:
            raise StopIteration
        number = self._get_number()
        words = ' '.join(self._say(number))
        self._index += 1
        return {
            self.column_names[0]: number,
            self.column_names[1]: words,
        }

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value  # pyright: ignore
        np.random.seed(self._seed)


class ImageDataset:
    """An Image dataset with values drawn from a normal distribution.

    Args:
        num_samples (int): number of samples. Defaults to 100.
        column_names list[str]: A list of features' and target name. Defaults to ['x'].
        seed (int): seed value for deterministic randomness.
        shape (Sequence[int]): shape of the image. Defaults to (3, 32, 32).
    """

    def __init__(
            self,
            num_samples: int = 100,
            column_names: list[str] = ['x'],
            seed: int = 987,
            shape: Sequence[int] = (3, 32, 32),
    ) -> None:
        self.shape = shape
        self.num_samples = num_samples
        self.column_encodings = ['pkl']
        self.column_sizes = [None]
        self.column_names = column_names
        self.seed = seed
        self._index = 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < self.num_samples:
            return {
                self.column_names[0]: torch.randn(self.num_samples, *self.shape),
            }
        raise IndexError(f'Index {index} out of bound for size {self.num_samples}')

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        if self._index >= self.num_samples:
            raise StopIteration
        x = torch.randn(self.num_samples, *self.shape)
        self._index += 1
        return {
            self.column_names[0]: x,
        }

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value  # pyright: ignore
        torch.manual_seed(self._seed)


def get_dataset_params(kwargs: dict[str, str]) -> dict[str, Any]:
    """Get the dataset parameters from command-line arguments.

    Args:
        kwargs (dict[str, str]): Command-line arguments.

    Returns:
        dict[str, Any]: Dataset parameters.
    """
    dataset_params = {}
    if 'num_samples' in kwargs:
        dataset_params['num_samples'] = int(kwargs['num_samples'])
    if 'seed' in kwargs:
        dataset_params['seed'] = int(kwargs['seed'])
    if 'shape' in kwargs:
        dataset_params['shape'] = ast.literal_eval(kwargs['shape'])
    if 'column_names' in kwargs:
        dataset_params['column_names'] = ast.literal_eval(kwargs['column_names'])
    if 'offset' in kwargs:
        dataset_params['offset'] = int(kwargs['offset'])
    return dataset_params


def parse_args() -> tuple[Namespace, dict[str, str]]:
    """Parse command-line arguments.

    Returns:
        tuple[Namespace, dict[str, str]]: Command-line arguments and named arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--name',
        type=str,
        default='SequenceDataset',
        help='Dataset name. Supported: SequenceDataset, NumberAndSayDataset, ImageDataset',
    )
    args.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output dataset directory to store MDS shard files (local or remote)',
    )
    # Create a mutually exclusive group to ensure only one can be specified at a time.
    me_group = args.add_mutually_exclusive_group()
    me_group.add_argument('--create', default=False, action='store_true', help='Create dataset')
    me_group.add_argument('--delete', default=False, action='store_true', help='Delete dataset')

    args, runtime_args = args.parse_known_args()
    kwargs = {get_kwargs(k): v for k, v in zip(runtime_args[::2], runtime_args[1::2])}
    return args, kwargs


def main(args: Namespace, kwargs: dict[str, str]) -> None:
    """Create and delete a dataset.

    Args:
        args (Namespace): Arguments.
        kwargs (dict[str, str]): Named arguments.
    """
    if args.create:
        dataset_params = get_dataset_params(kwargs)
        writer_params = get_writer_params(kwargs)
        if args.name.lower() not in _DATASET_MAP:
            raise ValueError(f'Unsupported dataset {args.name}. Supported: {_DATASET_MAP.keys()}')
        dataset = getattr(sys.modules[__name__], _DATASET_MAP[args.name.lower()])(**dataset_params)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))

        with MDSWriter(out=args.out, columns=columns, **writer_params) as out:
            for sample in dataset:
                out.write(sample)

    if args.delete:
        shutil.rmtree(args.out, ignore_errors=True)
        obj = urllib.parse.urlparse(args.out)
        cloud = obj.scheme
        if cloud == '':
            shutil.rmtree(args.out, ignore_errors=True)
        elif cloud == 'gs':
            delete_gcs(args.out)
        elif cloud == 's3':
            delete_s3(args.out)
        elif cloud == 'oci':
            delete_oci(args.out)


if __name__ == '__main__':
    args, kwargs = parse_args()
    main(args, kwargs)
