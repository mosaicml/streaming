# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import numpy as np

from streaming.base import MDSWriter


class SequenceDataset:
    """A Sequence dataset with incremental ID and a value with a multiple of 3.

    Args:
        size (int): number of samples. Defaults to 100.
        column_names List[str]: A list of features' and target name. Defaults to ['id', 'sample'].
    """

    def __init__(self, size: int = 100, column_names: List[str] = ['id', 'sample']) -> None:
        self.size = size
        self.column_encodings = ['str', 'int']
        self.column_sizes = [None, 8]
        self.column_names = column_names
        self._index = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < self.size:
            return {
                self.column_names[0]: f'{index:06}',
                self.column_names[1]: 3 * index,
            }
        raise IndexError('Index out of bound')

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._index >= self.size:
            raise StopIteration
        id = f'{self._index:06}'
        data = 3 * self._index
        self._index += 1
        return {
            self.column_names[0]: id,
            self.column_names[1]: data,
        }

    def get_sample_in_bytes(self, index: int) -> Dict[str, Any]:
        sample = self.__getitem__(index)
        sample[self.column_names[0]] = sample[self.column_names[0]].encode('utf-8')
        sample[self.column_names[1]] = np.int64(sample[self.column_names[1]]).tobytes()
        return sample


class NumberAndSayDataset:
    """Generate a synthetic number-saying dataset, i.e. converting a numbers from digits to words,
    for example, number 123 would spell as one hundred twenty three. The numbers are generated
    randomly and it supports a number up-to positive/negative approximately 99 Millions.

    Args:
        size (int): number of samples. Defaults to 100
        column_names List[str]: A list of features' and target name. Defaults to ['number',
            'words'].
        seed (int): seed value for deterministic randomness
    """

    ones = (
        'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen ' +
        'fifteen sixteen seventeen eighteen nineteen').split()

    tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()

    def __init__(self,
                 size: int = 100,
                 column_names: List[str] = ['number', 'words'],
                 seed: int = 987) -> None:
        self.size = size
        self.column_encodings = ['int', 'str']
        self.column_sizes = [8, None]
        self.column_names = column_names
        self._index = 0
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def _say(self, i: int) -> List[str]:
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

    def __next__(self) -> Dict[str, Any]:
        if self._index >= self.size:
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


def write_mds_dataset(
    out_root: str,
    columns: Dict[str, str],
    samples: Any,
    size_limit: int,
    compression: Optional[str] = None,
    hashes: Optional[List[str]] = None,
) -> None:
    with MDSWriter(out=out_root,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit) as out:
        for sample in samples:
            out.write(sample)
