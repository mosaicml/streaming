# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create a toy dataset using MDSWriter for regression testing."""

import os
import shutil
import tempfile
from argparse import ArgumentParser, Namespace

import numpy as np

from streaming import MDSWriter

_NUM_SAMPLES = 10000
# Word representation of a number
_ONES = ('zero one two three four five six seven eight nine ten eleven twelve '
         'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
_TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()

_COLUMNS = {
    'number': 'int',
    'words': 'str',
}


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:x
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--create', default=False, action='store_true')
    args.add_argument('--delete', default=False, action='store_true')
    args.add_argument(
        '--compression',
        type=str,
        help='Compression or compression:level for MDSWriter.',
    )
    args.add_argument(
        '--hashes',
        type=str,
        nargs='+',
        help='List of hash algorithms to apply to shard files for MDSWriter.',
    )
    args.add_argument(
        '--size_limit',
        type=int,
        default=1 << 26,
        help=('Shard size limit, after which point to start a new shard for '
              'MDSWriter. If ``None``, puts everything in one shard.'),
    )
    return args.parse_args()


def say(i: int) -> list[str]:
    """Get the word form of a number.

    Args:
        i (int): The number.

    Returns:
        List[str]: The number in word form.
    """
    if i < 0:
        return ['negative'] + say(-i)
    elif i <= 19:
        return [_ONES[i]]
    elif i < 100:
        return [_TENS[i // 10 - 2]] + ([_ONES[i % 10]] if i % 10 else [])
    elif i < 1_000:
        return [_ONES[i // 100], 'hundred'] + (say(i % 100) if i % 100 else [])
    elif i < 1_000_000:
        return (say(i // 1_000) + ['thousand'] + (say(i % 1_000) if i % 1_000 else []))
    elif i < 1_000_000_000:
        return (say(i // 1_000_000) + ['million'] + (say(i % 1_000_000) if i % 1_000_000 else []))
    else:
        assert False


def get_dataset(num_samples: int) -> list[dict[str, int | str]]:
    """Generate a number-saying dataset of the given size.

    Args:
        num_samples (int): Number of samples.

    Returns:
        list[dict[str, int | str]]: The two generated splits.
    """
    numbers = [((np.random.random() < 0.8) * 2 - 1) * i for i in range(num_samples)]
    samples = []
    for num in numbers:
        words = ' '.join(say(num))
        sample = {'number': num, 'words': words}
        samples.append(sample)
    return samples


def main(args: Namespace) -> None:
    """Benchmark time taken to generate the epoch for a given dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    tmp_dir = tempfile.gettempdir()
    tmp_upload_dir = os.path.join(tmp_dir, 'regression_upload')

    if args.create:
        dataset = get_dataset(_NUM_SAMPLES)
        with MDSWriter(
                out=tmp_upload_dir,
                columns=_COLUMNS,
                compression=args.compression,
                hashes=args.hashes,
                size_limit=args.size_limit,
        ) as out:
            for sample in dataset:
                out.write(sample)
    if args.delete:
        shutil.rmtree(tmp_upload_dir, ignore_errors=True)


if __name__ == '__main__':
    main(parse_args())
