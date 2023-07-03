# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create a toy dataset using MDSWriter for regression testing."""

import os
import shutil
import urllib
from argparse import ArgumentParser, Namespace
from typing import Union

import numpy as np
import utils

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

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--cloud', type=str)
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


def get_dataset(num_samples: int) -> list[dict[str, Union[int, str]]]:
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
    remote_dir = utils.get_remote_dir(args.cloud)
    if args.create:
        dataset = get_dataset(_NUM_SAMPLES)
        with MDSWriter(
                out=remote_dir,
                columns=_COLUMNS,
                compression=args.compression,
                hashes=args.hashes,
                size_limit=args.size_limit,
        ) as out:
            for sample in dataset:
                out.write(sample)
    if args.delete:
        if args.cloud is None:
            shutil.rmtree(remote_dir, ignore_errors=True)
        elif args.cloud == 'gs':
            from google.cloud.storage import Bucket, Client

            service_account_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            gcs_client = Client.from_service_account_json(service_account_path)
            obj = urllib.parse.urlparse(remote_dir)

            bucket = Bucket(gcs_client, obj.netloc)
            blobs = bucket.list_blobs(prefix=obj.path.lstrip('/'))

            for blob in blobs:
                blob.delete()


if __name__ == '__main__':
    main(parse_args())
