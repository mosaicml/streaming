# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create a toy dataset using MDSWriter for regression testing."""

import os
import random
import shutil
import string
import urllib.parse
from argparse import ArgumentParser, Namespace
from typing import Union

import numpy as np
import utils

from streaming import MDSWriter

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
    args.add_argument('--cloud_url', type=str)
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
    args.add_argument('--num_samples',
                      type=int,
                      default=10000,
                      help='Number of samples to generate')
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
    for num in range(num_samples):
        sample = {
            'number': num,
            'words': ''.join([random.choice(string.ascii_lowercase) for _ in range(num_samples)])
        }
        samples.append(sample)
    return samples


def delete_gcs(remote_dir: str) -> None:
    """Delete a remote directory from gcs.

    Args:
        remote_dir (str): Location of the remote directory.
    """
    from google.cloud.storage import Bucket, Client

    service_account_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    gcs_client = Client.from_service_account_json(service_account_path)
    obj = urllib.parse.urlparse(remote_dir)

    bucket = Bucket(gcs_client, obj.netloc)
    blobs = bucket.list_blobs(prefix=obj.path.lstrip('/'))

    for blob in blobs:
        blob.delete()


def delete_s3(remote_dir: str) -> None:
    """Delete a remote directory from s3.

    Args:
        remote_dir (str): Location of the remote directory.
    """
    import boto3

    obj = urllib.parse.urlparse(remote_dir)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(obj.netloc)
    bucket.objects.filter(Prefix=obj.path.lstrip('/')).delete()


def delete_oci(remote_dir: str) -> None:
    """Delete a remote directory from oci.

    Args:
        remote_dir (str): Location of the remote directory.
    """
    import oci

    obj = urllib.parse.urlparse(remote_dir)

    config = oci.config.from_file()
    oci_client = oci.object_storage.ObjectStorageClient(
        config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
    namespace = oci_client.get_namespace().data
    objects = oci_client.list_objects(namespace, obj.netloc, prefix=obj.path.lstrip('/'))

    for filenames in objects.data.objects:
        oci_client.delete_object(namespace, obj.netloc, filenames.name)


def main(args: Namespace) -> None:
    """Benchmark time taken to generate the epoch for a given dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    remote_dir = args.cloud_url if args.cloud_url is not None else utils.get_local_remote_dir()
    if args.create:
        dataset = get_dataset(args.num_samples)
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
        obj = urllib.parse.urlparse(remote_dir)
        cloud = obj.scheme
        if cloud == '':
            shutil.rmtree(remote_dir, ignore_errors=True)
        elif cloud == 'gs':
            delete_gcs(remote_dir)
        elif cloud == 's3':
            delete_s3(remote_dir)
        elif cloud == 'oci':
            delete_oci(remote_dir)


if __name__ == '__main__':
    main(parse_args())
