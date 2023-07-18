# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create a streaming dataset from toy data with various options for regression testing."""

import os
import shutil
import tempfile
import urllib.parse
from argparse import ArgumentParser, Namespace

import utils
from torch.utils.data import DataLoader

from streaming import StreamingDataset
from streaming.base.distributed import barrier

_TRAIN_EPOCHS = 2


def get_kwargs(key: str) -> str:
    """Parse key of named command-line arguments.

    Returns:
        str: Key of named arguments.
    """
    if key.startswith('--'):
        key = key[2:]
    key = key.replace('-', '_')
    return key


def parse_args() -> tuple[Namespace, dict[str, str]]:
    """Parse command-line arguments.

    Returns:
        tuple(Namespace, dict[str, str]): Command-line arguments and named arguments.
    """
    args = ArgumentParser()
    args.add_argument('--cloud', type=str)
    args.add_argument('--check_download', default=False, action='store_true')
    args.add_argument('--local', default=False, action='store_true')
    args.add_argument(
        '--keep_zip',
        default=False,
        action='store_true',
        help=('Whether to keep or delete the compressed form when decompressing'
              ' downloaded shards for StreamingDataset.'),
    )
    args.add_argument(
        '--shuffle',
        default=False,
        action='store_true',
        help=('Whether to iterate over the samples in randomized order for'
              ' StreamingDataset.'),
    )

    args, runtime_args = args.parse_known_args()
    kwargs = {get_kwargs(k): v for k, v in zip(runtime_args[::2], runtime_args[1::2])}
    return args, kwargs


def get_file_count(cloud: str) -> int:
    """Get the number of files in a remote directory.

    Args:
        cloud (str): Cloud provider.
    """
    remote_dir = utils.get_remote_dir(cloud)
    obj = urllib.parse.urlparse(remote_dir)
    files = []
    if cloud == 'gcs':
        from google.cloud.storage import Bucket, Client

        service_account_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        gcs_client = Client.from_service_account_json(service_account_path)

        bucket = Bucket(gcs_client, obj.netloc)
        files = bucket.list_blobs(prefix=obj.path.lstrip('/'))
    elif cloud == 's3':
        import boto3

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(obj.netloc)
        files = bucket.objects.filter(Prefix=obj.path.lstrip('/'))
    elif cloud == 'oci':
        import oci

        config = oci.config.from_file()
        oci_client = oci.object_storage.ObjectStorageClient(
            config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
        namespace = oci_client.get_namespace().data
        objects = oci_client.list_objects(namespace, obj.netloc, prefix=obj.path.lstrip('/'))

        files = objects.data.objects

    return sum(1 for _ in files)


def main(args: Namespace, kwargs: dict[str, str]) -> None:
    """Benchmark time taken to generate the epoch for a given dataset.

    Args:
        args (Namespace): Command-line arguments.
        kwargs (dict): Named arguments.
    """
    tmp_dir = tempfile.gettempdir()
    tmp_download_dir = os.path.join(tmp_dir, 'test_iterate_data_download')
    dataset = StreamingDataset(
        remote=utils.get_remote_dir(args.cloud),
        local=tmp_download_dir if args.local else None,
        split=kwargs.get('split'),
        download_retry=int(kwargs.get('download_retry', 2)),
        download_timeout=float(kwargs.get('download_timeout', 60)),
        validate_hash=kwargs.get('validate_hash'),
        keep_zip=args.keep_zip,
        epoch_size=int(kwargs['epoch_size']) if 'epoch_size' in kwargs else None,
        predownload=int(kwargs['predownload']) if 'predownload' in kwargs else None,
        cache_limit=kwargs.get('cache_limit'),
        partition_algo=kwargs.get('partition_algo', 'orig'),
        num_canonical_nodes=int(kwargs['num_canonical_nodes'])
        if 'num_canonical_nodes' in kwargs else None,
        batch_size=int(kwargs['batch_size']) if 'batch_size' in kwargs else None,
        shuffle=args.shuffle,
        shuffle_algo=kwargs.get('shuffle_algo', 'py1s'),
        shuffle_seed=int(kwargs.get('shuffle_seed', 9176)),
        shuffle_block_size=int(kwargs.get('shuffle_block_size', 1 << 18)),
    )

    if args.check_download and args.cloud is not None:
        num_cloud_files = get_file_count(args.cloud)
        local_dir = dataset.streams[0].local
        num_local_files = len([
            name for name in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, name))
        ])
        print(num_local_files)
        assert num_cloud_files == num_local_files

    dataloader = DataLoader(dataset)
    for _ in range(_TRAIN_EPOCHS):
        for _ in dataloader:
            pass

    barrier()
    # Clean up directories
    for stream in dataset.streams:
        shutil.rmtree(stream.local, ignore_errors=True)
    shutil.rmtree(tmp_download_dir, ignore_errors=True)


if __name__ == '__main__':
    args, kwargs = parse_args()
    main(args, kwargs)
