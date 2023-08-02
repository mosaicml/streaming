# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create a streaming dataset from toy data with various options for regression testing."""

import os
import time
import urllib.parse
from argparse import ArgumentParser, Namespace
from typing import Union

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from utils import get_dataloader_params, get_kwargs, get_streaming_dataset_params

from streaming import StreamingDataset
from streaming.base.distributed import (all_gather, barrier, get_rank, get_world_size,
                                        maybe_init_dist)


def parse_args() -> tuple[Namespace, dict[str, str]]:
    """Parse command-line arguments.

    Returns:
        tuple(Namespace, dict[str, str]): Command-line arguments and named arguments.
    """
    args = ArgumentParser()
    args.add_argument('--epochs',
                      type=int,
                      default=2,
                      help='Number of epochs to iterate over the data')
    args.add_argument('--validate-files', default=False, action='store_true', help='Verify files')
    args.add_argument('--shuffling-validation-file',
                      type=str,
                      help='File location to store sample order')
    args.add_argument('--validate-iter-time',
                      default=False,
                      action='store_true',
                      help='Test iter time')

    args, runtime_args = args.parse_known_args()
    kwargs = {get_kwargs(k): v for k, v in zip(runtime_args[::2], runtime_args[1::2])}
    return args, kwargs


def get_file_count(remote: str) -> Union[int, None]:
    """Get the number of files in a remote directory.

    Args:
        remote (str): Remote directory URL.
    """
    obj = urllib.parse.urlparse(remote)
    cloud = obj.scheme
    files = []
    if cloud == '':
        return len(
            [name for name in os.listdir(remote) if os.path.isfile(os.path.join(remote, name))])
    if cloud == 'gs':
        from google.cloud.storage import Bucket, Client

        service_account_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        gcs_client = Client.from_service_account_json(service_account_path)

        bucket = Bucket(gcs_client, obj.netloc)
        files = bucket.list_blobs(prefix=obj.path.lstrip('/'))
        return sum(1 for f in files if f.key[-1] != '/')
    elif cloud == 's3':
        import boto3

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(obj.netloc)
        files = bucket.objects.filter(Prefix=obj.path.lstrip('/'))
        return sum(1 for f in files if f.key[-1] != '/')
    elif cloud == 'oci':
        import oci

        config = oci.config.from_file()
        oci_client = oci.object_storage.ObjectStorageClient(
            config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
        namespace = oci_client.get_namespace().data
        objects = oci_client.list_objects(namespace, obj.netloc, prefix=obj.path.lstrip('/'))

        files = objects.data.objects
        return sum(1 for _ in files)
    else:
        raise ValueError(f'Unsupported remote directory prefix {cloud} in {remote}')


def main(args: Namespace, kwargs: dict[str, str]) -> None:
    """Benchmark time taken to generate the epoch for a given dataset.

    Args:
        args (Namespace): Command-line arguments.
        kwargs (dict): Named arguments.
    """
    # Initialize torch dist ourselves, if necessary.
    destroy_dist = maybe_init_dist()

    dataset_params = get_streaming_dataset_params(kwargs)
    dataloader_params = get_dataloader_params(kwargs)
    dataset = StreamingDataset(**dataset_params)
    dataloader = DataLoader(dataset=dataset, **dataloader_params)
    iter_time = []
    start_time = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            start_time = time.time()
        for batch in dataloader:
            if args.shuffling_validation_file is not None:
                samples = [int(sample) for sample in batch['id']]
                obj_gather_list = [
                    torch.zeros(1, dtype=torch.int64).cuda(get_rank())
                    for _ in range(get_world_size())
                ]
                all_gather(obj_gather_list,
                           torch.Tensor(samples).to(dtype=torch.int64).cuda(get_rank()))
                barrier()
                if get_rank() == 0:
                    with open(args.shuffling_validation_file, 'a') as f:
                        all_samples = [
                            str(sample) + '\n'
                            for tensors in obj_gather_list
                            for sample in tensors.tolist()
                        ]
                        f.writelines(all_samples)
        if epoch > 0:
            iter_time.append(time.time() - start_time)

    if args.validate_files and dataset.streams[0].remote is not None:
        num_remote_files = get_file_count(dataset.streams[0].remote)
        local_dir = dataset.streams[0].local
        num_local_files = len([
            name for name in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, name))
        ])
        assert num_remote_files == num_local_files, f'Expected {num_remote_files} files, got {num_local_files}'

    # TODO: Assert the iteration time is within a certain threshold
    if args.validate_iter_time:
        print(f'Average iter time: {sum(iter_time) / len(iter_time)} secs.')

    if destroy_dist:
        dist.destroy_process_group()


if __name__ == '__main__':
    args, kwargs = parse_args()
    main(args, kwargs)
