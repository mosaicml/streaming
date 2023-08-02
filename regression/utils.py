# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for regression testing."""

import os
import tempfile
import urllib.parse
from typing import Any, Dict


def get_kwargs(kwargs: str) -> str:
    """Parse key of named command-line arguments.

    Args:
        kwargs (str): Command-line arguments.

    Returns:
        str: Key of named arguments.
    """
    if kwargs.startswith('--'):
        kwargs = kwargs[2:]
    kwargs = kwargs.replace('-', '_')
    return kwargs


def get_streaming_dataset_params(kwargs: Dict[str, str]) -> Dict[str, Any]:
    """Get the streaming dataset parameters from command-line arguments.

    Args:
        kwargs (Dict[str, str]): Command-line arguments.

    Returns:
        Dict[str, Any]: Dataset parameters.
    """
    dataset_params = {}
    if 'remote' in kwargs:
        dataset_params['remote'] = kwargs['remote']
    if 'local' in kwargs:
        dataset_params['local'] = kwargs['local']
    if 'split' in kwargs:
        dataset_params['split'] = kwargs['split']
    if 'download_retry' in kwargs:
        dataset_params['download_retry'] = int(kwargs['download_retry'])
    if 'download_timeout' in kwargs:
        dataset_params['download_timeout'] = float(kwargs['download_timeout'])
    if 'validate_hash' in kwargs:
        dataset_params['validate_hash'] = kwargs['validate_hash']
    if 'keep_zip' in kwargs:
        dataset_params['keep_zip'] = kwargs['keep_zip'].lower().capitalize() == 'True'
    if 'epoch_size' in kwargs:
        dataset_params['epoch_size'] = kwargs['epoch_size']
    if 'predownload' in kwargs:
        dataset_params['predownload'] = int(kwargs['predownload'])
    if 'cache_limit' in kwargs:
        dataset_params['cache_limit'] = kwargs['cache_limit']
    if 'partition_algo' in kwargs:
        dataset_params['partition_algo'] = kwargs['partition_algo']
    if 'num_canonical_nodes' in kwargs:
        dataset_params['num_canonical_nodes'] = int(kwargs['num_canonical_nodes'])
    if 'batch_size' in kwargs:
        dataset_params['batch_size'] = int(kwargs['batch_size'])
    if 'shuffle' in kwargs:
        dataset_params['shuffle'] = kwargs['shuffle'].lower().capitalize() == 'True'
    if 'shuffle_algo' in kwargs:
        dataset_params['shuffle_algo'] = kwargs['shuffle_algo']
    if 'shuffle_seed' in kwargs:
        dataset_params['shuffle_seed'] = int(kwargs['shuffle_seed'])
    if 'shuffle_block_size' in kwargs:
        dataset_params['shuffle_block_size'] = int(kwargs['shuffle_block_size'])
    if 'sampling_method' in kwargs:
        dataset_params['sampling_method'] = kwargs['sampling_method']
    if 'proportion' in kwargs:
        dataset_params['proportion'] = float(kwargs['proportion'])
    if 'repeat' in kwargs:
        dataset_params['repeat'] = float(kwargs['repeat'])
    if 'choose' in kwargs:
        dataset_params['choose'] = int(kwargs['choose'])
    return dataset_params


def get_dataloader_params(kwargs: Dict[str, str]) -> Dict[str, Any]:
    """Get the dataloader parameters from command-line arguments.

    Args:
        kwargs (Dict[str, str]): Command-line arguments.

    Returns:
        Dict[str, Any]: Dataloader parameters.
    """
    dataloader_params = {}
    if 'num_workers' in kwargs:
        dataloader_params['num_workers'] = int(kwargs['num_workers'])
    if 'batch_size' in kwargs:
        dataloader_params['batch_size'] = int(kwargs['batch_size'])
    if 'pin_memory' in kwargs:
        dataloader_params['pin_memory'] = kwargs['pin_memory'].lower().capitalize() == 'True'
    if 'persistent_workers' in kwargs:
        dataloader_params['persistent_workers'] = kwargs['persistent_workers'].lower().capitalize(
        ) == 'True'
    return dataloader_params


def get_writer_params(kwargs: Dict[str, str]) -> Dict[str, Any]:
    """Get the writer parameters from command-line arguments.

    Args:
        kwargs (Dict[str, str]): Command-line arguments.

    Returns:
        Dict[str, Any]: Writer parameters.
    """
    writer_params = {}
    if 'keep_local' in kwargs:
        writer_params['keep_local'] = kwargs['keep_local'].lower().capitalize() == 'True'
    if 'compression' in kwargs:
        writer_params['compression'] = kwargs['compression']
    if 'hashes' in kwargs:
        writer_params['hashes'] = kwargs['hashes'].split(',')
    if 'size_limit' in kwargs:
        writer_params['size_limit'] = str(kwargs['size_limit'])
    if 'progress_bar' in kwargs:
        writer_params['progress_bar'] = kwargs['progress_bar'].lower().capitalize() == 'True'
    if 'max_workers' in kwargs:
        writer_params['max_workers'] = int(kwargs['max_workers'])
    print(writer_params)
    return writer_params


def get_local_remote_dir() -> str:
    """Get a local remote directory."""
    tmp_dir = tempfile.gettempdir()
    tmp_remote_dir = os.path.join(tmp_dir, 'regression_remote')
    return tmp_remote_dir


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
