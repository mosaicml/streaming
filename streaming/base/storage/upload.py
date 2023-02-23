# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Write files to remote location which can be either Cloud Storage or a local path."""

import logging
import os
import shutil
import urllib.parse
from tempfile import mkdtemp
from typing import Any, Optional

import tqdm

__all__ = ['CloudWriter', 'S3Writer', 'GCSWriter', 'OCIWriter', 'LocalWriter']

logger = logging.getLogger(__name__)


class CloudWriter:
    """Upload local files to a cloud storage."""

    def __new__(cls,
                local: Optional[str] = None,
                remote: Optional[str] = None,
                keep_local: bool = False,
                progress_bar: bool = False) -> Any:
        """Instantiate a cloud provider or a local writer based on remote location keyword.

        Args:
            local (str, optional): Optional local output dataset directory. If not
                provided, a random temp directory will be used. If ``remote`` is provided,
                this is where shards are cached before uploading. One or both of ``local``
                and ``remote`` must be provided. Defaults to ``None``.
            remote (str, optional): Optional remote output dataset directory. If not
                provided, no uploading will be done. Defaults to ``None``.
            keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
                shard file or remove it after uploading. Defaults to ``False``.
            progress_bar (bool): Whether to display a progress bar while uploading to a cloud
                provider. Defaults to ``False``.

        Raises:
            ValueError: Either local and/or remote path(s) must be provided
            KeyError: Invalid Cloud provider prefix

        Returns:
            _type_: _description_
        """
        if not local and not remote:
            raise ValueError('You must provide local and/or remote path(s).')

        mapping = {
            's3': S3Writer,
            'gs': GCSWriter,
            'oci': OCIWriter,
            '': LocalWriter,  # For local copy when remote is a local directory
            b'': LocalWriter  # No file copy to remote since remote is None
        }
        obj = urllib.parse.urlparse(remote)
        if obj.scheme in mapping:
            instance = super().__new__(mapping[obj.scheme])
            return instance
        else:
            raise KeyError('Invalid Cloud provider prefix in `remote` argument.')

    def __init__(self,
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        """Initialize and validate local and remote path.

        Args:
            local (str, optional): Optional local output dataset directory. If not
                provided, a random temp directory will be used. If ``remote`` is provided,
                this is where shards are cached before uploading. One or both of ``local``
                and ``remote`` must be provided. Defaults to ``None``.
            remote (str, optional): Optional remote output dataset directory. If not
                provided, no uploading will be done. Defaults to ``None``.
            keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
                shard file or remove it after uploading. Defaults to ``False``.
            progress_bar (bool): Whether to display a progress bar while uploading to a cloud
                provider. Defaults to ``False``.

        Raises:
            ValueError: Either local and/or remote path(s) must be provided
            FileExistsError: Local directory must be empty
        """
        self.keep_local = keep_local
        self.progress_bar = progress_bar
        if local is not None:
            self.local = local
            self.remote = remote
        elif remote is not None:
            self.local = mkdtemp()
            self.remote = remote
        else:
            raise ValueError('You must provide local and/or remote path(s).')

        if os.path.exists(self.local) and len(os.listdir(self.local)) != 0:
            raise FileExistsError(f'Directory is not empty: {self.local}')
        os.makedirs(self.local, exist_ok=True)

    def upload_file(self, filename: str):
        """Upload file from local instance to remote instance.

        Args:
            filename (str): File to upload

        Raises:
            NotImplementedError: Override this method in your sub-class
        """
        raise NotImplementedError('Override this method in your sub-class')

    def clear_local(self, local: str):
        """Remove the local file if it is enabled.

        Args:
            local (str): A local file path
        """
        if not self.keep_local and os.path.isfile(local):
            os.remove(local)


class S3Writer(CloudWriter):
    """Upload file from local machine to AWS S3 bucket.

    Args:
        local (str, optional): Optional local output dataset directory. If not
            provided, a random temp directory will be used. If ``remote`` is provided,
            this is where shards are cached before uploading. One or both of ``local``
            and ``remote`` must be provided. Defaults to ``None``.
        remote (str, optional): Optional remote output dataset directory. If not
            provided, no uploading will be done. Defaults to ``None``.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Whether to display a progress bar while uploading to a cloud
            provider. Defaults to ``False``.
    """

    def __init__(self,
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(local, remote, keep_local, progress_bar)

        import boto3
        from botocore.config import Config
        config = Config()
        self.s3 = boto3.client('s3', config=config)

    def upload_file(self, filename: str):
        """Upload file from local instance to AWS S3 bucket.

        Args:
            filename (str): File to upload
        """
        local_filename = os.path.join(self.local, filename)
        remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
        obj = urllib.parse.urlparse(remote_filename)
        logger.debug(f'Uploading to {remote_filename}')
        file_size = os.stat(local_filename).st_size
        with tqdm.tqdm(total=file_size,
                       unit='B',
                       unit_scale=True,
                       desc=f'Uploading {filename}',
                       disable=(not self.progress_bar)) as pbar:
            self.s3.upload_file(
                local_filename,
                obj.netloc,
                obj.path.lstrip('/'),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
        self.clear_local(local=local_filename)


class GCSWriter(CloudWriter):
    """Upload file from local machine to Google Cloud Storage bucket.

    Args:
        local (str, optional): Optional local output dataset directory. If not
            provided, a random temp directory will be used. If ``remote`` is provided,
            this is where shards are cached before uploading. One or both of ``local``
            and ``remote`` must be provided. Defaults to ``None``.
        remote (str, optional): Optional remote output dataset directory. If not
            provided, no uploading will be done. Defaults to ``None``.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Whether to display a progress bar while uploading to a cloud
            provider. Defaults to ``False``.
    """

    def __init__(self,
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(local, remote, keep_local, progress_bar)

        import boto3

        self.gcs_client = boto3.client('s3',
                                       region_name='auto',
                                       endpoint_url='https://storage.googleapis.com',
                                       aws_access_key_id=os.environ['GCS_KEY'],
                                       aws_secret_access_key=os.environ['GCS_SECRET'])

    def upload_file(self, filename: str):
        """Upload file from local instance to Google Cloud Storage bucket.

        Args:
            filename (str): File to upload
        """
        local_filename = os.path.join(self.local, filename)
        remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
        obj = urllib.parse.urlparse(remote_filename)
        logger.debug(f'Uploading to {remote_filename}')
        file_size = os.stat(local_filename).st_size
        with tqdm.tqdm(total=file_size,
                       unit='B',
                       unit_scale=True,
                       desc=f'Uploading {filename}',
                       disable=(not self.progress_bar)) as pbar:
            self.gcs_client.upload_file(
                local_filename,
                obj.netloc,
                obj.path.lstrip('/'),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
        self.clear_local(local=local_filename)


class OCIWriter(CloudWriter):
    """Upload file from local machine to Oracle Cloud Infrastructure (OCI) Cloud Storage.

    Args:
        local (str, optional): Optional local output dataset directory. If not
            provided, a random temp directory will be used. If ``remote`` is provided,
            this is where shards are cached before uploading. One or both of ``local``
            and ``remote`` must be provided. Defaults to ``None``.
        remote (str, optional): Optional remote output dataset directory. If not
            provided, no uploading will be done. Defaults to ``None``.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Whether to display a progress bar while uploading to a cloud
            provider. Defaults to ``False``.
    """

    def __init__(self,
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(local, remote, keep_local, progress_bar)

        import oci
        config = oci.config.from_file()
        client = oci.object_storage.ObjectStorageClient(
            config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
        self.namespace = client.get_namespace().data
        self.upload_manager = oci.object_storage.UploadManager(client)

    def upload_file(self, filename: str):
        """Upload file from local instance to OCI Cloud Storage bucket.

        Args:
            filename (str): File to upload
        """
        local_filename = os.path.join(self.local, filename)
        remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
        obj = urllib.parse.urlparse(remote_filename)
        bucket_name = obj.netloc.split('@' + self.namespace)[0]
        # Remove leading and trailing forward slash from string
        object_path = obj.path.strip('/')
        logger.debug(f'Uploading to {remote_filename}')
        file_size = os.stat(local_filename).st_size
        with tqdm.tqdm(total=file_size,
                       unit='B',
                       unit_scale=True,
                       desc=f'Uploading {filename}',
                       disable=(not self.progress_bar)) as pbar:
            self.upload_manager.upload_file(
                namespace_name=self.namespace,
                bucket_name=bucket_name,
                object_name=object_path,
                file_path=local_filename,
                progress_callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
        self.clear_local(local=local_filename)


class LocalWriter(CloudWriter):
    """Copy file from one local directory to another local directory.

    Args:
        local (str, optional): Optional local output dataset directory. If not
            provided, a random temp directory will be used. If ``remote`` is provided,
            this is where shards are cached before uploading. One or both of ``local``
            and ``remote`` must be provided. Defaults to ``None``.
        remote (str, optional): Optional remote output dataset directory. If not
            provided, no uploading will be done. Defaults to ``None``.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Whether to display a progress bar while uploading to a cloud
            provider. Defaults to ``False``.
    """

    def __init__(self,
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(local, remote, keep_local, progress_bar)
        # Create remote directory if it doesn't exist
        if self.remote:
            os.makedirs(self.remote, exist_ok=True)

    def upload_file(self, filename: str):
        """Copy file from one local path to another local path.

        Args:
            filename (str): File to copy
        """
        if self.remote:
            local_filename = os.path.join(self.local, filename)
            remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
            logger.debug(f'Copying to {remote_filename}')
            shutil.copy(local_filename, remote_filename)
            self.clear_local(local=local_filename)
