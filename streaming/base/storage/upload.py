# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Write files to remote location which can be either Cloud Storage or a local path."""

import logging
import os
import shutil
import sys
import urllib.parse
from tempfile import mkdtemp
from typing import Any, List, Union

import tqdm

__all__ = ['CloudWriter', 'S3Writer', 'GCSWriter', 'OCIWriter', 'LocalWriter']

logger = logging.getLogger(__name__)

MAPPING = {
    's3': 'S3Writer',
    'gs': 'GCSWriter',
    'oci': 'OCIWriter',
    '': 'LocalWriter',
}


class CloudWriter:
    """Upload local files to a cloud storage."""

    @classmethod
    def get(cls,
            out: Union[str, List[str]],
            keep_local: bool = False,
            progress_bar: bool = False) -> Any:
        """Instantiate a cloud provider or a local writer based on remote location keyword.

        Args:
            out (str | List[str]): Output dataset directory to save shard files.
                1. If `out` is a local directory, shard files are saved locally
                2. If `out` is a remote directory, a random local temporary directory is created to
                   cached the shard files and then the shard files are uploaded to a remote
                   location. At the end, a temp directory is deleted once shards are uploaded.
                3. If `out` is a list of `(local_dir, remote_dir)`, shard files are saved in the
                   `local_dir` and also uploaded to a remote location.
            keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
                shard file or remove it after uploading. Defaults to ``False``.
            progress_bar (bool): Whether to display a progress bar while uploading to a cloud
                provider. Defaults to ``False``.

        Returns:
            CloudWriter: An instance of sub-class.
        """
        cls._validate(cls, out)
        obj = urllib.parse.urlparse(out) if isinstance(out, str) else urllib.parse.urlparse(out[1])
        return getattr(sys.modules[__name__], MAPPING[obj.scheme])(out, keep_local, progress_bar)

    def _validate(self, out: Union[str, List[str]]) -> None:
        """Validate the `out` argument.

        Args:
            out (str | List[str]): Output dataset directory to save shard files.
                1. If `out` is a local directory, shard files are saved locally
                2. If `out` is a remote directory, a random local temporary directory is created to
                   cached the shard files and then the shard files are uploaded to a remote
                   location. At the end, a temp directory is deleted once shards are uploaded.
                3. If `out` is a list of `(local_dir, remote_dir)`, shard files are saved in the
                   `local_dir` and also uploaded to a remote location.

        Raises:
            ValueError: Invalid number of `out` argument.
            ValueError: Invalid Cloud provider prefix.
        """
        if isinstance(out, str):
            obj = urllib.parse.urlparse(out)
        else:
            if len(out) != 2:
                raise ValueError(''.join([
                    f'Invalid `out` argument. It is either a string of local/remote directory ',
                    'or a list of two strings with [local, remote].'
                ]))
            obj = urllib.parse.urlparse(out[1])
        if obj.scheme not in MAPPING:
            raise ValueError('Invalid Cloud provider prefix.')

    def __init__(self,
                 out: Union[str, List[str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        """Initialize and validate local and remote path.

        Args:
            out (str | List[str]): Output dataset directory to save shard files.
                1. If `out` is a local directory, shard files are saved locally
                2. If `out` is a remote directory, a random local temporary directory is created to
                   cached the shard files and then the shard files are uploaded to a remote
                   location. At the end, a temp directory is deleted once shards are uploaded.
                3. If `out` is a list of `(local_dir, remote_dir)`, shard files are saved in the
                   `local_dir` and also uploaded to a remote location.
            keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
                shard file or remove it after uploading. Defaults to ``False``.
            progress_bar (bool): Whether to display a progress bar while uploading to a cloud
                provider. Defaults to ``False``.

        Raises:
            FileExistsError: Local directory must be empty.
        """
        self._validate(out)
        self.keep_local = keep_local
        self.progress_bar = progress_bar

        if isinstance(out, str):
            # It is a remote directory
            if urllib.parse.urlparse(out).scheme:
                self.local = mkdtemp()
                self.remote = out
            # It is a local directory
            else:
                self.local = out
                self.remote = None
        else:
            self.local = out[0]
            self.remote = out[1]

        if os.path.exists(self.local) and len(os.listdir(self.local)) != 0:
            raise FileExistsError(f'Directory is not empty: {self.local}')
        os.makedirs(self.local, exist_ok=True)

    def upload_file(self, filename: str):
        """Upload file from local instance to remote instance.

        Args:
            filename (str): File to upload.

        Raises:
            NotImplementedError: Override this method in your sub-class.
        """
        raise NotImplementedError('Override this method in your sub-class')

    def clear_local(self, local: str):
        """Remove the local file if it is enabled.

        Args:
            local (str): A local file path.
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
                 out: Union[str, List[str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)

        import boto3
        from botocore.config import Config
        config = Config()
        self.s3 = boto3.client('s3', config=config)

    def upload_file(self, filename: str):
        """Upload file from local instance to AWS S3 bucket.

        Args:
            filename (str): File to upload.
        """
        local_filename = os.path.join(self.local, filename)
        remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
        obj = urllib.parse.urlparse(remote_filename)
        logger.debug(f'Uploading to {remote_filename}')
        file_size = os.stat(local_filename).st_size
        with tqdm.tqdm(total=file_size,
                       unit='B',
                       unit_scale=True,
                       desc=f'Uploading to {remote_filename}',
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
                 out: Union[str, List[str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)

        import boto3

        self.gcs_client = boto3.client('s3',
                                       region_name='auto',
                                       endpoint_url='https://storage.googleapis.com',
                                       aws_access_key_id=os.environ['GCS_KEY'],
                                       aws_secret_access_key=os.environ['GCS_SECRET'])

    def upload_file(self, filename: str):
        """Upload file from local instance to Google Cloud Storage bucket.

        Args:
            filename (str): File to upload.
        """
        local_filename = os.path.join(self.local, filename)
        remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
        obj = urllib.parse.urlparse(remote_filename)
        logger.debug(f'Uploading to {remote_filename}')
        file_size = os.stat(local_filename).st_size
        with tqdm.tqdm(total=file_size,
                       unit='B',
                       unit_scale=True,
                       desc=f'Uploading to {remote_filename}',
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
                 out: Union[str, List[str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)

        import oci
        config = oci.config.from_file()
        client = oci.object_storage.ObjectStorageClient(
            config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
        self.namespace = client.get_namespace().data
        self.upload_manager = oci.object_storage.UploadManager(client)

    def upload_file(self, filename: str):
        """Upload file from local instance to OCI Cloud Storage bucket.

        Args:
            filename (str): File to upload.
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
                       desc=f'Uploading to {remote_filename}',
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
                 out: Union[str, List[str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)
        # Create remote directory if it doesn't exist
        if self.remote:
            os.makedirs(self.remote, exist_ok=True)

    def upload_file(self, filename: str):
        """Copy file from one local path to another local path.

        Args:
            filename (str): Relative filepath to copy.
        """
        if self.remote:
            local_filename = os.path.join(self.local, filename)
            remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
            logger.debug(f'Copying to {remote_filename}')
            shutil.copy(local_filename, remote_filename)
            self.clear_local(local=local_filename)
