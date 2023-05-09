# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Write files to remote location which can be either Cloud Storage or a local path."""

import logging
import os
import shutil
import sys
import urllib.parse
from tempfile import mkdtemp
from typing import Any, Tuple, Union

import tqdm

from streaming.base.storage.download import BOTOCORE_CLIENT_ERROR_CODES

__all__ = [
    'CloudUploader', 'S3Uploader', 'GCSUploader', 'OCIUploader', 'R2Uploader', 'AzureUploader',
    'LocalUploader'
]

logger = logging.getLogger(__name__)

UPLOADERS = {
    's3': 'S3Uploader',
    'gs': 'GCSUploader',
    'oci': 'OCIUploader',
    'r2': 'R2Uploader',
    'azure': 'AzureUploader',
    '': 'LocalUploader',
}


class CloudUploader:
    """Upload local files to a cloud storage."""

    @classmethod
    def get(cls,
            out: Union[str, Tuple[str, str]],
            keep_local: bool = False,
            progress_bar: bool = False) -> Any:
        """Instantiate a cloud provider uploader or a local uploader based on remote path.

        Args:
            out (str | Tuple[str, str]): Output dataset directory to save shard files.

                1. If ``out`` is a local directory, shard files are saved locally.
                2. If ``out`` is a remote directory, a local temporary directory is created to
                   cache the shard files and then the shard files are uploaded to a remote
                   location. At the end, the temp directory is deleted once shards are uploaded.
                3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
                   the `local_dir` and also uploaded to a remote location.
            keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
                shard file or remove it after uploading. Defaults to ``False``.
            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.

        Returns:
            CloudUploader: An instance of sub-class.
        """
        cls._validate(cls, out)
        obj = urllib.parse.urlparse(out) if isinstance(out, str) else urllib.parse.urlparse(out[1])
        return getattr(sys.modules[__name__], UPLOADERS[obj.scheme])(out, keep_local, progress_bar)

    def _validate(self, out: Union[str, Tuple[str, str]]) -> None:
        """Validate the `out` argument.

        Args:
            out (str | Tuple[str, str]): Output dataset directory to save shard files.

                1. If ``out`` is a local directory, shard files are saved locally.
                2. If ``out`` is a remote directory, a local temporary directory is created to
                   cache the shard files and then the shard files are uploaded to a remote
                   location. At the end, the temp directory is deleted once shards are uploaded.
                3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
                   the `local_dir` and also uploaded to a remote location.

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
        if obj.scheme not in UPLOADERS:
            raise ValueError('Invalid Cloud provider prefix.')

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        """Initialize and validate local and remote path.

        Args:
            out (str | Tuple[str, str]): Output dataset directory to save shard files.

                1. If ``out`` is a local directory, shard files are saved locally.
                2. If ``out`` is a remote directory, a local temporary directory is created to
                   cache the shard files and then the shard files are uploaded to a remote
                   location. At the end, the temp directory is deleted once shards are uploaded.
                3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
                   the `local_dir` and also uploaded to a remote location.
            keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
                shard file or remove it after uploading. Defaults to ``False``.
            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.

        Raises:
            FileExistsError: Local directory must be empty.
        """
        self._validate(out)
        self.keep_local = keep_local
        self.progress_bar = progress_bar

        if isinstance(out, str):
            # It is a remote directory
            if urllib.parse.urlparse(out).scheme != '':
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


class S3Uploader(CloudUploader):
    """Upload file from local machine to AWS S3 bucket.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
               the `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
            a remote location. Default to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)

        import boto3
        from botocore.config import Config
        config = Config()
        # Create a session and use it to make our client. Unlike Resources and Sessions,
        # clients are generally thread-safe.
        session = boto3.session.Session()
        self.s3 = session.client('s3', config=config)
        self.check_bucket_exists(self.remote)  # pyright: ignore

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

    def check_bucket_exists(self, remote: str):
        """Raise an exception if the bucket does not exist.

        Args:
            remote (str): S3 bucket path.

        Raises:
            error: Bucket does not exist.
        """
        from botocore.exceptions import ClientError

        bucket_name = urllib.parse.urlparse(remote).netloc
        try:
            self.s3.head_bucket(Bucket=bucket_name)
        except ClientError as error:
            if error.response['Error']['Code'] == BOTOCORE_CLIENT_ERROR_CODES:
                error.args = (f'Either bucket `{bucket_name}` does not exist! ' +
                              f'or check the bucket permission.',)
            raise error


class GCSUploader(CloudUploader):
    """Upload file from local machine to Google Cloud Storage bucket.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
               the `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
            a remote location. Default to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)

        import boto3

        # Create a session and use it to make our client. Unlike Resources and Sessions,
        # clients are generally thread-safe.
        session = boto3.session.Session()
        self.gcs_client = session.client('s3',
                                         region_name='auto',
                                         endpoint_url='https://storage.googleapis.com',
                                         aws_access_key_id=os.environ['GCS_KEY'],
                                         aws_secret_access_key=os.environ['GCS_SECRET'])
        self.check_bucket_exists(self.remote)  # pyright: ignore

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

    def check_bucket_exists(self, remote: str):
        """Raise an exception if the bucket does not exist.

        Args:
            remote (str): GCS bucket path.

        Raises:
            error: Bucket does not exist.
        """
        from botocore.exceptions import ClientError

        bucket_name = urllib.parse.urlparse(remote).netloc
        try:
            self.gcs_client.head_bucket(Bucket=bucket_name)
        except ClientError as error:
            if error.response['Error']['Code'] == BOTOCORE_CLIENT_ERROR_CODES:
                error.args = (f'Either bucket `{bucket_name}` does not exist! ' +
                              f'or check the bucket permission.',)
            raise error


class OCIUploader(CloudUploader):
    """Upload file from local machine to Oracle Cloud Infrastructure (OCI) Cloud Storage.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
               the `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
            a remote location. Default to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)

        import oci
        config = oci.config.from_file()
        self.client = oci.object_storage.ObjectStorageClient(
            config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
        self.namespace = self.client.get_namespace().data
        self.upload_manager = oci.object_storage.UploadManager(self.client)
        self.check_bucket_exists(self.remote)  # pyright: ignore

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

    def check_bucket_exists(self, remote: str):
        """Raise an exception if the bucket does not exist.

        Args:
            remote (str): OCI bucket path.

        Raises:
            error: Bucket does not exist.
        """
        from oci.exceptions import ServiceError

        obj = urllib.parse.urlparse(remote)
        bucket_name = obj.netloc.split('@' + self.namespace)[0]
        try:
            self.client.head_bucket(bucket_name=bucket_name, namespace_name=self.namespace)
        except ServiceError as error:
            if error.status == 404:
                error.args = (f'Bucket `{bucket_name}` does not exist! ' +
                              f'Check the bucket permission or create the bucket.',)
            raise error


class R2Uploader(CloudUploader):
    """Upload file from local machine to Cloudflare R2 bucket.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
               the `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
            a remote location. Default to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)

        import boto3

        # Create a session and use it to make our client. Unlike Resources and Sessions,
        # clients are generally thread-safe.
        session = boto3.session.Session()
        self.r2_client = session.client('s3',
                                        region_name='auto',
                                        endpoint_url=os.environ['S3_ENDPOINT_URL'])
        self.check_bucket_exists(self.remote)  # pyright: ignore

    def upload_file(self, filename: str):
        """Upload file from local instance to Cloudflare R2 bucket.

        Args:
            filename (str): File to upload.
        """
        local_filename = os.path.join(self.local, filename)
        remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
        # fix paths for windows
        local_filename = local_filename.replace('\\', '/')
        remote_filename = remote_filename.replace('\\', '/')
        obj = urllib.parse.urlparse(remote_filename)
        logger.debug(f'Uploading to {remote_filename}')
        file_size = os.stat(local_filename).st_size
        with tqdm.tqdm(total=file_size,
                       unit='B',
                       unit_scale=True,
                       desc=f'Uploading to {remote_filename}',
                       disable=(not self.progress_bar)) as pbar:
            self.r2_client.upload_file(
                local_filename,
                obj.netloc,
                obj.path.lstrip('/'),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
        self.clear_local(local=local_filename)

    def check_bucket_exists(self, remote: str):
        """Raise an exception if the bucket does not exist.

        Args:
            remote (str): R2 bucket path.

        Raises:
            error: Bucket does not exist.
        """
        from botocore.exceptions import ClientError

        bucket_name = urllib.parse.urlparse(remote).netloc
        try:
            self.r2_client.head_bucket(Bucket=bucket_name)
        except ClientError as error:
            if error.response['Error']['Code'] == BOTOCORE_CLIENT_ERROR_CODES:
                error.args = (f'Either bucket `{bucket_name}` does not exist! ' +
                              f'or check the bucket permission.',)
            raise error


class AzureUploader(CloudUploader):
    """Upload file from local machine to Microsoft Azure bucket.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.
            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
               the `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
            a remote location. Default to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar)

        from azure.storage.blob import BlobServiceClient

        # Create a session and use it to make our client. Unlike Resources and Sessions,
        # clients are generally thread-safe.
        self.azure_service = BlobServiceClient(
            account_url=f"https://{os.environ['AZURE_ACCOUNT_NAME']}.blob.core.windows.net",
            credential=os.environ['AZURE_ACCOUNT_ACCESS_KEY'])
        self.check_bucket_exists(self.remote)  # pyright: ignore

    def upload_file(self, filename: str):
        """Upload file from local instance to Cloudflare R2 bucket.

        Args:
            filename (str): File to upload.
        """
        local_filename = os.path.join(self.local, filename)
        local_filename = local_filename.replace('\\', '/')
        remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
        remote_filename = remote_filename.replace('\\', '/')
        obj = urllib.parse.urlparse(remote_filename)
        logger.debug(f'Uploading to {remote_filename}')
        file_size = os.stat(local_filename).st_size
        container_client = self.azure_service.get_container_client(container=obj.netloc)

        with tqdm.tqdm(total=file_size,
                       unit='B',
                       unit_scale=True,
                       desc=f'Uploading to {remote_filename}',
                       disable=(not self.progress_bar)) as pbar:
            with open(local_filename, 'rb') as data:
                container_client.upload_blob(
                    name=obj.path.lstrip('/'),
                    data=data,
                    progress_hook=lambda bytes_transferred, _: pbar.update(bytes_transferred),
                    overwrite=True)
        self.clear_local(local=local_filename)

    def check_bucket_exists(self, remote: str):
        """Raise an exception if the bucket does not exist.

        Args:
            remote (str): azure bucket path.

        Raises:
            error: Bucket does not exist.
        """
        bucket_name = urllib.parse.urlparse(remote).netloc
        if self.azure_service.get_container_client(container=bucket_name).exists() == False:
            raise FileNotFoundError(
                f'Either bucket `{bucket_name}` does not exist! ' +
                f'or check the bucket permission.',)


class LocalUploader(CloudUploader):
    """Copy file from one local directory to another local directory.

    Args:
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in
               the `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset
            shard file or remove it after uploading. Defaults to ``False``.
        progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
            a remote location. Default to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
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
