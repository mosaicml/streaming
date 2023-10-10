# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Write files to remote location which can be either Cloud Storage or a local path."""

import logging
import os
import pathlib
import shutil
import sys
import urllib.parse
from enum import Enum
from tempfile import mkdtemp
from typing import Any, List, Optional, Tuple, Union

import tqdm

from streaming.base.storage.download import (BOTOCORE_CLIENT_ERROR_CODES,
                                             GCS_ERROR_NO_AUTHENTICATION)
from streaming.base.util import get_import_exception_message, retry

__all__ = [
    'CloudUploader',
    'S3Uploader',
    'GCSUploader',
    'OCIUploader',
    'AzureUploader',
    'DatabricksUnityCatalogUploader',
    'DBFSUploader',
    'LocalUploader',
]

logger = logging.getLogger(__name__)

UPLOADERS = {
    's3': 'S3Uploader',
    'gs': 'GCSUploader',
    'oci': 'OCIUploader',
    'azure': 'AzureUploader',
    'azure-dl': 'AzureDataLakeUploader',
    'dbfs:/Volumes': 'DatabricksUnityCatalogUploader',
    'dbfs': 'DBFSUploader',
    '': 'LocalUploader',
}


class GCSAuthentication(Enum):
    HMAC = 1
    SERVICE_ACCOUNT = 2


class CloudUploader:
    """Upload local files to a cloud storage."""

    @classmethod
    def get(cls,
            out: Union[str, Tuple[str, str]],
            keep_local: bool = False,
            progress_bar: bool = False,
            retry: int = 2,
            exist_ok: bool = False) -> Any:
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
            retry (int): Number of times to retry uploading a file. Defaults to ``2``.
            exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
                exists and has contents. Defaults to ``False``.

        Returns:
            CloudUploader: An instance of sub-class.
        """
        cls._validate(cls, out)
        obj = urllib.parse.urlparse(out) if isinstance(out, str) else urllib.parse.urlparse(out[1])
        provider_prefix = obj.scheme
        if obj.scheme == 'dbfs':
            path = pathlib.Path(out) if isinstance(out, str) else pathlib.Path(out[1])
            prefix = os.path.join(path.parts[0], path.parts[1])
            if prefix == 'dbfs:/Volumes':
                provider_prefix = prefix
        return getattr(sys.modules[__name__],
                       UPLOADERS[provider_prefix])(out, keep_local, progress_bar, retry, exist_ok)

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
                raise ValueError(f'Invalid `out` argument. It is either a string of ' +
                                 f'local/remote directory or a list of two strings with ' +
                                 f'[local, remote].')
            obj = urllib.parse.urlparse(out[1])
        if obj.scheme not in UPLOADERS:
            raise ValueError(f'Invalid Cloud provider prefix: {obj.scheme}.')

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
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
            retry (int): Number of times to retry uploading a file. Defaults to ``2``.
            exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
                exists and has contents. Defaults to ``False``.

        Raises:
            FileExistsError: Local directory must be empty.
        """
        self._validate(out)
        self.keep_local = keep_local
        self.progress_bar = progress_bar
        self.retry = retry

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
            if not exist_ok:
                raise FileExistsError(f'Directory is not empty: {self.local}')
            else:
                logger.warning(
                    f'Directory {self.local} exists and not empty. But continue to mkdir since exist_ok is set to be True.'
                )

        os.makedirs(self.local, exist_ok=True)

    def upload_file(self, filename: str):
        """Upload file from local instance to remote instance.

        Args:
            filename (str): File to upload.

        Raises:
            NotImplementedError: Override this method in your sub-class.
        """
        raise NotImplementedError(f'{type(self).__name__}.upload_file is not implemented')

    def list_objects(self, prefix: Optional[str] = None) -> Optional[List[str]]:
        """List all objects in the object store with the given prefix.

        Args:
            prefix (Optional[str], optional): The prefix to search for. Defaults to ``None``.

        Returns:
            List[str]: A list of object names that match the prefix.
        """
        raise NotImplementedError(f'{type(self).__name__}.list_objects is not implemented')

    def clear_local(self, local: str):
        """Remove the local file if it is enabled.

        Args:
            local (str): A local file path.
        """
        if not self.keep_local and os.path.isfile(local):
            os.remove(local)


class S3Uploader(CloudUploader):
    """Upload file from local machine to AWS S3 bucket (or any S3 compatible object store).

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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)

        import boto3
        from botocore.config import Config

        config = Config()
        # Create a session and use it to make our client. Unlike Resources and Sessions,
        # clients are generally thread-safe.
        session = boto3.session.Session()
        self.s3 = session.client('s3',
                                 config=config,
                                 endpoint_url=os.environ.get('S3_ENDPOINT_URL'))
        self.check_bucket_exists(self.remote)  # pyright: ignore

    def upload_file(self, filename: str):
        """Upload file from local instance to AWS S3 bucket.

        Args:
            filename (str): File to upload.
        """

        @retry(num_attempts=self.retry)
        def _upload_file():
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

        _upload_file()

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

    def list_objects(self, prefix: Optional[str] = None) -> Optional[List[str]]:
        """List all objects in the S3 object store with the given prefix.

        Args:
            prefix (Optional[str], optional): The prefix to search for. Defaults to ``None``.

        Returns:
            List[str]: A list of object names that match the prefix.
        """
        if prefix is None:
            prefix = ''

        obj = urllib.parse.urlparse(self.remote)
        bucket_name = obj.netloc
        prefix = os.path.join(str(obj.path).lstrip('/'), prefix)

        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        try:
            return [obj['Key'] for page in pages for obj in page['Contents']]
        except KeyError:
            return []


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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)
        if 'GCS_KEY' in os.environ and 'GCS_SECRET' in os.environ:
            import boto3

            # Create a session and use it to make our client. Unlike Resources and Sessions,
            # clients are generally thread-safe.
            session = boto3.session.Session()
            self.gcs_client = session.client(
                's3',
                region_name='auto',
                endpoint_url='https://storage.googleapis.com',
                aws_access_key_id=os.environ['GCS_KEY'],
                aws_secret_access_key=os.environ['GCS_SECRET'],
            )
            self.authentication = GCSAuthentication.HMAC
        else:
            from google.auth import default as default_auth
            from google.auth.exceptions import DefaultCredentialsError
            from google.cloud.storage import Client
            try:
                credentials, _ = default_auth()
                self.gcs_client = Client(credentials=credentials)
                self.authentication = GCSAuthentication.SERVICE_ACCOUNT
            except (DefaultCredentialsError, EnvironmentError):
                raise ValueError(GCS_ERROR_NO_AUTHENTICATION)

        self.check_bucket_exists(self.remote)  # pyright: ignore

    def upload_file(self, filename: str) -> None:
        """Upload file from local instance to Google Cloud Storage bucket.

        Args:
            filename (str): File to upload.
        """

        @retry(num_attempts=self.retry)
        def _upload_file():
            local_filename = os.path.join(self.local, filename)
            remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
            obj = urllib.parse.urlparse(remote_filename)
            logger.debug(f'Uploading to {remote_filename}')

            if self.authentication == GCSAuthentication.HMAC:
                file_size = os.stat(local_filename).st_size
                with tqdm.tqdm(
                        total=file_size,
                        unit='B',
                        unit_scale=True,
                        desc=f'Uploading to {remote_filename}',
                        disable=(not self.progress_bar),
                ) as pbar:
                    self.gcs_client.upload_file(
                        local_filename,
                        obj.netloc,
                        obj.path.lstrip('/'),
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                    )
            elif self.authentication == GCSAuthentication.SERVICE_ACCOUNT:
                from google.cloud.storage import Blob, Bucket

                blob = Blob(obj.path.lstrip('/'), Bucket(self.gcs_client, obj.netloc))
                blob.upload_from_filename(local_filename)

            self.clear_local(local=local_filename)

        _upload_file()

    def check_bucket_exists(self, remote: str) -> None:
        """Raise an exception if the bucket does not exist.

        Args:
            remote (str): GCS bucket path.

        Raises:
            error: Bucket does not exist.
        """
        bucket_name = urllib.parse.urlparse(remote).netloc

        if self.authentication == GCSAuthentication.HMAC:
            from botocore.exceptions import ClientError

            try:
                self.gcs_client.head_bucket(Bucket=bucket_name)
            except ClientError as error:
                if (error.response['Error']['Code'] == BOTOCORE_CLIENT_ERROR_CODES):
                    error.args = (f'Either bucket `{bucket_name}` does not exist! ' +
                                  f'or check the bucket permission.',)
                raise error
        elif self.authentication == GCSAuthentication.SERVICE_ACCOUNT:
            self.gcs_client.get_bucket(bucket_name)

    def list_objects(self, prefix: Optional[str] = None) -> Optional[List[str]]:
        """List all objects in the GCS object store with the given prefix.

        Args:
            prefix (Optional[str], optional): The prefix to search for. Defaults to None.

        Returns:
            List[str]: A list of object names that match the prefix.
        """
        if prefix is None:
            prefix = ''

        obj = urllib.parse.urlparse(self.remote)
        bucket_name = obj.netloc

        if self.authentication == GCSAuthentication.HMAC:
            prefix = os.path.join(str(obj.path).lstrip('/'), prefix)
            paginator = self.gcs_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            try:
                return [obj['Key'] for page in pages for obj in page['Contents']]
            except KeyError:
                return []
        elif self.authentication == GCSAuthentication.SERVICE_ACCOUNT:
            prefix = os.path.join(str(obj.path).lstrip('/'), prefix)
            return [
                b.name for b in self.gcs_client.get_bucket(bucket_name).list_blobs(prefix=prefix)
            ]


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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)

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

        @retry(num_attempts=self.retry)
        def _upload_file():
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

        _upload_file()

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

    def list_objects(self, prefix: Optional[str] = None) -> Optional[List[str]]:
        """List all objects in the OCI object store with the given prefix.

        Args:
            prefix (Optional[str], optional): The prefix to search for. Defaults to ``None``.

        Returns:
            List[str]: A list of object names that match the prefix.
        """
        if prefix is None:
            prefix = ''

        obj = urllib.parse.urlparse(self.remote)
        bucket_name = obj.netloc.split('@' + self.namespace)[0]
        prefix = os.path.join(str(obj.path).strip('/'), prefix)

        object_names = []
        next_start_with = None
        response_complete = False
        try:
            while not response_complete:
                response = self.client.list_objects(namespace_name=self.namespace,
                                                    bucket_name=bucket_name,
                                                    prefix=prefix,
                                                    start=next_start_with).data
                object_names.extend([resp_obj.name for resp_obj in response.objects])
                next_start_with = response.next_start_with
                if not next_start_with:
                    response_complete = True
            return object_names
        except Exception as e:
            if isinstance(e, oci.exceptions.ServiceError):
                if e.status == 404:  # type: ignore
                    if e.code == 'ObjectNotFound':  # type: ignore
                        raise FileNotFoundError(f'Object {bucket_name}/{prefix} not found. {e.message}') from e  # type: ignore
                    if e.code == 'BucketNotFound':  # type: ignore
                        raise ValueError(f'Bucket {bucket_name} not found. {e.message}') from e  # type: ignore
                    raise e
            raise e
        return []


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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)

        from azure.storage.blob import BlobServiceClient

        # Create a session and use it to make our client. Unlike Resources and Sessions,
        # clients are generally thread-safe.
        self.azure_service = BlobServiceClient(
            account_url=f"https://{os.environ['AZURE_ACCOUNT_NAME']}.blob.core.windows.net",
            credential=os.environ['AZURE_ACCOUNT_ACCESS_KEY'],
        )
        self.check_bucket_exists(self.remote)  # pyright: ignore

    def upload_file(self, filename: str):
        """Upload file from local instance to Microsoft Azure bucket.

        Args:
            filename (str): File to upload.
        """

        @retry(num_attempts=self.retry)
        def _upload_file():
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

        _upload_file()

    def check_bucket_exists(self, remote: str):
        """Raise an exception if the bucket does not exist.

        Args:
            remote (str): azure bucket path.

        Raises:
            error: Bucket does not exist.
        """
        bucket_name = urllib.parse.urlparse(remote).netloc
        if self.azure_service.get_container_client(container=bucket_name).exists() is False:
            raise FileNotFoundError(
                f'Either bucket `{bucket_name}` does not exist! ' +
                f'or check the bucket permission.',)


class AzureDataLakeUploader(CloudUploader):
    """Upload file from local machine to Microsoft Azure DataLake.

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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)

        from azure.storage.filedatalake import DataLakeServiceClient

        # Create a session and use it to make our client. Unlike Resources and Sessions,
        # clients are generally thread-safe.
        self.azure_service = DataLakeServiceClient(
            account_url=f"https://{os.environ['AZURE_ACCOUNT_NAME']}.dfs.core.windows.net",
            credential=os.environ['AZURE_ACCOUNT_ACCESS_KEY'])
        self.check_container_exists(self.remote)  # pyright: ignore

    def upload_file(self, filename: str):
        """Upload file from local instance to Azure DataLalke container.

        Args:
            filename (str): File to upload.
        """

        @retry(num_attempts=self.retry)
        def _upload_file():
            local_filename = os.path.join(self.local, filename)
            local_filename = local_filename.replace('\\', '/')
            remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
            remote_filename = remote_filename.replace('\\', '/')
            obj = urllib.parse.urlparse(remote_filename)
            logger.debug(f'Uploading to {remote_filename}')
            file_size = os.stat(local_filename).st_size
            file_client = self.azure_service.get_file_client(file_system=obj.netloc,
                                                             file_path=obj.path.lstrip('/'))

            with tqdm.tqdm(total=file_size,
                           unit='B',
                           unit_scale=True,
                           desc=f'Uploading to {remote_filename}',
                           disable=(not self.progress_bar)) as pbar:
                with open(local_filename, 'rb') as data:
                    file_client.upload_data(data=data, overwrite=True)
                    pbar.update(file_size)
            self.clear_local(local=local_filename)

        _upload_file()

    def check_container_exists(self, remote: str):
        """Raise an exception if the container does not exist.

        Args:
            remote (str): azure container path.

        Raises:
            error: Container does not exist.
        """
        container_name = urllib.parse.urlparse(remote).netloc
        if self.azure_service.get_file_system_client(file_system=container_name).exists() is False:
            raise FileNotFoundError(
                f'Either container `{container_name}` does not exist! ' +
                f'or check the container permission.',)


class DatabricksUploader(CloudUploader):
    """Parent class for uploading files from local machine to a Databricks workspace.

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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)
        self.client = self._create_workspace_client()

    def _create_workspace_client(self):
        try:
            from databricks.sdk import WorkspaceClient
            return WorkspaceClient()
        except ImportError as e:
            e.msg = get_import_exception_message(e.name, 'databricks')  # pyright: ignore
            raise e


class DatabricksUnityCatalogUploader(DatabricksUploader):
    """Upload file from local machine to Databricks Unity Catalog.

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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)

    def upload_file(self, filename: str):
        """Upload file from local instance to Databricks Unity Catalog.

        Args:
            filename (str): Relative filepath to copy.
        """

        @retry(num_attempts=self.retry)
        def _upload_file():
            local_filename = os.path.join(self.local, filename)
            local_filename = local_filename.replace('\\', '/')
            remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
            remote_filename = remote_filename.replace('\\', '/')
            remote_filename_wo_prefix = urllib.parse.urlparse(remote_filename).path
            with open(local_filename, 'rb') as f:
                self.client.files.upload(remote_filename_wo_prefix, f)

        _upload_file()


class DBFSUploader(DatabricksUploader):
    """Upload file from local machine to Databricks File System (DBFS).

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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)
        self.dbfs_path = self.remote.lstrip('dbfs:')  # pyright: ignore
        self.check_folder_exists()

    def upload_file(self, filename: str):
        """Upload file from local instance to DBFS. Does not overwrite.

        Args:
            filename (str): Relative filepath to copy.
        """

        @retry(num_attempts=self.retry)
        def _upload_file():
            local_filename = os.path.join(self.local, filename)
            local_filename = local_filename.replace('\\', '/')
            remote_filename = os.path.join(self.dbfs_path, filename)
            remote_filename = remote_filename.replace('\\', '/')
            file_path = urllib.parse.urlparse(remote_filename)
            with open(local_filename, 'rb') as f:
                self.client.dbfs.upload(file_path.path, f)

        _upload_file()

    def check_folder_exists(self):
        """Raise an exception if the DBFS folder does not exist.

        Raises:
            error: Folder does not exist.
        """
        from databricks.sdk.core import DatabricksError
        try:
            if not self.client.dbfs.exists(self.dbfs_path):
                raise FileNotFoundError(f'Databricks File System path {self.dbfs_path} not found')
        except DatabricksError as e:
            if e.error_code == 'PERMISSION_DENIED':
                e.args = (
                    f'Ensure the file path or credentials are set correctly. For ' +
                    f'Databricks Unity Catalog, file path must starts with `dbfs:/Volumes` ' +
                    f'and for Databricks File System, file path must starts with `dbfs`. ' +
                    e.args[0],)
            raise e


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
        retry (int): Number of times to retry uploading a file. Defaults to ``2``.
        exist_ok (bool): When exist_ok = False, raise error if the local part of ``out`` already
            exists and has contents. Defaults to ``False``.
    """

    def __init__(self,
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 progress_bar: bool = False,
                 retry: int = 2,
                 exist_ok: bool = False) -> None:
        super().__init__(out, keep_local, progress_bar, retry, exist_ok)
        # Create remote directory if it doesn't exist
        if self.remote:
            os.makedirs(self.remote, exist_ok=True)

    def upload_file(self, filename: str):
        """Copy file from one local path to another local path.

        Args:
            filename (str): Relative filepath to copy.
        """

        @retry(num_attempts=self.retry)
        def _upload_file():
            if self.remote:
                local_filename = os.path.join(self.local, filename)
                remote_filename = os.path.join(self.remote, filename)  # pyright: ignore
                logger.debug(f'Copying to {remote_filename}')
                shutil.copy(local_filename, remote_filename)
                self.clear_local(local=local_filename)

        _upload_file()

    def list_objects(self, prefix: Optional[str] = None) -> List[str]:
        """List all objects locally with the given prefix.

        Args:
            prefix (Optional[str], optional): The prefix to search for. Defaults to ``None``.

        Returns:
            List[str]: A list of object names that match the prefix.
        """
        if prefix is None:
            prefix = ''
        file_paths = []
        for dirpath, _, files in os.walk(os.path.join(self.local, prefix)):
            for file in files:
                file_paths.append(os.path.join(dirpath, file))
        return file_paths
