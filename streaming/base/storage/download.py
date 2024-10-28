# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shard downloading from various storage providers."""

import abc
import os
import pathlib
import shutil
import sys
import urllib.parse
from typing import Any, Optional

from streaming.base.util import get_import_exception_message

__all__ = [
    'CloudDownloader',
    'S3Downloader',
    'SFTPDownloader',
    'GCSDownloader',
    'OCIDownloader',
    'AzureDownloader',
    'AzureDataLakeDownloader',
    'HFDownloader',
    'DatabricksUnityCatalogDownloader',
    'DBFSDownloader',
    'AlipanDownloader',
    'LocalDownloader',
]

BOTOCORE_CLIENT_ERROR_CODES = {'403', '404', 'NoSuchKey'}

GCS_ERROR_NO_AUTHENTICATION = """\
Either set the environment variables `GCS_KEY` and `GCS_SECRET` or use any of the methods in \
https://cloud.google.com/docs/authentication/external/set-up-adc to set up Application Default \
Credentials. See also https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/gcp.html.
"""


class CloudDownloader(abc.ABC):
    """Download files from a remote storage to a local filesystem."""

    @classmethod
    def get(cls, remote_dir: Optional[str]) -> 'CloudDownloader':
        """Get the downloader for the remote path.

        Args:
            remote (str): Remote path.

        Returns:
            CloudDownloader: Downloader for the remote path.

        Raises:
            ValueError: If the remote path is not provided or remote path is not supported.
        """
        if not remote_dir:
            raise ValueError('Remote path must be provided.')

        prefix = urllib.parse.urlparse(remote_dir).scheme
        if prefix == 'dbfs' and remote_dir.startswith('dbfs:/Volumes'):
            prefix = 'dbfs-uc'

        if prefix not in DOWNLOADER_MAPPINGS:
            raise ValueError(f'Unsupported remote path: {remote_dir}')

        return DOWNLOADER_MAPPINGS[prefix]()

    @classmethod
    def direct_download(cls, remote: Optional[str], local: str, timeout: float = 60.0) -> None:
        """Directly download a file from remote storage to local filesystem.

        Args:
            remote (str): Remote path.
            local (str): Local path.
            timeout (float): How long to wait for file to download before raising an exception.

        Raises:
            ValueError: If the remote path is not provided or remote path is not supported.
        """
        downloader = cls.get(remote)
        downloader.download(remote, local, timeout)
        downloader.clean_up()

    def download(self, remote: Optional[str], local: str, timeout: float = 60.0) -> None:
        """Download a file from remote storage to local filesystem.

        Args:
            remote (str): Remote path.
            local (str): Local path.
            timeout (float): How long to wait for file to download before raising an exception.

        Raises:
            ValueError: If the remote path does not contain the expected prefix or remote is
                not provided.
        """
        if not remote:
            raise ValueError(
                'In the absence of local dataset, path to remote dataset must be provided')

        if sys.platform == 'win32':
            remote = pathlib.PureWindowsPath(remote).as_posix()
            local = pathlib.PureWindowsPath(local).as_posix()

        self._validate_remote_path(remote)
        self._download_impl(remote, local, timeout)

    @staticmethod
    @abc.abstractmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: Prefix of the downloader.
        """

    @abc.abstractmethod
    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        raise NotImplementedError

    @abc.abstractmethod
    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        """Implementation of the download function.

        Args:
            remote (str): Remote path.
            local (str): Local path.
            timeout (float): How long to wait for file to download before raising an exception.
        """
        raise NotImplementedError

    def _validate_remote_path(self, remote: str) -> None:
        """Validate the remote path.

        Args:
            remote (str): Remote path.

        Raises:
            ValueError: If the remote path does not contain the expected prefix.
        """
        # Single slash because some protocols only use a single slash, not double
        url_scheme = urllib.parse.urlparse(remote).scheme

        if url_scheme != self.prefix():
            raise ValueError(
                f'Expected remote path to start with url scheme of `{url_scheme}`, got {remote}.')


class S3Downloader(CloudDownloader):
    """Download files from AWS S3 to local filesystem."""

    def __init__(self):
        super().__init__()

        self._s3_client: Optional[Any] = None  # Hard to tell exactly what the typing of this is
        self._requester_pays_buckets = [
            name.strip()
            for name in os.environ.get('MOSAICML_STREAMING_AWS_REQUESTER_PAYS', '').split(',')
        ]

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns s3.
        """
        return 's3'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        self._s3_client = None

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        """Implementation of the download function."""
        from boto3.s3.transfer import TransferConfig
        from botocore.exceptions import ClientError, NoCredentialsError

        if self._s3_client is None:
            try:
                self._create_s3_client(timeout=timeout)
            except NoCredentialsError:
                # Public S3 buckets without credentials
                self._create_s3_client(unsigned=True, timeout=timeout)
            except Exception as e:
                raise e
        assert self._s3_client is not None

        obj = urllib.parse.urlparse(remote)
        extra_args = {}
        # When enabled, the requester instead of the bucket owner pays the cost of the request
        # and the data download from the bucket.
        if obj.netloc in self._requester_pays_buckets:
            extra_args['RequestPayer'] = 'requester'

        try:
            self._s3_client.download_file(obj.netloc,
                                          obj.path.lstrip('/'),
                                          local,
                                          ExtraArgs=extra_args,
                                          Config=TransferConfig(use_threads=False))
        except ClientError as e:
            if e.response['Error']['Code'] in BOTOCORE_CLIENT_ERROR_CODES:
                e.args = (
                    f'Object {remote} not found! Either check the bucket path or the bucket ' +
                    'permission. If the bucket is a requester pays bucket, then provide the ' +
                    'bucket name to the environment variable ' +
                    '`MOSAICML_STREAMING_AWS_REQUESTER_PAYS`.',)
                raise e
            elif e.response['Error']['Code'] == '400':
                # Recreate s3 client as public
                # TODO(ethantang-db) not sure if this is the right way, can there be a bucket with
                # a mixed amount of content that is both private and public? If so, then we might
                # need two clients, one for public and one for private
                self._create_s3_client(unsigned=True, timeout=timeout)
                self._download_impl(remote, local, timeout)
            else:
                raise e
        except Exception as e:
            raise e

    def _create_s3_client(self, unsigned: bool = False, timeout: float = 60.0) -> Any:
        """Create an S3 client."""
        from boto3.session import Session
        from botocore import UNSIGNED
        from botocore.config import Config

        retries = {
            'mode': 'adaptive',
        }
        if unsigned:
            # Client will be using unsigned mode in which public
            # resources can be accessed without credentials
            config = Config(read_timeout=timeout, signature_version=UNSIGNED, retries=retries)
        else:
            config = Config(read_timeout=timeout, retries=retries)

        # Creating the session
        self._s3_client = Session().client('s3',
                                           config=config,
                                           endpoint_url=os.environ.get('S3_ENDPOINT_URL'))


class SFTPDownloader(CloudDownloader):
    """Download files from SFTP to local filesystem."""

    def __init__(self):
        """Initialize the downloader."""
        super().__init__()

        from urllib.parse import SplitResult

        from paramiko import SSHClient

        self._ssh_client: Optional[SSHClient] = None
        self._url: Optional[SplitResult] = None

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns sftp.
        """
        return 'sftp'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        if self._ssh_client is not None:
            self._ssh_client.close()
            self._ssh_client = None

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        """Implementation of the download function."""
        url = urllib.parse.urlsplit(remote)
        local_tmp = local + '.tmp'

        if os.path.exists(local_tmp):
            os.remove(local_tmp)

        if self._ssh_client is None:
            self._create_ssh_client(url)
        assert self._ssh_client is not None

        sftp_client = self._ssh_client.open_sftp()
        sftp_client.get(remotepath=url.path, localpath=local_tmp)
        os.rename(local_tmp, local)

    def _validate_remote_path(self, remote: str) -> None:
        """Validates the remote path for sftp client."""
        super()._validate_remote_path(remote)

        url = urllib.parse.urlsplit(remote)
        if url.hostname is None:
            raise ValueError('If specifying a URI, the URI must include the hostname.')
        if url.query or url.fragment:
            raise ValueError('Query and fragment parameters are not supported as part of a URI.')

        if self._url is None:
            self._url = url
            return

        assert self._url.hostname == url.hostname
        assert self._url.port == url.port or (self._url.port is None and url.port == 22)
        assert self._url.username == url.username
        assert self._url.password == url.password

    def _create_ssh_client(self, url: urllib.parse.SplitResult) -> None:
        """Create an SSH client."""
        assert url.hostname is not None, 'Hostname must be provided for SFTP download.'

        from paramiko import SSHClient

        # Get SSH key file if specified
        key_filename = os.environ.get('COMPOSER_SFTP_KEY_FILE', None)
        known_hosts_filename = os.environ.get('COMPOSER_SFTP_KNOWN_HOSTS_FILE', None)

        self._ssh_client = SSHClient()
        self._ssh_client.load_system_host_keys(known_hosts_filename)
        self._ssh_client.connect(
            hostname=url.hostname,  # ignore: reportGeneralTypeIssues
            port=url.port if url.port is not None else 22,
            username=url.username,
            password=url.password,
            key_filename=key_filename,
        )


class GCSDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

        from google.cloud.storage import Client

        self._gcs_client: Optional[Any | Client] = None

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns gs.
        """
        return 'gs'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        self._gcs_client = None

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        from google.cloud.storage import Client

        if self._gcs_client is None:
            self._create_gcs_client()
        assert self._gcs_client is not None

        url = urllib.parse.urlparse(remote)

        if isinstance(self._gcs_client, Client):
            from google.cloud.storage import Blob, Bucket
            blob = Blob(url.path.lstrip('/'), Bucket(self._gcs_client, url.netloc))
            blob.download_to_filename(local)
        else:
            from boto3.s3.transfer import TransferConfig
            from botocore.exceptions import ClientError

            try:
                self._gcs_client.download_file(url.netloc,
                                               url.path.lstrip('/'),
                                               local,
                                               Config=TransferConfig(use_threads=False))
            except ClientError as e:
                if e.response['Error']['Code'] in BOTOCORE_CLIENT_ERROR_CODES:
                    raise FileNotFoundError(f'Object {remote} not found') from e
            except Exception as e:
                raise e

    def _create_gcs_client(self) -> None:
        if 'GCS_KEY' in os.environ and 'GCS_SECRET' in os.environ:
            from boto3.session import Session

            self._gcs_client = Session().client('s3',
                                                region_name='auto',
                                                endpoint_url='https://storage.googleapis.com',
                                                aws_access_key_id=os.environ['GCS_KEY'],
                                                aws_secret_access_key=os.environ['GCS_SECRET'])
        else:
            from google.auth import default as default_auth
            from google.auth.exceptions import DefaultCredentialsError
            from google.cloud.storage import Client

            try:
                credentials, _ = default_auth()
                self._gcs_client = Client(credentials=credentials)
            except (DefaultCredentialsError, EnvironmentError):
                raise ValueError(GCS_ERROR_NO_AUTHENTICATION)


class OCIDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

        import oci
        self._oci_client: Optional[oci.object_storage.ObjectStorageClient] = None

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns oci.
        """
        return 'oci'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        self._oci_client = None

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        if self._oci_client is None:
            self._create_oci_client()
        assert self._oci_client is not None

        url = urllib.parse.urlparse(remote)
        bucket_name = url.netloc.split('@' + self._oci_client.get_namespace().data)[0]
        object_path = url.path.strip('/')
        object_details = self._oci_client.get_object(self._oci_client.get_namespace().data,
                                                     bucket_name, object_path)
        local_tmp = local + '.tmp'
        with open(local_tmp, 'wb') as f:
            for chunk in object_details.data.raw.stream(2048**2, decode_content=False):
                f.write(chunk)
        os.rename(local_tmp, local)

    def _create_oci_client(self) -> None:
        import oci

        config = oci.config.from_file()
        self._oci_client = oci.object_storage.ObjectStorageClient(
            config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)


class HFDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns hf.
        """
        return 'hf'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        pass

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        from huggingface_hub import hf_hub_download

        _, _, _, repo_org, repo_name, path = remote.split('/', 5)
        local_dirname = os.path.dirname(local)
        hf_hub_download(repo_id=f'{repo_org}/{repo_name}',
                        filename=path,
                        repo_type='dataset',
                        local_dir=local_dirname)

        downloaded_name = os.path.join(local_dirname, path)
        os.rename(downloaded_name, local)


class AzureDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

        from azure.storage.blob import BlobServiceClient

        self._azure_client: Optional[BlobServiceClient] = None

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns azure.
        """
        return 'azure'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        self._azure_client = None

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        if self._azure_client is None:
            self._create_azure_client()
        assert self._azure_client is not None

        obj = urllib.parse.urlparse(remote)
        file_path = obj.path.lstrip('/').split('/')
        container_name = file_path[0]
        blob_name = os.path.join(*file_path[1:])
        blob_client = self._azure_client.get_blob_client(container=container_name, blob=blob_name)
        local_tmp = local + '.tmp'
        with open(local_tmp, 'wb') as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
        os.rename(local_tmp, local)

    def _create_azure_client(self) -> None:
        from azure.storage.blob import BlobServiceClient

        self._azure_client = BlobServiceClient(
            account_url=f"https://{os.environ['AZURE_ACCOUNT_NAME']}.blob.core.windows.net",
            credential=os.environ['AZURE_ACCOUNT_ACCESS_KEY'])


class AzureDataLakeDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

        from azure.storage.filedatalake import DataLakeServiceClient

        self._azure_dl_client: Optional[DataLakeServiceClient] = None

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns adl.
        """
        return 'azure-dl'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        self._azure_dl_client = None

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        if self._azure_dl_client is None:
            self._create_azure_dl_client()
        assert self._azure_dl_client is not None

        obj = urllib.parse.urlparse(remote)
        try:
            file_client = self._azure_dl_client.get_file_client(file_system=obj.netloc,
                                                                file_path=obj.path.lstrip('/'))
            local_tmp = local + '.tmp'
            with open(local_tmp, 'wb') as f:
                file_data = file_client.download_file()
                file_data.readinto(f)
            os.rename(local_tmp, local)
        except ResourceNotFoundError as e:
            raise FileNotFoundError(f'Object {remote} not found.') from e
        except Exception as e:
            raise e

    def _create_azure_dl_client(self) -> None:
        from azure.storage.filedatalake import DataLakeServiceClient

        self._azure_dl_client = DataLakeServiceClient(
            account_url=f"https://{os.environ['AZURE_ACCOUNT_NAME']}.dfs.core.windows.net",
            credential=os.environ['AZURE_ACCOUNT_ACCESS_KEY'],
        )


class DatabricksUnityCatalogDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as e:
            e.msg = get_import_exception_message(e.name, 'databricks')  # pyright: ignore
            raise e

        self._db_uc_client: Optional[WorkspaceClient] = None

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns dbfs-uc.
        """
        return 'dbfs-uc'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        self._db_uc_client = None

    def _validate_remote_path(self, remote: str):
        path = pathlib.Path(remote)
        provider_prefix = os.path.join(path.parts[0], path.parts[1])
        if provider_prefix != 'dbfs:/Volumes':
            raise ValueError(
                'Expected path prefix to be `dbfs:/Volumes` if it is a Databricks Unity ' +
                f'Catalog, instead, got {provider_prefix} for remote={remote}.')

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        from databricks.sdk.core import DatabricksError

        if self._db_uc_client is None:
            self._create_db_uc_client()
        assert self._db_uc_client is not None

        file_path = urllib.parse.urlparse(remote)
        local_tmp = local + '.tmp'
        response = self._db_uc_client.files.download(file_path.path).contents

        assert response is not None

        try:
            with response:
                with open(local_tmp, 'wb') as f:
                    # Download data in chunks to avoid memory issues.
                    for chunk in iter(lambda: response.read(64 * 1024 * 1024), b''):
                        f.write(chunk)
        except DatabricksError as e:
            if e.error_code == 'REQUEST_LIMIT_EXCEEDED':
                e.args = (
                    'Dataset download request has been rejected due to too many concurrent ' +
                    'operations. Increase the `download_retry` value to retry downloading ' +
                    'a file.',)
            if e.error_code == 'NOT_FOUND':
                raise FileNotFoundError(f'Object {remote} not found.') from e
            raise e
        except Exception as e:
            raise e
        os.rename(local_tmp, local)

    def _create_db_uc_client(self) -> None:
        from databricks.sdk import WorkspaceClient
        self._db_uc_client = WorkspaceClient()


class DBFSDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as e:
            e.msg = get_import_exception_message(e.name, 'databricks')  # pyright: ignore
            raise e

        self._dbfs_client: Optional[WorkspaceClient] = None

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns dbfs.
        """
        return 'dbfs'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        self._dbfs_client = None

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        from databricks.sdk.core import DatabricksError

        if self._dbfs_client is None:
            self._create_dbfs_client()
        assert self._dbfs_client is not None

        file_path = urllib.parse.urlparse(remote)
        local_tmp = local + '.tmp'
        response = self._dbfs_client.files.download(file_path.path).contents

        assert response is not None

        try:
            with response:
                with open(local_tmp, 'wb') as f:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        f.write(chunk)
        except DatabricksError as e:
            if e.error_code == 'PERMISSION_DENIED':
                e.args = (
                    f'Ensure the file path or credentials are set correctly. For ' +
                    f'Databricks Unity Catalog, file path must starts with `dbfs:/Volumes` ' +
                    f'and for Databricks File System, file path must starts with `dbfs`. ' +
                    e.args[0],)
            raise e
        except Exception as e:
            raise e
        os.rename(local_tmp, local)

    def _create_dbfs_client(self) -> None:
        from databricks.sdk import WorkspaceClient
        self._dbfs_client = WorkspaceClient()


class AlipanDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns alipan.
        """
        return 'alipan'

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        pass

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        from alipcs_py.alipcs import AliPCSApiMix
        from alipcs_py.commands.download import download_file

        web_refresh_token = os.environ['ALIPAN_WEB_REFRESH_TOKEN']
        web_token_type = 'Bearer'
        alipan_encrypt_password = os.environ.get('ALIPAN_ENCRYPT_PASSWORD', '').encode()

        api = AliPCSApiMix(web_refresh_token, web_token_type=web_token_type)

        obj = urllib.parse.urlparse(remote)
        if obj.scheme != 'alipan':
            raise ValueError(
                f'Expected obj.scheme to be `alipan`, instead, got {obj.scheme} for remote={remote}'
            )
        if obj.netloc != '':
            raise ValueError(
                f'Expected remote to be alipan:///path/to/some, instead, got remote={remote}')

        remote_path = obj.path
        filename = pathlib.PosixPath(remote_path).name
        localdir = pathlib.Path(local).parent

        remote_pcs_file = api.get_file(remotepath=remote_path)
        if remote_pcs_file is None:
            raise FileNotFoundError(f'Object {remote} not found.')

        download_file(
            api,
            remote_pcs_file,
            localdir=localdir,
            downloader='me',
            concurrency=1,
            show_progress=False,
            encrypt_password=alipan_encrypt_password,
        )
        os.rename(localdir / filename, local)


class LocalDownloader(CloudDownloader):

    def __init__(self):
        super().__init__()

    @staticmethod
    def prefix() -> str:
        """Return the prefix of the downloader.

        Returns:
            str: returns file.
        """
        return ''

    def clean_up(self) -> None:
        """Clean up the downloader when it is done being used."""
        pass

    def _download_impl(self, remote: str, local: str, timeout: float) -> None:
        """Download a file from remote to local.

        Args:
            remote (str): Remote path (local or unix filesystem).
            local (str): Local path (local filesystem).
        """
        local_tmp = local + '.tmp'
        if os.path.exists(local_tmp):
            os.remove(local_tmp)
        shutil.copy(remote, local_tmp)
        os.rename(local_tmp, local)


def _register_cloud_downloader_subclasses() -> dict[str, type[CloudDownloader]]:
    """Register all CloudDownloader subclasses."""
    sub_classes = CloudDownloader.__subclasses__()

    downloader_mappings = {}

    for sub_class in sub_classes:
        downloader_mappings[sub_class.prefix()] = sub_class

    return downloader_mappings


DOWNLOADER_MAPPINGS = _register_cloud_downloader_subclasses()
