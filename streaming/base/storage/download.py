# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shard downloading from various storage providers."""

import os
import pathlib
import shutil
import urllib.parse
from time import sleep, time
from typing import Any, Dict, Optional

from streaming.base.util import get_import_exception_message

__all__ = [
    'download_from_s3',
    'download_from_sftp',
    'download_from_gcs',
    'download_from_oci',
    'download_from_azure',
    'download_from_azure_datalake',
    'download_from_hf',
    'download_from_databricks_unity_catalog',
    'download_from_dbfs',
    'download_from_alipan',
    'download_from_local',
]

BOTOCORE_CLIENT_ERROR_CODES = {'403', '404', 'NoSuchKey'}

GCS_ERROR_NO_AUTHENTICATION = """\
Either set the environment variables `GCS_KEY` and `GCS_SECRET` or use any of the methods in \
https://cloud.google.com/docs/authentication/external/set-up-adc to set up Application Default \
Credentials. See also https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/gcp.html.
"""


def download_from_s3(remote: str, local: str, timeout: float) -> None:
    """Download a file from remote AWS S3 (or any S3 compatible object store) to local.

    Args:
        remote (str): Remote path (S3).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for shard to download before raising an exception.
    """

    def _download_file(unsigned: bool = False,
                       extra_args: Optional[Dict[str, Any]] = None) -> None:
        """Download the file from AWS S3 bucket. The bucket can be either public or private.

        Args:
            unsigned (bool, optional):  Set to True if it is a public bucket.
                Defaults to ``False``.
            extra_args (Dict[str, Any], optional): Extra arguments supported by boto3.
                Defaults to ``None``.
        """
        retries = {
            'mode': 'adaptive',
        }
        if unsigned:
            # Client will be using unsigned mode in which public
            # resources can be accessed without credentials
            config = Config(read_timeout=timeout, signature_version=UNSIGNED, retries=retries)
        else:
            config = Config(read_timeout=timeout, retries=retries)

        if extra_args is None:
            extra_args = {}

        # Create a new session per thread
        session = boto3.session.Session()
        # Create a resource client using a thread's session object
        s3 = session.client('s3', config=config, endpoint_url=os.environ.get('S3_ENDPOINT_URL'))
        # Threads calling S3 operations return RuntimeError (cannot schedule new futures after
        # interpreter shutdown). Temporary solution is to have `use_threads` as `False`.
        # Issue: https://github.com/boto/boto3/issues/3113
        s3.download_file(obj.netloc,
                         obj.path.lstrip('/'),
                         local,
                         ExtraArgs=extra_args,
                         Config=TransferConfig(use_threads=False))

    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError

    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(
            f'Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote}')

    extra_args = {}
    # When enabled, the requester instead of the bucket owner pays the cost of the request
    # and the data download from the bucket.
    if os.environ.get('MOSAICML_STREAMING_AWS_REQUESTER_PAYS') is not None:
        requester_pays_buckets = os.environ.get(  # yapf: ignore
            'MOSAICML_STREAMING_AWS_REQUESTER_PAYS').split(',')  # pyright: ignore
        requester_pays_buckets = [name.strip() for name in requester_pays_buckets]
        if obj.netloc in requester_pays_buckets:
            extra_args['RequestPayer'] = 'requester'

    try:
        _download_file(extra_args=extra_args)
    except NoCredentialsError:
        # Public S3 buckets without credentials
        _download_file(unsigned=True, extra_args=extra_args)
    except ClientError as e:
        if e.response['Error']['Code'] in BOTOCORE_CLIENT_ERROR_CODES:
            e.args = (f'Object {remote} not found! Either check the bucket path or the bucket ' +
                      f'permission. If the bucket is a requester pays bucket, then provide the ' +
                      f'bucket name to the environment variable ' +
                      f'`MOSAICML_STREAMING_AWS_REQUESTER_PAYS`.',)
            raise e
        elif e.response['Error']['Code'] == '400':
            # Public S3 buckets without credentials
            _download_file(unsigned=True, extra_args=extra_args)
    except Exception:
        raise


def download_from_sftp(remote: str, local: str) -> None:
    """Download a file from remote SFTP server to local filepath.

    Authentication must be provided via username/password in the ``remote`` URI, or a valid SSH
    config, or a default key discoverable in ``~/.ssh/``.

    Args:
        remote (str): Remote path (SFTP).
        local (str): Local path (local filesystem).
    """
    from paramiko import SSHClient

    # Parse URL
    url = urllib.parse.urlsplit(remote)
    if url.scheme.lower() != 'sftp':
        raise ValueError('If specifying a URI, only the sftp scheme is supported.')
    if not url.hostname:
        raise ValueError('If specifying a URI, the URI must include the hostname.')
    if url.query or url.fragment:
        raise ValueError('Query and fragment parameters are not supported as part of a URI.')
    hostname = url.hostname
    port = url.port
    username = url.username
    password = url.password
    remote_path = url.path

    # Get SSH key file if specified
    key_filename = os.environ.get('COMPOSER_SFTP_KEY_FILE', None)
    known_hosts_filename = os.environ.get('COMPOSER_SFTP_KNOWN_HOSTS_FILE', None)

    # Default port
    port = port if port else 22

    # Local tmp
    local_tmp = local + '.tmp'
    if os.path.exists(local_tmp):
        os.remove(local_tmp)

    with SSHClient() as ssh_client:
        # Connect SSH Client
        ssh_client.load_system_host_keys(known_hosts_filename)
        ssh_client.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            key_filename=key_filename,
        )

        # SFTP Client
        sftp_client = ssh_client.open_sftp()
        sftp_client.get(remotepath=remote_path, localpath=local_tmp)
    os.rename(local_tmp, local)


def download_from_gcs(remote: str, local: str) -> None:
    """Download a file from remote GCS to local.

    Args:
        remote (str): Remote path (GCS).
        local (str): Local path (local filesystem).
    """
    from google.auth.exceptions import DefaultCredentialsError
    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 'gs':
        raise ValueError(
            f'Expected obj.scheme to be `gs`, instead, got {obj.scheme} for remote={remote}')

    if 'GCS_KEY' in os.environ and 'GCS_SECRET' in os.environ:
        _gcs_with_hmac(remote, local, obj)
    else:
        try:
            _gcs_with_service_account(local, obj)
        except (DefaultCredentialsError, EnvironmentError):
            raise ValueError(GCS_ERROR_NO_AUTHENTICATION)


def _gcs_with_hmac(remote: str, local: str, obj: urllib.parse.ParseResult) -> None:
    """Download a file from remote GCS to local using user level credentials.

    Args:
        remote (str): Remote path (GCS).
        local (str): Local path (local filesystem).
        obj (ParseResult): ParseResult object of remote.
    """
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError

    # Create a new session per thread
    session = boto3.session.Session()
    # Create a resource client using a thread's session object
    gcs_client = session.client('s3',
                                region_name='auto',
                                endpoint_url='https://storage.googleapis.com',
                                aws_access_key_id=os.environ['GCS_KEY'],
                                aws_secret_access_key=os.environ['GCS_SECRET'])
    try:
        # Threads calling S3 operations return RuntimeError (cannot schedule new futures after
        # interpreter shutdown). Temporary solution is to have `use_threads` as `False`.
        # Issue: https://github.com/boto/boto3/issues/3113
        gcs_client.download_file(obj.netloc,
                                 obj.path.lstrip('/'),
                                 local,
                                 Config=TransferConfig(use_threads=False))
    except ClientError as e:
        if e.response['Error']['Code'] in BOTOCORE_CLIENT_ERROR_CODES:
            raise FileNotFoundError(f'Object {remote} not found.') from e
    except Exception:
        raise


def _gcs_with_service_account(local: str, obj: urllib.parse.ParseResult) -> None:
    """Download a file from remote GCS to local using service account credentials.

    Args:
        local (str): Local path (local filesystem).
        obj (ParseResult): ParseResult object of remote path (GCS).
    """
    from google.auth import default as default_auth
    from google.cloud.storage import Blob, Bucket, Client

    credentials, _ = default_auth()
    gcs_client = Client(credentials=credentials)
    blob = Blob(obj.path.lstrip('/'), Bucket(gcs_client, obj.netloc))
    blob.download_to_filename(local)


def download_from_oci(remote: str, local: str) -> None:
    """Download a file from remote OCI to local.

    Args:
        remote (str): Remote path (OCI).
        local (str): Local path (local filesystem).
    """
    import oci
    config = oci.config.from_file()
    client = oci.object_storage.ObjectStorageClient(
        config=config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
    namespace = client.get_namespace().data
    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 'oci':
        raise ValueError(
            f'Expected obj.scheme to be `oci`, instead, got {obj.scheme} for remote={remote}')

    bucket_name = obj.netloc.split('@' + namespace)[0]
    # Remove leading and trailing forward slash from string
    object_path = obj.path.strip('/')
    object_details = client.get_object(namespace, bucket_name, object_path)
    local_tmp = local + '.tmp'
    with open(local_tmp, 'wb') as f:
        for chunk in object_details.data.raw.stream(2048**2, decode_content=False):
            f.write(chunk)
    os.rename(local_tmp, local)


def download_from_hf(remote: str, local: str) -> None:
    """Download a file from remote Hugging Face to local.

    Args:
        remote (str): Remote path (Hugging Face).
        local (str): Local path (local filesystem).
    """
    from huggingface_hub import hf_hub_download

    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 'hf':
        raise ValueError(f'Expected remote path to start with `hf://`, got {remote}.')

    _, _, _, repo_org, repo_name, path = remote.split('/', 5)
    local_dirname = os.path.dirname(local)
    hf_hub_download(repo_id=f'{repo_org}/{repo_name}',
                    filename=path,
                    repo_type='dataset',
                    local_dir=local_dirname)

    downloaded_name = os.path.join(local_dirname, path)
    os.rename(downloaded_name, local)


def download_from_azure(remote: str, local: str) -> None:
    """Download a file from remote Microsoft Azure to local.

    Args:
        remote (str): Remote path (azure).
        local (str): Local path (local filesystem).
    """
    from azure.core.exceptions import ResourceNotFoundError
    from azure.storage.blob import BlobServiceClient

    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 'azure':
        raise ValueError(
            f'Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote}')

    # Create a new session per thread
    service = BlobServiceClient(
        account_url=f"https://{os.environ['AZURE_ACCOUNT_NAME']}.blob.core.windows.net",
        credential=os.environ['AZURE_ACCOUNT_ACCESS_KEY'])
    try:
        blob_client = service.get_blob_client(container=obj.netloc, blob=obj.path.lstrip('/'))
        local_tmp = local + '.tmp'
        with open(local_tmp, 'wb') as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)
        os.rename(local_tmp, local)
    except ResourceNotFoundError as e:
        raise FileNotFoundError(f'Object {remote} not found.') from e
    except Exception:
        raise


def download_from_azure_datalake(remote: str, local: str) -> None:
    """Download a file from remote Microsoft Azure Data Lake to local.

    Args:
        remote (str): Remote path (azure).
        local (str): Local path (local filesystem).
    """
    from azure.core.exceptions import ResourceNotFoundError
    from azure.storage.filedatalake import DataLakeServiceClient

    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 'azure-dl':
        raise ValueError(
            f'Expected obj.scheme to be `azure-dl`, got {obj.scheme} for remote={remote}')

    # Create a new session per thread
    service = DataLakeServiceClient(
        account_url=f"https://{os.environ['AZURE_ACCOUNT_NAME']}.dfs.core.windows.net",
        credential=os.environ['AZURE_ACCOUNT_ACCESS_KEY'],
    )
    try:
        file_client = service.get_file_client(file_system=obj.netloc,
                                              file_path=obj.path.lstrip('/'))
        local_tmp = local + '.tmp'
        with open(local_tmp, 'wb') as my_file:
            file_data = file_client.download_file()
            file_data.readinto(my_file)
        os.rename(local_tmp, local)
    except ResourceNotFoundError as e:
        raise FileNotFoundError(f'Object {remote} not found.') from e
    except Exception:
        raise


def download_from_databricks_unity_catalog(remote: str, local: str) -> None:
    """Download a file from remote Databricks Unity Catalog to local.

    .. note::
        The Databricks UC Volume path must be of the form
        `dbfs:/Volumes/<catalog-name>/<schema-name>/<volume-name>/path`.

    Args:
        remote (str): Remote path (Databricks Unity Catalog).
        local (str): Local path (local filesystem).
    """
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import DatabricksError
    except ImportError as e:
        e.msg = get_import_exception_message(e.name, 'databricks')  # pyright: ignore
        raise e

    path = pathlib.Path(remote)
    provider_prefix = os.path.join(path.parts[0], path.parts[1])
    if provider_prefix != 'dbfs:/Volumes':
        raise ValueError(
            f'Expected path prefix to be `dbfs:/Volumes` if it is a Databricks Unity Catalog, ' +
            f'instead, got {provider_prefix} for remote={remote}.')

    client = WorkspaceClient()
    file_path = urllib.parse.urlparse(remote)
    local_tmp = local + '.tmp'
    try:
        with client.files.download(file_path.path).contents as response:
            with open(local_tmp, 'wb') as f:
                # Download data in chunks to avoid memory issues.
                for chunk in iter(lambda: response.read(64 * 1024 * 1024), b''):
                    f.write(chunk)
    except DatabricksError as e:
        if e.error_code == 'REQUEST_LIMIT_EXCEEDED':
            e.args = (f'Dataset download request has been rejected due to too many concurrent ' +
                      f'operations. Increase the `download_retry` value to retry downloading ' +
                      f'a file.',)
        if e.error_code == 'NOT_FOUND':
            raise FileNotFoundError(f'Object {remote} not found.') from e
        raise e
    os.rename(local_tmp, local)


def download_from_dbfs(remote: str, local: str) -> None:
    """Download a file from remote Databricks File System to local.

    Args:
        remote (str): Remote path (dbfs).
        local (str): Local path (local filesystem).
    """
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import DatabricksError
    except ImportError as e:
        e.msg = get_import_exception_message(e.name, 'databricks')  # pyright: ignore
        raise e

    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 'dbfs':
        raise ValueError(f'Expected path prefix to be `dbfs` if it is a Databricks File System, ' +
                         f'instead, got {obj.scheme} for remote={remote}.')

    client = WorkspaceClient()
    file_path = urllib.parse.urlparse(remote)
    local_tmp = local + '.tmp'
    try:
        with client.dbfs.download(file_path.path) as response:
            with open(local_tmp, 'wb') as f:
                # Multiple shard files are getting downloaded in parallel, so we need to
                # read the data in chunks to avoid memory issues.
                # Read 1MB of data at a time since that's the max limit it can read at a time.
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    f.write(chunk)
    except DatabricksError as e:
        if e.error_code == 'PERMISSION_DENIED':
            e.args = (f'Ensure the file path or credentials are set correctly. For ' +
                      f'Databricks Unity Catalog, file path must starts with `dbfs:/Volumes` ' +
                      f'and for Databricks File System, file path must starts with `dbfs`. ' +
                      e.args[0],)
        raise e
    os.rename(local_tmp, local)


def download_from_alipan(remote: str, local: str) -> None:
    """Download a file from remote Alipan to local.

    Args:
        remote (str): Remote path (Alipan).
        local (str): Local path (local filesystem).
    """
    from alipcs_py.alipcs import AliPCSApiMix
    from alipcs_py.commands.download import download_file

    web_refresh_token = os.environ['ALIPAN_WEB_REFRESH_TOKEN']
    web_token_type = 'Bearer'
    alipan_encrypt_password = os.environ.get('ALIPAN_ENCRYPT_PASSWORD', '').encode()

    api = AliPCSApiMix(web_refresh_token, web_token_type=web_token_type)

    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 'alipan':
        raise ValueError(
            f'Expected obj.scheme to be `alipan`, instead, got {obj.scheme} for remote={remote}')
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


def download_from_local(remote: str, local: str) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (local filesystem).
        local (str): Local path (local filesystem).
    """
    local_tmp = local + '.tmp'
    if os.path.exists(local_tmp):
        os.remove(local_tmp)
    shutil.copy(remote, local_tmp)
    os.rename(local_tmp, local)


def download_file(remote: Optional[str], local: str, timeout: float):
    """Use the correct download handler to download the file.

    Args:
        remote (str, optional): Remote path (local filesystem).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for file to download before raising an exception.
    """
    if os.path.exists(local):
        return

    # fix paths for windows
    local = local.replace('\\', '/')
    if remote:
        remote = remote.replace('\\', '/')

    local_dir = os.path.dirname(local)
    os.makedirs(local_dir, exist_ok=True)

    if not remote:
        raise ValueError(
            'In the absence of local dataset, path to remote dataset must be provided')
    elif remote.startswith('s3://'):
        download_from_s3(remote, local, timeout)
    elif remote.startswith('sftp://'):
        download_from_sftp(remote, local)
    elif remote.startswith('gs://'):
        download_from_gcs(remote, local)
    elif remote.startswith('oci://'):
        download_from_oci(remote, local)
    elif remote.startswith('hf://'):
        download_from_hf(remote, local)
    elif remote.startswith('azure://'):
        download_from_azure(remote, local)
    elif remote.startswith('azure-dl://'):
        download_from_azure_datalake(remote, local)
    elif remote.startswith('dbfs:/Volumes'):
        download_from_databricks_unity_catalog(remote, local)
    elif remote.startswith('dbfs:/'):
        download_from_dbfs(remote, local)
    elif remote.startswith('alipan://'):
        download_from_alipan(remote, local)
    else:
        download_from_local(remote, local)


def wait_for_download(local: str, timeout: float = 60) -> None:
    """Wait for a download by another thread/process to complete.

    Args:
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for file to download before raising an exception.
            Defaults to ``60``.
    """
    t0 = time()
    while not os.path.exists(local):
        elapsed = time() - t0
        if timeout < elapsed:
            raise TimeoutError(
                f'Waited longer than {timeout}s for other worker to download {local}.')
        sleep(0.25)
