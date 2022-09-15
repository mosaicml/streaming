# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import urllib.parse
from time import sleep, time
from typing import Optional

__all__ = ['download_or_wait']

S3_NOT_FOUND_CODES = {'403', '404', 'NoSuchKey'}


def download_from_s3(remote: str, local: str, timeout: float) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (S3).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for shard to download before raising an exception.
    """
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(f"Expected obj.scheme to be 's3', got {obj.scheme} for remote={remote}")

    config = Config(read_timeout=timeout)
    s3 = boto3.client('s3', config=config)
    try:
        s3.download_file(obj.netloc, obj.path.lstrip('/'), local)
    except ClientError as e:
        if e.response['Error']['Code'] in S3_NOT_FOUND_CODES:
            raise FileNotFoundError(f'Object {remote} not found.') from e


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


def download(remote: Optional[str], local: str, timeout: float):
    """Use the correct download handler to download the file.

    Args:
        remote (str, optional): Remote path (local filesystem).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for file to download before raising an exception.
    """
    if os.path.exists(local):
        return

    local_dir = os.path.dirname(local)
    os.makedirs(local_dir, exist_ok=True)

    if not remote:
        raise ValueError(
            'In the absence of local dataset, path to remote dataset must be provided')
    elif remote.startswith('s3://'):
        download_from_s3(remote, local, timeout)
    elif remote.startswith('sftp://'):
        download_from_sftp(remote, local)
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


def download_or_wait(remote: Optional[str],
                     local: str,
                     wait: bool = False,
                     retry: int = 2,
                     timeout: float = 60) -> None:
    """Downloads a file from remote to local, or waits for it to be downloaded.

    Does not do any thread safety checks, so we assume the calling function is using ``wait``
    correctly.

    Args:
        remote (str, optional): Remote path (S3, SFTP, or local filesystem).
        local (str): Local path (local filesystem).
        wait (bool): If ``true``, then do not actively download the file, but instead wait (up to
            ``timeout`` seconds) for the file to arrive. Defaults to ``False``.
        retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        timeout (float): How long to wait for file to download before raising an exception.
            Defaults to ``60``.
    """
    errors = []
    for _ in range(1 + retry):
        try:
            if wait:
                wait_for_download(local, timeout)
            else:
                download(remote, local, timeout)
            break
        except FileNotFoundError:  # Bubble up file not found error.
            raise
        except Exception as e:  # Retry for all other causes of failure.
            errors.append(e)
    if retry < len(errors):
        raise RuntimeError(
            f'Failed to download {remote} -> {local}. Got errors:\n{errors}') from errors[-1]
