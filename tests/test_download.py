# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Any, Tuple
from unittest.mock import Mock, patch

import boto3
import pytest
from botocore.exceptions import ClientError

from streaming.base.storage.download import (download_file, download_from_azure,
                                             download_from_azuredl, download_from_gcs,
                                             download_from_local, download_from_s3,
                                             download_or_wait)
from tests.conftest import GCS_URL, MY_BUCKET, R2_URL

MY_PREFIX = 'train'


@pytest.fixture(scope='function')
def remote_local_file() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""

    def _method(cloud_prefix: str = '', filename: str = 'file.txt') -> Tuple[str, str]:
        try:
            mock_local_dir = tempfile.TemporaryDirectory()
            mock_local_filepath = os.path.join(mock_local_dir.name, filename)
            mock_remote_filepath = os.path.join(cloud_prefix, MY_BUCKET, MY_PREFIX, filename)
            return mock_remote_filepath, mock_local_filepath
        finally:
            mock_local_dir.cleanup()  # pyright: ignore

    return _method


class TestAzureClient:

    @pytest.mark.usefixtures('remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(
                cloud_prefix='aaazure://')
            download_from_azure(mock_remote_filepath, mock_local_filepath)
            download_from_azuredl(mock_remote_filepath, mock_local_filepath)


class TestS3Client:

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_download_from_s3(self, remote_local_file: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://', filename=file_name)
            client = boto3.client('s3', region_name='us-east-1')
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            download_from_s3(mock_remote_filepath, tmp.name, 60)
            assert os.path.isfile(tmp.name)

    @pytest.mark.usefixtures('s3_client', 's3_test', 'r2_credentials', 'remote_local_file')
    def test_download_from_s3_with_endpoint_URL(self, remote_local_file: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://', filename=file_name)
            client = boto3.client('s3', region_name='us-east-1')
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            download_from_s3(mock_remote_filepath, tmp.name, 60)
            assert os.path.isfile(tmp.name)
            assert os.environ['S3_ENDPOINT_URL'] == R2_URL

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_clienterror_exception(self, remote_local_file: Any):
        with pytest.raises(ClientError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='s3://')
            download_from_s3(mock_remote_filepath, mock_local_filepath, 60)

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='s9://')
            download_from_s3(mock_remote_filepath, mock_local_filepath, 60)


class TestGCSClient:

    @pytest.mark.usefixtures('gcs_client', 'gcs_test', 'remote_local_file')
    def test_download_from_gcs(self, remote_local_file: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='gs://', filename=file_name)
            client = boto3.client('s3',
                                  region_name='us-east-1',
                                  endpoint_url=GCS_URL,
                                  aws_access_key_id=os.environ['GCS_KEY'],
                                  aws_secret_access_key=os.environ['GCS_SECRET'])
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            download_from_gcs(mock_remote_filepath, tmp.name)
            assert os.path.isfile(tmp.name)

    @pytest.mark.usefixtures('gcs_client', 'gcs_test', 'remote_local_file')
    def test_filenotfound_exception(self, remote_local_file: Any):
        with pytest.raises(FileNotFoundError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='gs://')
            download_from_gcs(mock_remote_filepath, mock_local_filepath)

    @pytest.mark.usefixtures('gcs_client', 'gcs_test', 'remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='s3://')
            download_from_gcs(mock_remote_filepath, mock_local_filepath)


def test_download_from_local():
    mock_remote_dir = tempfile.TemporaryDirectory()
    mock_local_dir = tempfile.TemporaryDirectory()
    file_name = 'file.txt'
    mock_remote_file = os.path.join(mock_remote_dir.name, file_name)
    mock_local_file = os.path.join(mock_local_dir.name, file_name)
    # Creates a new empty file
    with open(mock_remote_file, 'w') as _:
        pass

    download_from_local(mock_remote_file, mock_local_file)
    assert os.path.isfile(mock_local_file)


class TestDownload:

    @patch('streaming.base.storage.download.download_from_s3')
    @pytest.mark.usefixtures('remote_local_file')
    def test_download_from_s3_gets_called(self, mocked_requests: Mock, remote_local_file: Any):
        mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='s3://')
        download_file(mock_remote_filepath, mock_local_filepath, 60)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath, mock_local_filepath, 60)

    @patch('streaming.base.storage.download.download_from_gcs')
    @pytest.mark.usefixtures('remote_local_file')
    def test_download_from_gcs_gets_called(self, mocked_requests: Mock, remote_local_file: Any):
        mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='gs://')
        download_file(mock_remote_filepath, mock_local_filepath, 60)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath, mock_local_filepath)

    @patch('streaming.base.storage.download.download_from_azure')
    @pytest.mark.usefixtures('remote_local_file')
    def test_download_from_azure_gets_called(self, mocked_requests: Mock, remote_local_file: Any):
        mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='azure://')
        download_file(mock_remote_filepath, mock_local_filepath, 60)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath, mock_local_filepath)

    @patch('streaming.base.storage.download.download_from_azuredl')
    @pytest.mark.usefixtures('remote_local_file')
    def test_download_from_azuredl_gets_called(self, mocked_requests: Mock,
                                               remote_local_file: Any):
        mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='azure-dl://')
        download_file(mock_remote_filepath, mock_local_filepath, 60)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath, mock_local_filepath)

    @patch('streaming.base.storage.download.download_from_sftp')
    @pytest.mark.usefixtures('remote_local_file')
    def test_download_from_sftp_gets_called(self, mocked_requests: Mock, remote_local_file: Any):
        mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='sftp://')
        download_file(mock_remote_filepath, mock_local_filepath, 60)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath, mock_local_filepath)

    @patch('streaming.base.storage.download.download_from_local')
    @pytest.mark.usefixtures('remote_local_file')
    def test_download_from_local_gets_called(self, mocked_requests: Mock, remote_local_file: Any):
        mock_remote_filepath, mock_local_filepath = remote_local_file()
        download_file(mock_remote_filepath, mock_local_filepath, 60)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath, mock_local_filepath)

    @pytest.mark.usefixtures('remote_local_file')
    def test_download_invalid_missing_remote(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            _, mock_local_filepath = remote_local_file()
            download_file(None, mock_local_filepath, 60)


class TestDownloadOrWait:

    @patch('streaming.base.storage.download.wait_for_download')
    @pytest.mark.usefixtures('remote_local_file')
    def test_wait_for_download(self, mocked_requests: Mock, remote_local_file: Any):
        mock_remote_filepath, mock_local_filepath = remote_local_file()
        download_or_wait(mock_remote_filepath, mock_local_filepath, True, 2, 60)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_local_filepath, 60)

    @patch('streaming.base.storage.download.download_file')
    @pytest.mark.usefixtures('remote_local_file')
    def test_download(self, mocked_requests: Mock, remote_local_file: Any):
        mock_remote_filepath, mock_local_filepath = remote_local_file()
        download_or_wait(mock_remote_filepath, mock_local_filepath, False, 2, 60)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath, mock_local_filepath, 60)

    @patch('streaming.base.storage.download.wait_for_download')
    @pytest.mark.usefixtures('remote_local_file')
    def test_failed_download_exception(self, mocked_requests: Mock, remote_local_file: Any):
        with pytest.raises(RuntimeError):
            mocked_requests.side_effect = Exception
            mock_remote_filepath, mock_local_filepath = remote_local_file()
            download_or_wait(mock_remote_filepath, mock_local_filepath, True, 2, 60)

    @patch('streaming.base.storage.download.wait_for_download')
    @pytest.mark.usefixtures('remote_local_file')
    def test_failed_download_filenotfound_exception(self, mocked_requests: Mock,
                                                    remote_local_file: Any):
        with pytest.raises(FileNotFoundError):
            mocked_requests.side_effect = FileNotFoundError
            mock_remote_filepath, mock_local_filepath = remote_local_file()
            download_or_wait(mock_remote_filepath, mock_local_filepath, True, 2, 60)
