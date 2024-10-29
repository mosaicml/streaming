# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Any
from unittest.mock import Mock, patch

import boto3
import pytest
from botocore.exceptions import ClientError
from google.cloud.storage import Client

from streaming.base.storage.download import (AlipanDownloader, AzureDataLakeDownloader,
                                             AzureDownloader, CloudDownloader,
                                             DatabricksUnityCatalogDownloader, DBFSDownloader,
                                             GCSDownloader, HFDownloader, LocalDownloader,
                                             OCIDownloader, S3Downloader, SFTPDownloader)
from tests.conftest import GCS_URL, MY_BUCKET, R2_URL

MY_PREFIX = 'train'


@pytest.fixture(scope='function')
def remote_local_file() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""

    def _method(cloud_prefix: str = '', filename: str = 'file.txt') -> tuple[str, str]:
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
            downloader = AzureDownloader()
            downloader.download(mock_remote_filepath, mock_local_filepath)


class TestAzureDataLakeClient:

    @pytest.mark.usefixtures('remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(
                cloud_prefix='aadatalake://')
            downloader = AzureDataLakeDownloader()
            downloader.download(mock_remote_filepath, mock_local_filepath)


class TestHFClient:

    @pytest.mark.usefixtures('remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='hf://')
            downloader = HFDownloader()
            downloader.download(mock_remote_filepath, mock_local_filepath)


class TestS3Client:

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_download_from_s3(self, remote_local_file: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://', filename=file_name)
            client = boto3.client('s3', region_name='us-east-1')
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            downloader = S3Downloader()
            downloader.download(mock_remote_filepath, tmp.name, 60.0)
            assert os.path.isfile(tmp.name)

    @pytest.mark.usefixtures('s3_client', 's3_test', 'r2_credentials', 'remote_local_file')
    def test_download_from_s3_with_endpoint_URL(self, remote_local_file: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://', filename=file_name)
            client = boto3.client('s3', region_name='us-east-1')
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            downloader = S3Downloader()
            downloader.download(mock_remote_filepath, tmp.name, 60.0)
            assert os.path.isfile(tmp.name)
            assert os.environ['S3_ENDPOINT_URL'] == R2_URL

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_clienterror_exception(self, remote_local_file: Any):
        with pytest.raises(ClientError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='s3://')
            downloader = S3Downloader()
            downloader.download(mock_remote_filepath, mock_local_filepath, 60)

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='s9://')
            downloader = S3Downloader()
            downloader.download(mock_remote_filepath, mock_local_filepath, 60)


class TestGCSClient:

    @pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test', 'remote_local_file')
    def test_download_from_gcs(self, remote_local_file: Any):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = 'file.txt'
            tmp = os.path.join(tmp_dir, file_name)
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='gs://', filename=file_name)
            client = boto3.client('s3',
                                  region_name='us-east-1',
                                  endpoint_url=GCS_URL,
                                  aws_access_key_id=os.environ['GCS_KEY'],
                                  aws_secret_access_key=os.environ['GCS_SECRET'])
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            downloader = GCSDownloader()
            downloader.download(mock_remote_filepath, tmp)
            assert os.path.isfile(tmp)

    @patch('google.auth.default')
    @patch('google.cloud.storage.Client', spec=Client)
    @patch('streaming.base.storage.download.isinstance')
    @pytest.mark.usefixtures('gcs_service_account_credentials')
    @pytest.mark.parametrize('out', ['gs://bucket/dir'])
    def test_download_service_account(self, mock_isinstance: Mock, mock_client: Mock,
                                      mock_default: Mock, out: str):

        # Because of how mock works on the types... have to patch isinstance
        def isinstance_impl(obj: Any, cls: Any):
            return obj.__class__ == cls.__class__

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = os.path.join(tmp_dir, 'file.txt')
            mock_isinstance.side_effect = isinstance_impl
            credentials_mock = Mock()
            mock_default.return_value = credentials_mock, None
            downloader = GCSDownloader()
            downloader.download(out, tmp)
            mock_client.assert_called_once_with(credentials=credentials_mock)
            assert os.path.isfile(tmp)

    @pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test', 'remote_local_file')
    def test_filenotfound_exception(self, remote_local_file: Any):
        with pytest.raises(FileNotFoundError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='gs://')
            downloader = GCSDownloader()
            downloader.download(mock_remote_filepath, mock_local_filepath)

    @pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test', 'remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='s3://')
            downloader = GCSDownloader()
            downloader.download(mock_remote_filepath, mock_local_filepath)

    def test_no_credentials_error(self, remote_local_file: Any):
        """Ensure we raise a value error correctly if we have no credentials available."""
        with pytest.raises(ValueError):
            mock_remote_filepath, mock_local_filepath = remote_local_file(cloud_prefix='gs://')
            downloader = GCSDownloader()
            downloader.download(mock_remote_filepath, mock_local_filepath)


class TestDatabricksUnityCatalog:

    @pytest.mark.parametrize('cloud_prefix', ['dbfs:/Volumess', 'dbfs:/Content'])
    def test_invalid_prefix_from_db_uc(self, remote_local_file: Any, cloud_prefix: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, 'file.txt')
            mock_remote_filepath, _ = remote_local_file(cloud_prefix=cloud_prefix,
                                                        filename=file_name)
            with pytest.raises(Exception, match='Expected path prefix to be `dbfs:/Volumes`.*'):
                downloader = DatabricksUnityCatalogDownloader()
                downloader.download(mock_remote_filepath, file_name)


class TestDatabricksFileSystem:

    def test_invalid_prefix_from_dbfs(self, remote_local_file: Any):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, 'file.txt')
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='dbfsx:/', filename=file_name)
            with pytest.raises(Exception, match='Expected remote path to start with.*'):
                downloader = DBFSDownloader()
                downloader.download(mock_remote_filepath, file_name)


def test_download_from_local():
    mock_remote_dir = tempfile.TemporaryDirectory()
    mock_local_dir = tempfile.TemporaryDirectory()
    file_name = 'file.txt'
    mock_remote_file = os.path.join(mock_remote_dir.name, file_name)
    mock_local_file = os.path.join(mock_local_dir.name, file_name)
    # Creates a new empty file
    with open(mock_remote_file, 'w') as _:
        pass

    downloader = LocalDownloader()
    downloader.download(mock_remote_file, mock_local_file)
    assert os.path.isfile(mock_local_file)


class TestDownload:

    @pytest.mark.parametrize('cloud_prefix,downloader_type', [
        ('s3://', S3Downloader),
        ('gs://', GCSDownloader),
        ('hf://', HFDownloader),
        ('oci://', OCIDownloader),
        ('azure://', AzureDownloader),
        ('azure-dl://', AzureDataLakeDownloader),
        ('sftp://', SFTPDownloader),
        ('dbfs:/Volumes', DatabricksUnityCatalogDownloader),
        ('dbfs:/', DBFSDownloader),
        ('alipan://', AlipanDownloader),
        ('', LocalDownloader),
    ])
    @pytest.mark.usefixtures('remote_local_file')
    def test_getting_downloader(self, remote_local_file: Any, cloud_prefix: str,
                                downloader_type: type[CloudDownloader]):
        mock_remote_filepath, _ = remote_local_file(cloud_prefix=cloud_prefix)
        downloader = CloudDownloader.get(mock_remote_filepath)
        assert isinstance(downloader, downloader_type)

    def test_download_no_remote_dir(self):
        downloader = CloudDownloader.get()
        assert isinstance(downloader, CloudDownloader)

    def test_downloader_without_remote_and_local(self):
        with pytest.raises(ValueError):
            downloader = CloudDownloader.get('/path/to/local/file')
            downloader.download('', '/path/to/local/file')
