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
TEST_FILE = 'file.txt'


@pytest.fixture(scope='function')
def remote_local_file() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""

    def _method(cloud_prefix: str = '', filename: str = TEST_FILE) -> tuple[str, str]:
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
            tmp = os.path.join(tmp_dir, TEST_FILE)
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='gs://', filename=TEST_FILE)
            client = boto3.client('s3',
                                  region_name='us-east-1',
                                  endpoint_url=GCS_URL,
                                  aws_access_key_id=os.environ['GCS_KEY'],
                                  aws_secret_access_key=os.environ['GCS_SECRET'])
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, TEST_FILE), Body='')
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

        def isinstance_impl(obj: Any, cls: Any):
            return obj.__class__ == cls.__class__

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = os.path.join(tmp_dir, 'file.txt')
            mock_isinstance.side_effect = isinstance_impl
            credentials_mock = Mock()
            mock_default.return_value = credentials_mock, None
            downloader = GCSDownloader()
            downloader._create_gcs_client()
            if downloader._gcs_client is not None:
                downloader._gcs_client._extra_headers = {}

            with patch('google.cloud.storage.Blob') as mock_blob_cls:
                mock_blob_instance = Mock()
                mock_blob_instance.download_to_filename.side_effect = lambda path: open(
                    path, 'w').write('mock data')
                mock_blob_cls.return_value = mock_blob_instance

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
            file_name = os.path.join(tmp_dir, TEST_FILE)
            mock_remote_filepath, _ = remote_local_file(cloud_prefix=cloud_prefix)
            with pytest.raises(Exception, match='Expected path prefix to be `dbfs:/Volumes`.*'):
                downloader = DatabricksUnityCatalogDownloader()
                downloader.download(mock_remote_filepath, file_name)

    @patch('databricks.sdk.WorkspaceClient', autospec=True)
    def test_databricks_error_file_not_found(self, workspace_client_mock: Mock,
                                             remote_local_file: Any):
        from databricks.sdk.errors.base import DatabricksError
        workspace_client_mock_instance = workspace_client_mock.return_value
        workspace_client_mock_instance.files = Mock()
        workspace_client_mock_instance.files.download = Mock()
        download_return_val = workspace_client_mock_instance.files.download.return_value
        download_return_val.contents = Mock()
        download_return_val.contents.__enter__ = Mock(
            side_effect=DatabricksError('Error', error_code='NOT_FOUND'))
        download_return_val.contents.__exit__ = Mock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, TEST_FILE)
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='dbfs:/Volumes')
            with pytest.raises(FileNotFoundError):
                downloader = DatabricksUnityCatalogDownloader()
                downloader.download(mock_remote_filepath, file_name)

    @patch('databricks.sdk.WorkspaceClient', autospec=True)
    def test_databricks_error(self, workspace_client_mock: Mock, remote_local_file: Any):
        from databricks.sdk.errors.base import DatabricksError
        workspace_client_mock_instance = workspace_client_mock.return_value
        workspace_client_mock_instance.files = Mock()
        workspace_client_mock_instance.files.download = Mock()
        download_return_val = workspace_client_mock_instance.files.download.return_value
        download_return_val.contents = Mock()
        download_return_val.contents.__enter__ = Mock(
            side_effect=DatabricksError('Error', error_code='REQUEST_LIMIT_EXCEEDED'))
        download_return_val.contents.__exit__ = Mock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, TEST_FILE)
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='dbfs:/Volumes')
            with pytest.raises(DatabricksError):
                downloader = DatabricksUnityCatalogDownloader()
                downloader.download(mock_remote_filepath, file_name)


class TestDatabricksFileSystem:

    def test_invalid_prefix_from_dbfs(self, remote_local_file: Any):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, TEST_FILE)
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='dbfsx:/')
            with pytest.raises(Exception, match='Expected remote path to start with.*'):
                downloader = DBFSDownloader()
                downloader.download(mock_remote_filepath, file_name)


def test_download_from_local():
    mock_remote_dir = tempfile.TemporaryDirectory()
    mock_local_dir = tempfile.TemporaryDirectory()
    mock_remote_file = os.path.join(mock_remote_dir.name, TEST_FILE)
    mock_local_file = os.path.join(mock_local_dir.name, TEST_FILE)
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
