# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Any, Tuple
from unittest.mock import Mock, patch

import boto3
import pytest

from streaming.base.storage.download import (list_objects, list_objects_from_gcs,
                                             list_objects_from_local, list_objects_from_s3)
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


class TestS3Client:

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_list_objects_from_s3(self, remote_local_file: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://', filename=file_name)
            client = boto3.client('s3', region_name='us-east-1')
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            objs = list_objects_from_s3(mock_remote_filepath)
            assert isinstance(objs, list)

    @pytest.mark.usefixtures('s3_client', 's3_test', 'r2_credentials', 'remote_local_file')
    def test_list_objects_from_s3_with_endpoint_URL(self, remote_local_file: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://', filename=file_name)
            client = boto3.client('s3', region_name='us-east-1')
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            _ = list_objects_from_s3(mock_remote_filepath)
            assert os.environ['S3_ENDPOINT_URL'] == R2_URL

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_clienterror_exception(self, remote_local_file: Any):
        mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://')
        objs = list_objects_from_s3(mock_remote_filepath)
        if objs:
            assert (len(objs) == 0)

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='s9://')
            _ = list_objects_from_s3(mock_remote_filepath)


class TestGCSClient:

    @pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test', 'remote_local_file')
    def test_list_objects_from_gcs(self, remote_local_file: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='gs://', filename=file_name)
            client = boto3.client('s3',
                                  region_name='us-east-1',
                                  endpoint_url=GCS_URL,
                                  aws_access_key_id=os.environ['GCS_KEY'],
                                  aws_secret_access_key=os.environ['GCS_SECRET'])
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')
            objs = list_objects_from_gcs(mock_remote_filepath)
            assert isinstance(objs, list)

    @patch('google.auth.default')
    @patch('google.cloud.storage.Client')
    @pytest.mark.usefixtures('gcs_service_account_credentials')
    @pytest.mark.parametrize('out', ['gs://bucket/dir'])
    def test_download_service_account(self, mock_client: Mock, mock_default: Mock, out: str):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as _:
            credentials_mock = Mock()
            mock_default.return_value = credentials_mock, None
            objs = list_objects_from_gcs(out)
            mock_client.assert_called_once_with(credentials=credentials_mock)
            assert isinstance(objs, list)

    @pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test', 'remote_local_file')
    def test_filenotfound_exception(self, remote_local_file: Any):
        mock_remote_filepath, _ = remote_local_file(cloud_prefix='gs://')
        _ = list_objects_from_gcs(mock_remote_filepath)

    @pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test', 'remote_local_file')
    def test_invalid_cloud_prefix(self, remote_local_file: Any):
        with pytest.raises(ValueError):
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://')
            _ = list_objects_from_gcs(mock_remote_filepath)

    def test_no_credentials_error(self, remote_local_file: Any):
        """Ensure we raise a value error correctly if we have no credentials available."""
        with pytest.raises(ValueError):
            mock_remote_filepath, _ = remote_local_file(cloud_prefix='gs://')
            _ = list_objects_from_gcs(mock_remote_filepath)


def test_list_objects_from_local():
    mock_local_dir = tempfile.TemporaryDirectory()
    file_name = 'file.txt'
    mock_local_file = os.path.join(mock_local_dir.name, file_name)
    # Creates a new empty file
    with open(mock_local_file, 'w') as _:
        pass
    with pytest.raises(NotADirectoryError):
        _ = list_objects_from_local(mock_local_file)


class TestListObjects:

    @patch('streaming.base.storage.download.list_objects_from_s3')
    @pytest.mark.usefixtures('remote_local_file')
    def test_list_objects_from_s3_gets_called(self, mocked_requests: Mock, remote_local_file: Any):
        mock_remote_filepath, _ = remote_local_file(cloud_prefix='s3://')
        list_objects(mock_remote_filepath)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath)

    @patch('streaming.base.storage.download.list_objects_from_gcs')
    @pytest.mark.usefixtures('remote_local_file')
    def test_list_objects_from_gcs_gets_called(self, mocked_requests: Mock,
                                               remote_local_file: Any):
        mock_remote_filepath, _ = remote_local_file(cloud_prefix='gs://')
        list_objects(mock_remote_filepath)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath)

    @patch('streaming.base.storage.download.list_objects_from_local')
    @pytest.mark.usefixtures('remote_local_file')
    def test_list_objects_from_local_gets_called(self, mocked_requests: Mock,
                                                 remote_local_file: Any):
        mock_remote_filepath, _ = remote_local_file()
        list_objects(mock_remote_filepath)
        mocked_requests.assert_called_once()
        mocked_requests.assert_called_once_with(mock_remote_filepath)

    @pytest.mark.usefixtures('remote_local_file')
    def test_list_objects_invalid_missing_remote(self):
        obj = list_objects(None)
        assert (obj == os.listdir())
