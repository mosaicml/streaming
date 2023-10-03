# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Any, Tuple
from unittest.mock import Mock, patch

import boto3
import pytest

from streaming.base.storage.upload import CloudUploader
from tests.conftest import MY_BUCKET

MY_PREFIX = 'train'


@pytest.fixture(scope='function')
def remote_local_dir() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""

    def _method(cloud_prefix: str = '') -> Tuple[str, str]:
        try:
            mock_local_dir = tempfile.TemporaryDirectory()
            mock_local = mock_local_dir.name
            mock_remote = os.path.join(cloud_prefix, MY_BUCKET, MY_PREFIX)
            return mock_remote, mock_local
        finally:
            mock_local_dir.cleanup()  # pyright: ignore

    return _method


class TestS3Client:

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_dir')
    def test_list_objects_from_s3(self, remote_local_dir: Any):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            file_name = tmp.name.split(os.sep)[-1]
            mock_remote_dir, _ = remote_local_dir(cloud_prefix='s3://')
            client = boto3.client('s3', region_name='us-east-1')
            client.put_object(Bucket=MY_BUCKET, Key=os.path.join(MY_PREFIX, file_name), Body='')

            cu = CloudUploader.get(mock_remote_dir, exist_ok=True, keep_local=True)
            objs = cu.list_objects(mock_remote_dir)
            assert isinstance(objs, list)

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_dir')
    def test_clienterror_exception(self, remote_local_dir: Any):
        mock_remote_dir, _ = remote_local_dir(cloud_prefix='s3://')
        cu = CloudUploader.get(mock_remote_dir, exist_ok=True, keep_local=True)
        objs = cu.list_objects()
        if objs:
            assert (len(objs) == 0)

    @pytest.mark.usefixtures('s3_client', 's3_test', 'remote_local_dir')
    def test_invalid_cloud_prefix(self, remote_local_dir: Any):
        with pytest.raises(ValueError):
            mock_remote_dir, _ = remote_local_dir(cloud_prefix='s9://')
            cu = CloudUploader.get(mock_remote_dir, exist_ok=True, keep_local=True)
            _ = cu.list_objects()


class TestGCSClient:

    @pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test', 'remote_local_dir')
    def test_invalid_cloud_prefix(self, remote_local_dir: Any):
        with pytest.raises(ValueError):
            mock_remote_dir, _ = remote_local_dir(cloud_prefix='gs9://')
            cu = CloudUploader.get(mock_remote_dir, exist_ok=True, keep_local=True)
            _ = cu.list_objects()

    def test_no_credentials_error(self, remote_local_dir: Any):
        """Ensure we raise a value error correctly if we have no credentials available."""
        with pytest.raises(ValueError):
            mock_remote_dir, _ = remote_local_dir(cloud_prefix='gs://')
            cu = CloudUploader.get(mock_remote_dir, exist_ok=True, keep_local=True)
            _ = cu.list_objects()


class TestListObjects:

    @patch('streaming.base.storage.LocalUploader.list_objects')
    @pytest.mark.usefixtures('remote_local_dir')
    def test_list_objects_from_local_gets_called(self, mocked_requests: Mock,
                                                 remote_local_dir: Any):
        mock_remote_dir, _ = remote_local_dir()
        cu = CloudUploader.get(mock_remote_dir, exist_ok=True, keep_local=True)
        cu.list_objects()
        mocked_requests.assert_called_once()
