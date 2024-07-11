# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
from typing import Any, List, Tuple
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock, patch

import boto3
import pytest

from streaming.base.storage.upload import (AlipanUploader, AzureDataLakeUploader, AzureUploader,
                                           CloudUploader, DatabricksUnityCatalogUploader,
                                           DBFSUploader, GCSAuthentication, GCSUploader,
                                           HFUploader, LocalUploader, S3Uploader)
from tests.conftest import MY_BUCKET, R2_URL

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


class TestCloudUploader:

    @patch('streaming.base.storage.upload.S3Uploader.check_bucket_exists')
    @patch('streaming.base.storage.upload.GCSUploader.check_bucket_exists')
    @pytest.mark.parametrize(
        'mapping',
        [
            ['s3://bucket/dir/file', S3Uploader],
            [None, 's3://bucket/dir/file', S3Uploader],
            ['gs://bucket/dir/file', GCSUploader],
            [None, 'gs://bucket/dir/file', GCSUploader],
            ['/tmp/dir/filepath', LocalUploader],
            ['./relative/dir/filepath', LocalUploader],
        ],
    )
    @pytest.mark.usefixtures('gcs_hmac_credentials')
    def test_instantiation_type(
        self,
        s3_mocked_requests: Mock,
        gcs_mocked_requests: Mock,
        local_remote_dir: Tuple[str, str],
        mapping: List[Any],
    ):
        s3_mocked_requests.side_effect = None
        gcs_mocked_requests.side_effect = None
        local, _ = local_remote_dir
        if len(mapping) == 2:
            cw = CloudUploader.get(out=mapping[0])
        else:
            mapping[0] = local
            out_root = (mapping[0], mapping[1])
            cw = CloudUploader.get(out_root)
        assert isinstance(cw, mapping[-1])

    @pytest.mark.parametrize('out', [(), ('s3://bucket/dir',), ('./dir1', './dir2', './dir3')])
    def test_invalid_out_parameter_length(self, out: Any):
        with pytest.raises(ValueError, match=f'Invalid `out` argument.*'):
            _ = CloudUploader.get(out=out)

    @pytest.mark.parametrize('out', [('./dir1', 'gcs://bucket/dir/'), ('./dir1', None)])
    def test_invalid_out_parameter_type(self, out: Any):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = CloudUploader.get(out=out)

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = CloudUploader.get(out=local)

    def test_local_directory_is_created(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        _ = CloudUploader(out=local)
        assert os.path.exists(local)

    def test_delete_local_file(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        local_file_path = os.path.join(local, 'file.txt')
        cw = CloudUploader.get(out=local)
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        cw.clear_local(local_file_path)
        assert not os.path.exists(local_file_path)

    @pytest.mark.parametrize('out', ['s3://bucket/dir', 'gs://bucket/dir'])
    @pytest.mark.usefixtures('gcs_hmac_credentials')
    def test_check_bucket_exists_exception(self, out: str):
        import botocore
        with pytest.raises(botocore.exceptions.ClientError):
            _ = CloudUploader.get(out=out)

    @patch('streaming.base.storage.LocalUploader.list_objects')
    @pytest.mark.usefixtures('remote_local_dir')
    def test_list_objects_from_local_gets_called(self, mocked_requests: Mock,
                                                 remote_local_dir: Any):
        mock_remote_dir, _ = remote_local_dir()
        cu = CloudUploader.get(mock_remote_dir, exist_ok=True, keep_local=True)
        cu.list_objects()
        mocked_requests.assert_called_once()


class TestS3Uploader:

    @patch('streaming.base.storage.upload.S3Uploader.check_bucket_exists')
    @pytest.mark.parametrize('out', ['s3://bucket/dir', ('./dir1', 's3://bucket/dir/')])
    def test_instantiation(self, mocked_requests: Mock, out: Any):
        mocked_requests.side_effect = None
        _ = S3Uploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0], ignore_errors=True)

    @pytest.mark.parametrize('out', ['ss4://bucket/dir'])
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = S3Uploader(out=out)

    @pytest.mark.parametrize('out', ['ss4://bucket/dir', ('./dir1', 'gcs://bucket/dir/')])
    def test_invalid_remote_list(self, out: Any):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = S3Uploader(out=out)

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = S3Uploader(out=local)

    @pytest.mark.usefixtures('s3_client', 's3_test')
    def test_upload_file(self, local_remote_dir: Tuple[str, str]):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            filename = tmp.name.split(os.sep)[-1]
            local, _ = local_remote_dir
            remote = 's3://streaming-test-bucket/path'
            local_file_path = os.path.join(local, filename)
            s3w = S3Uploader(out=(local, remote))
            with open(local_file_path, 'w') as _:
                pass
            s3w.upload_file(filename)
            assert not os.path.exists(local_file_path)

    @pytest.mark.usefixtures('r2_client', 'r2_test')
    def test_upload_file_to_r2(self, local_remote_dir: Tuple[str, str]):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            filename = tmp.name.split(os.sep)[-1]
            local, _ = local_remote_dir
            remote = 's3://streaming-test-bucket/path'
            local_file_path = os.path.join(local, filename)
            s3w = S3Uploader(out=(local, remote))
            with open(local_file_path, 'w') as _:
                pass
            s3w.upload_file(filename)
            assert os.environ['S3_ENDPOINT_URL'] == R2_URL
            assert not os.path.exists(local_file_path)

    @pytest.mark.parametrize('out', ['s3://bucket/dir'])
    def test_check_bucket_exists_exception(self, out: str):
        import botocore

        with pytest.raises(botocore.exceptions.ClientError):
            _ = S3Uploader(out=out)

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

    @pytest.mark.usefixtures('s3_client', 's3_test')
    def test_extra_args(self, local_remote_dir: Tuple[str, str]):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            filename = tmp.name.split(os.sep)[-1]
            local, _ = local_remote_dir
            remote = 's3://streaming-test-bucket/path'
            local_file_path = os.path.join(local, filename)
            s3w = S3Uploader(out=(local, remote))
            s3w.s3.upload_file = MagicMock()
            with mock.patch.dict('os.environ'):
                # No ACL
                with open(local_file_path, 'w') as _:
                    pass
                os.environ.pop('S3_CANNED_ACL', None)
                s3w.upload_file(filename)
                s3w.s3.upload_file.assert_called_once()
                s3w.s3.upload_file.assert_called_with(ANY, ANY, ANY, ExtraArgs={}, Callback=ANY)

                # With `authenticated-read` ACL
                with open(local_file_path, 'w') as _:
                    pass
                os.environ['S3_CANNED_ACL'] = 'authenticated-read'
                s3w.upload_file(filename)
                s3w.s3.upload_file.assert_called_with(ANY,
                                                      ANY,
                                                      ANY,
                                                      ExtraArgs={'ACL': 'authenticated-read'},
                                                      Callback=ANY)

                # With `bucket-owner-full-control` ACL
                with open(local_file_path, 'w') as _:
                    pass
                os.environ['S3_CANNED_ACL'] = 'bucket-owner-full-control'
                s3w.upload_file(filename)
                s3w.s3.upload_file.assert_called_with(
                    ANY, ANY, ANY, ExtraArgs={'ACL': 'bucket-owner-full-control'}, Callback=ANY)


class TestGCSUploader:

    @patch('streaming.base.storage.upload.GCSUploader.check_bucket_exists')
    @pytest.mark.parametrize('out', ['gs://bucket/dir', ('./dir1', 'gs://bucket/dir/')])
    @pytest.mark.usefixtures('gcs_hmac_credentials')
    def test_instantiation(self, mocked_requests: Mock, out: Any):
        mocked_requests.side_effect = None
        _ = GCSUploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0], ignore_errors=True)

    @pytest.mark.parametrize('out', ['gcs://bucket/dir'])
    @pytest.mark.usefixtures('gcs_hmac_credentials')
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = GCSUploader(out=out)

    @pytest.mark.parametrize('out', ['gcs://bucket/dir', ('./dir1', 'ocix://bucket/dir/')])
    @pytest.mark.usefixtures('gcs_hmac_credentials')
    def test_invalid_remote_list(self, out: Any):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = GCSUploader(out=out)

    @pytest.mark.usefixtures('gcs_hmac_credentials')
    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = GCSUploader(out=local)

    @pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test')
    def test_upload_file(self, local_remote_dir: Tuple[str, str]):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            filename = tmp.name.split(os.sep)[-1]
            local, _ = local_remote_dir
            remote = 'gs://streaming-test-bucket/path'
            local_file_path = os.path.join(local, filename)
            gcsw = GCSUploader(out=(local, remote))
            with open(local_file_path, 'w') as _:
                pass
            gcsw.upload_file(filename)
            assert not os.path.exists(local_file_path)

    @pytest.mark.usefixtures('gcs_hmac_credentials')
    @pytest.mark.parametrize('out', ['gs://bucket/dir'])
    def test_check_bucket_exists_exception(self, out: str):
        import botocore

        with pytest.raises(botocore.exceptions.ClientError):
            _ = GCSUploader(out=out)

    @patch('streaming.base.storage.upload.GCSUploader.check_bucket_exists')
    @pytest.mark.usefixtures('gcs_hmac_credentials')
    @pytest.mark.parametrize('out', ['gs://bucket/dir'])
    def test_hmac_authentication(self, mocked_requests: Mock, out: str):
        uploader = GCSUploader(out=out)
        assert uploader.authentication == GCSAuthentication.HMAC

    @patch('google.auth.default')
    @patch('google.cloud.storage.Client')
    @pytest.mark.usefixtures('gcs_service_account_credentials')
    @pytest.mark.parametrize('out', ['gs://bucket/dir'])
    def test_service_account_authentication(self, mock_client: Mock, mock_default: Mock, out: str):
        mock_default.return_value = Mock(), None
        uploader = GCSUploader(out=out)
        assert uploader.authentication == GCSAuthentication.SERVICE_ACCOUNT

    @patch('streaming.base.storage.upload.GCSUploader.check_bucket_exists')
    @patch('google.auth.default')
    @patch('google.cloud.storage.Client')
    @pytest.mark.usefixtures('gcs_service_account_credentials', 'gcs_hmac_credentials')
    @pytest.mark.parametrize('out', ['gs://bucket/dir'])
    def test_service_account_and_hmac_authentication(self, mocked_requests: Mock,
                                                     mock_default: Mock, mock_client: Mock,
                                                     out: str):
        mock_default.return_value = Mock(), None
        uploader = GCSUploader(out=out)
        assert uploader.authentication == GCSAuthentication.HMAC

    @pytest.mark.parametrize('out', ['gs://bucket/dir'])
    def test_no_authentication(self, out: str):
        with pytest.raises(
                ValueError,
                match=
            (f'Either set the environment variables `GCS_KEY` and `GCS_SECRET` or use any of '
             f'the methods in https://cloud.google.com/docs/authentication/external/set-up-adc '
             f'to set up Application Default Credentials. See also '
             f'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/gcp.html.')):
            _ = GCSUploader(out=out)

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


class TestAzureUploader:

    @patch('streaming.base.storage.upload.AzureUploader.check_bucket_exists')
    @pytest.mark.usefixtures('azure_credentials')
    @pytest.mark.parametrize('out', ['azure://bucket/dir', ('./dir1', 'azure://bucket/dir/')])
    def test_instantiation(self, mocked_requests: Mock, out: Any):
        mocked_requests.side_effect = None
        _ = AzureUploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0], ignore_errors=True)

    @pytest.mark.parametrize('out', ['ss4://bucket/dir'])
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = AzureUploader(out=out)

    @pytest.mark.parametrize('out', ['ss4://bucket/dir', ('./dir1', 'gcs://bucket/dir/')])
    def test_invalid_remote_list(self, out: Any):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = AzureUploader(out=out)

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = AzureUploader(out=local)


class TestAzureDataLakeUploader:

    @patch('streaming.base.storage.upload.AzureDataLakeUploader.check_container_exists')
    @pytest.mark.usefixtures('azure_credentials')
    @pytest.mark.parametrize('out',
                             ['azure://container/dir', ('./dir1', 'azure://container/dir/')])
    def test_instantiation(self, mocked_requests: Mock, out: Any):
        mocked_requests.side_effect = None
        _ = AzureDataLakeUploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0], ignore_errors=True)

    @pytest.mark.parametrize('out', ['ss4://container/dir'])
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = AzureDataLakeUploader(out=out)

    @pytest.mark.parametrize('out', ['ss4://container/dir', ('./dir1', 'gcs://container/dir/')])
    def test_invalid_remote_list(self, out: Any):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = AzureDataLakeUploader(out=out)

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = AzureDataLakeUploader(out=local)


class TestHFUploader:

    @patch('streaming.base.storage.upload.HFUploader.check_dataset_exists')
    @pytest.mark.usefixtures('hf_credentials')
    @pytest.mark.parametrize('out', ['hf://datasets/org_name/repo_name/path'])
    def test_instantiation(self, mocked_requests: Mock, out: Any):
        mocked_requests.side_effect = None
        _ = HFUploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0], ignore_errors=True)

    @pytest.mark.parametrize('out', ['ss4://container/dir'])
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = HFUploader(out=out)

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = HFUploader(out=local)


class TestDatabricksUnityCatalogUploader:

    @patch('streaming.base.storage.upload.DatabricksUploader._create_workspace_client')
    @pytest.mark.parametrize(
        'out', ['dbfs:/Volumes/container/dir', ('./dir1', 'dbfs:/Volumes/container/dir/')])
    def test_instantiation(self, mock_create_client: Mock, out: Any):
        mock_create_client.side_effect = None
        _ = DatabricksUnityCatalogUploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0], ignore_errors=True)

    @patch('streaming.base.storage.upload.DatabricksUploader._create_workspace_client')
    @pytest.mark.parametrize('out', ['ss4://bucket/dir', ('./dir1', 'gcs://bucket/dir/')])
    def test_invalid_remote_list(self, mock_create_client: Mock, out: Any):
        mock_create_client.side_effect = None
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = DatabricksUnityCatalogUploader(out=out)

    @patch('streaming.base.storage.upload.DatabricksUploader._create_workspace_client')
    def test_local_directory_is_empty(self, mock_create_client: Mock,
                                      local_remote_dir: Tuple[str, str]):
        mock_create_client.side_effect = None
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = DatabricksUnityCatalogUploader(out=local)


class TestDBFSUploader:

    @patch('streaming.base.storage.upload.DatabricksUploader._create_workspace_client')
    @pytest.mark.parametrize('out', ['dbfs:/container/dir', ('./dir1', 'dbfs:/container/dir/')])
    def test_instantiation(self, mock_create_client: Mock, out: Any):
        mock_create_client.side_effect = None
        _ = DBFSUploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0], ignore_errors=True)

    @patch('streaming.base.storage.upload.DatabricksUploader._create_workspace_client')
    @pytest.mark.parametrize('out', ['ss4://bucket/dir', ('./dir1', 'gcs://bucket/dir/')])
    def test_invalid_remote_list(self, mock_create_client: Mock, out: Any):
        mock_create_client.side_effect = None
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = DBFSUploader(out=out)

    @patch('streaming.base.storage.upload.DatabricksUploader._create_workspace_client')
    def test_local_directory_is_empty(self, mock_create_client: Mock,
                                      local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            mock_create_client.side_effect = None
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = DBFSUploader(out=local)


class TestAlipanUploader:

    @patch('streaming.base.storage.upload.AlipanUploader.check_token')
    @pytest.mark.usefixtures('alipan_credentials')
    @pytest.mark.parametrize('out',
                             ['alipan:///container/dir', ('./dir1', 'alipan:///container/dir/')])
    def test_instantiation(self, mock_create_client: Mock, out: Any):
        mock_create_client.side_effect = None
        _ = AlipanUploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0], ignore_errors=True)

    @patch('streaming.base.storage.upload.AlipanUploader.check_token')
    @pytest.mark.usefixtures('alipan_credentials')
    @pytest.mark.parametrize('out', ['alipann://bucket/dir', ('./dir1', 'gcs://bucket/dir/')])
    def test_invalid_remote_list(self, mock_create_client: Mock, out: Any):
        mock_create_client.side_effect = None
        with pytest.raises(ValueError, match=f'Invalid Cloud provider prefix.*'):
            _ = AlipanUploader(out=out)

    @patch('streaming.base.storage.upload.AlipanUploader.check_token')
    @pytest.mark.usefixtures('alipan_credentials')
    def test_local_directory_is_empty(self, mock_create_client: Mock,
                                      local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError, match=f'Directory is not empty.*'):
            mock_create_client.side_effect = None
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = AlipanUploader(out=local)


class TestLocalUploader:

    def test_upload_file(self, local_remote_dir: Tuple[str, str]):
        local, remote = local_remote_dir
        filename = 'file.txt'
        local_file_path = os.path.join(local, filename)
        remote_file_path = os.path.join(remote, filename)
        lw = LocalUploader(out=(local, remote))
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        assert not os.path.exists(remote_file_path)
        lw.upload_file(filename)
        assert os.path.exists(remote_file_path)

    def test_instantiation_remote_none(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        lc = LocalUploader(out=local)
        assert lc.local == local
        assert lc.remote is None

    def test_upload_file_remote_none(self, local_remote_dir: Tuple[str, str]):
        local, remote = local_remote_dir
        filename = 'file.txt'
        local_file_path = os.path.join(local, filename)
        remote_file_path = os.path.join(remote, filename)
        lc = LocalUploader(out=local)
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        lc.upload_file(filename)
        assert not os.path.exists(remote_file_path)

    def test_upload_file_from_local_to_remote(self, local_remote_dir: Tuple[str, str]):
        local, remote = local_remote_dir
        filename = 'file.txt'
        local_file_path = os.path.join(local, filename)
        remote_file_path = os.path.join(remote, filename)
        lc = LocalUploader(out=(local, remote))
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        lc.upload_file(filename)
        assert os.path.exists(remote_file_path)

    def test_upload_file_exception(self, local_remote_dir: Tuple[str, str]):
        local, remote = local_remote_dir
        filename = 'file.txt'
        lc = LocalUploader(out=(local, remote))
        with pytest.raises(FileNotFoundError, match=f'No such file or directory:.*'):
            lc.upload_file(filename)

    def test_list_objects_no_prefix(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        lc = LocalUploader(out=local)

        # Generate some local files for testing
        test_dir = os.path.join(local, 'test_dir')
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, 'file2.txt'),
                  'w') as f2, open(os.path.join(test_dir, 'file1.txt'), 'w') as f1:
            f1.write('Content of file1')
            f2.write('Content of file2')

        result = lc.list_objects()
        expected = [os.path.join(test_dir, 'file1.txt'), os.path.join(test_dir, 'file2.txt')]
        assert (result == expected)

    def test_list_objects_with_prefix(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        lc = LocalUploader(out=local)

        # Generate some local files for testing
        test_dir = os.path.join(local, 'test_dir')
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, 'prefix_file2.txt'),
                  'w') as f2, open(os.path.join(test_dir, 'prefix_file1.txt'), 'w') as f1:
            f1.write('Content of file1')
            f2.write('Content of file2')

        result = lc.list_objects(prefix='test_dir/prefix')
        expected = [
            os.path.join(test_dir, 'prefix_file1.txt'),
            os.path.join(test_dir, 'prefix_file2.txt')
        ]
        assert (result == expected)
