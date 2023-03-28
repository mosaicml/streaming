# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
from typing import Any, List, Tuple
from unittest.mock import Mock, patch

import pytest

from streaming.base.storage.upload import CloudUploader, GCSUploader, LocalUploader, S3Uploader


class TestCloudUploader:

    @patch('streaming.base.storage.upload.S3Uploader.check_bucket_exists')
    @patch('streaming.base.storage.upload.GCSUploader.check_bucket_exists')
    @pytest.mark.parametrize(
        'mapping',
        [['s3://bucket/dir/file', S3Uploader], [None, 's3://bucket/dir/file', S3Uploader],
         ['gs://bucket/dir/file', GCSUploader], [None, 'gs://bucket/dir/file', GCSUploader],
         ['/tmp/dir/filepath', LocalUploader], ['./relative/dir/filepath', LocalUploader]])
    def test_instantiation_type(self, s3_mocked_requests: Mock, gcs_mocked_requests: Mock,
                                local_remote_dir: Tuple[str, str], mapping: List[Any]):
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
        with pytest.raises(ValueError) as exc_info:
            _ = CloudUploader.get(out=out)
        assert exc_info.match(r'Invalid `out` argument.*')

    @pytest.mark.parametrize('out', [('./dir1', 'gcs://bucket/dir/'), ('./dir1', None)])
    def test_invalid_out_parameter_type(self, out: Any):
        with pytest.raises(ValueError) as exc_info:
            _ = CloudUploader.get(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError) as exc_info:
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = CloudUploader.get(out=local)
        assert exc_info.match(r'Directory is not empty.*')

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
    def test_check_bucket_exists_exception(self, out: str):
        import botocore
        with pytest.raises(botocore.exceptions.ClientError):
            _ = CloudUploader.get(out=out)


class TestS3Uploader:

    @patch('streaming.base.storage.upload.S3Uploader.check_bucket_exists')
    @pytest.mark.parametrize('out', ['s3://bucket/dir', ('./dir1', 's3://bucket/dir/')])
    def test_instantiation(self, mocked_requests: Mock, out: Any):
        mocked_requests.side_effect = None
        _ = S3Uploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0])

    @pytest.mark.parametrize('out', ['ss4://bucket/dir'])
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError) as exc_info:
            _ = S3Uploader(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    @pytest.mark.parametrize('out', ['ss4://bucket/dir', ('./dir1', 'gcs://bucket/dir/')])
    def test_invalid_remote_list(self, out: Any):
        with pytest.raises(ValueError) as exc_info:
            _ = S3Uploader(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError) as exc_info:
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = S3Uploader(out=local)
        assert exc_info.match(r'Directory is not empty.*')

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

    @pytest.mark.parametrize('out', ['s3://bucket/dir'])
    def test_check_bucket_exists_exception(self, out: str):
        import botocore
        with pytest.raises(botocore.exceptions.ClientError):
            _ = S3Uploader(out=out)


class TestGCSUploader:

    @patch('streaming.base.storage.upload.GCSUploader.check_bucket_exists')
    @pytest.mark.parametrize('out', ['gs://bucket/dir', ('./dir1', 'gs://bucket/dir/')])
    def test_instantiation(self, mocked_requests: Mock, out: Any):
        mocked_requests.side_effect = None
        _ = GCSUploader(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0])

    @pytest.mark.parametrize('out', ['gcs://bucket/dir'])
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError) as exc_info:
            _ = GCSUploader(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    @pytest.mark.parametrize('out', ['gcs://bucket/dir', ('./dir1', 'ocix://bucket/dir/')])
    def test_invalid_remote_list(self, out: Any):
        with pytest.raises(ValueError) as exc_info:
            _ = GCSUploader(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError) as exc_info:
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = GCSUploader(out=local)
        assert exc_info.match(r'Directory is not empty.*')

    @pytest.mark.usefixtures('gcs_client', 'gcs_test')
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

    @pytest.mark.parametrize('out', ['gs://bucket/dir'])
    def test_check_bucket_exists_exception(self, out: str):
        import botocore
        with pytest.raises(botocore.exceptions.ClientError):
            _ = GCSUploader(out=out)


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
