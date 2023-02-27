# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
from typing import Any, List, Tuple

import pytest

from streaming.base.storage.upload import CloudWriter, GCSWriter, LocalWriter, S3Writer


class TestCloudWriter:

    @pytest.mark.parametrize(
        'mapping', [['s3://bucket/dir/file', S3Writer], [None, 's3://bucket/dir/file', S3Writer],
                    ['gs://bucket/dir/file', GCSWriter], [None, 'gs://bucket/dir/file', GCSWriter],
                    ['/tmp/dir/filepath', LocalWriter], ['./relative/dir/filepath', LocalWriter]])
    def test_instantiation_type(self, local_remote_dir: Tuple[str, str], mapping: List[Any]):
        local, _ = local_remote_dir
        if len(mapping) == 2:
            cw = CloudWriter.get(out=mapping[0])
        else:
            mapping[0] = local
            out_root = (mapping[0], mapping[1])
            cw = CloudWriter.get(out_root)
        assert isinstance(cw, mapping[-1])

    @pytest.mark.parametrize('out', [(), ('s3://bucket/dir',), ('./dir1', './dir2', './dir3')])
    def test_invalid_out_parameter_length(self, out: Any):
        with pytest.raises(ValueError) as exc_info:
            _ = CloudWriter.get(out=out)
        assert exc_info.match(r'Invalid `out` argument.*')

    @pytest.mark.parametrize('out', [('./dir1', 'gcs://bucket/dir/'), ('./dir1', None)])
    def test_invalid_out_parameter_type(self, out: Any):
        with pytest.raises(ValueError) as exc_info:
            _ = CloudWriter.get(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError) as exc_info:
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = CloudWriter.get(out=local)
        assert exc_info.match(r'Directory is not empty.*')

    def test_local_directory_is_created(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        _ = CloudWriter(out=local)
        assert os.path.exists(local)

    def test_delete_local_file(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        local_file_path = os.path.join(local, 'file.txt')
        cw = CloudWriter.get(out=local)
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        cw.clear_local(local_file_path)
        assert not os.path.exists(local_file_path)


class TestS3Writer():

    @pytest.mark.parametrize('out', ['s3://bucket/dir', ('./dir1', 's3://bucket/dir/')])
    def test_instantiation(self, out: Any):
        _ = S3Writer(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0])

    @pytest.mark.parametrize('out', ['ss4://bucket/dir'])
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError) as exc_info:
            _ = S3Writer(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    @pytest.mark.parametrize('out', ['ss4://bucket/dir', ('./dir1', 'gcs://bucket/dir/')])
    def test_invalid_remote_list(self, out: Any):
        with pytest.raises(ValueError) as exc_info:
            _ = S3Writer(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError) as exc_info:
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = S3Writer(out=local)
        assert exc_info.match(r'Directory is not empty.*')

    @pytest.mark.usefixtures('s3_client', 's3_test')
    def test_upload_file(self, local_remote_dir: Tuple[str, str]):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            filename = tmp.name.split(os.sep)[-1]
            local, _ = local_remote_dir
            remote = 's3://streaming-test-bucket/path'
            local_file_path = os.path.join(local, filename)
            s3w = S3Writer(out=(local, remote))
            with open(local_file_path, 'w') as _:
                pass
            s3w.upload_file(filename)
            assert not os.path.exists(local_file_path)


class TestGCSWriter():

    @pytest.mark.parametrize('out', ['gs://bucket/dir', ('./dir1', 'gs://bucket/dir/')])
    def test_instantiation(self, out: Any):
        _ = GCSWriter(out=out)
        if not isinstance(out, str):
            shutil.rmtree(out[0])

    @pytest.mark.parametrize('out', ['gcs://bucket/dir'])
    def test_invalid_remote_str(self, out: str):
        with pytest.raises(ValueError) as exc_info:
            _ = GCSWriter(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    @pytest.mark.parametrize('out', ['gcs://bucket/dir', ('./dir1', 'ocix://bucket/dir/')])
    def test_invalid_remote_list(self, out: Any):
        with pytest.raises(ValueError) as exc_info:
            _ = GCSWriter(out=out)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError) as exc_info:
            local, _ = local_remote_dir
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = GCSWriter(out=local)
        assert exc_info.match(r'Directory is not empty.*')

    @pytest.mark.usefixtures('gcs_client', 'gcs_test')
    def test_upload_file(self, local_remote_dir: Tuple[str, str]):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            filename = tmp.name.split(os.sep)[-1]
            local, _ = local_remote_dir
            remote = 'gs://streaming-test-bucket/path'
            local_file_path = os.path.join(local, filename)
            gcsw = GCSWriter(out=(local, remote))
            with open(local_file_path, 'w') as _:
                pass
            gcsw.upload_file(filename)
            assert not os.path.exists(local_file_path)


class TestLocalWriter:

    def test_upload_file(self, local_remote_dir: Tuple[str, str]):
        local, remote = local_remote_dir
        filename = 'file.txt'
        local_file_path = os.path.join(local, filename)
        remote_file_path = os.path.join(remote, filename)
        lw = LocalWriter(out=(local, remote))
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        assert not os.path.exists(remote_file_path)
        lw.upload_file(filename)
        assert os.path.exists(remote_file_path)

    def test_instantiation_remote_none(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        lc = LocalWriter(out=local)
        assert lc.local == local
        assert lc.remote is None

    def test_upload_file_remote_none(self, local_remote_dir: Tuple[str, str]):
        local, remote = local_remote_dir
        filename = 'file.txt'
        local_file_path = os.path.join(local, filename)
        remote_file_path = os.path.join(remote, filename)
        lc = LocalWriter(out=local)
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        lc.upload_file(filename)
        assert not os.path.exists(remote_file_path)
