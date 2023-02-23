# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Any, Tuple

import pytest

from streaming.base.storage.upload import CloudWriter, GCSWriter, LocalWriter, S3Writer


class TestCloudWriter:

    def test_instantiation_type(self, local_remote_dir: Tuple[str, str]):
        mapping = {
            's3://bucket/dir/file': S3Writer,
            'gs://bucket/dir/file': GCSWriter,
            '/tmp/dir/filepath': LocalWriter,
            './relative/dir/filepath': LocalWriter,
            None: LocalWriter
        }
        local, _ = local_remote_dir
        for remote, class_type in mapping.items():
            cw = CloudWriter(local=local, remote=remote)
            assert isinstance(cw, class_type)

    @pytest.mark.parametrize('remote', [None, ''])
    @pytest.mark.parametrize('local', [None, ''])
    def test_empty_local_and_remote(self, local: Any, remote: Any):
        with pytest.raises(ValueError) as exc_info:
            _ = CloudWriter(local=local, remote=remote)
        assert exc_info.match(r'You must provide local and/or remote path.*')

    @pytest.mark.parametrize('remote', ['s33://bucket/path', 'rx://folder/'])
    def test_invalid_remote_path(self, remote: str):
        with pytest.raises(KeyError) as exc_info:
            _ = CloudWriter(local=None, remote=remote)
        assert exc_info.match(r'Invalid Cloud provider prefix.*')

    def test_local_directory_is_empty(self, local_remote_dir: Tuple[str, str]):
        with pytest.raises(FileExistsError) as exc_info:
            local, _ = local_remote_dir
            print(f'{local=}')
            os.makedirs(local, exist_ok=True)
            local_file_path = os.path.join(local, 'file.txt')
            # Creating an empty file at specified location
            with open(local_file_path, 'w') as _:
                pass
            _ = CloudWriter(local=local, remote=None)
        assert exc_info.match(r'Directory is not empty.*')

    def test_local_directory_is_created(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        _ = CloudWriter(local=local, remote=None)
        assert os.path.exists(local)

    def test_delete_local_file(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        local_file_path = os.path.join(local, 'file.txt')
        cw = CloudWriter(local=local, remote=None)
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        cw.clear_local(local_file_path)
        assert not os.path.exists(local_file_path)


class TestS3Writer():

    def test_empty_local_and_remote(self):
        with pytest.raises(ValueError):
            _ = S3Writer(local=None, remote=None)

    def test_empty_local(self):
        remote = 's3://bucket/path'
        s3w = S3Writer(local=None, remote=remote)
        assert s3w.local is not None
        assert s3w.remote == remote

    def test_empty_remote(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        remote = 's3://bucket/path'
        s3w = S3Writer(local=local, remote=remote)
        assert s3w.local == local
        assert s3w.remote is remote

    @pytest.mark.usefixtures('s3_client', 's3_test')
    def test_upload_file(self, local_remote_dir: Tuple[str, str]):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            filename = tmp.name.split(os.sep)[-1]
            local, _ = local_remote_dir
            remote = 's3://streaming-test-bucket/path'
            local_file_path = os.path.join(local, filename)
            s3w = S3Writer(local=local, remote=remote)
            with open(local_file_path, 'w') as _:
                pass
            s3w.upload_file(filename)
            assert not os.path.exists(local_file_path)


class TestGCSWriter():

    def test_empty_local_and_remote(self):
        with pytest.raises(ValueError):
            _ = GCSWriter(local=None, remote=None)

    def test_empty_local(self):
        remote = 'gs://bucket/path'
        gcsw = GCSWriter(local=None, remote=remote)
        assert gcsw.local is not None
        assert gcsw.remote == remote

    def test_empty_remote(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        remote = 'gs://bucket/path'
        gcsw = GCSWriter(local=local, remote=remote)
        assert gcsw.local == local
        assert gcsw.remote is remote

    @pytest.mark.usefixtures('gcs_client', 'gcs_test')
    def test_upload_file(self, local_remote_dir: Tuple[str, str]):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as tmp:
            filename = tmp.name.split(os.sep)[-1]
            local, _ = local_remote_dir
            remote = 'gs://streaming-test-bucket/path'
            local_file_path = os.path.join(local, filename)
            gcsw = GCSWriter(local=local, remote=remote)
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
        lw = LocalWriter(local=local, remote=remote)
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        assert not os.path.exists(remote_file_path)
        lw.upload_file(filename)
        assert os.path.exists(remote_file_path)

    def test_instantiation_remote_none(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        lc = LocalWriter(local=local, remote=None)
        assert lc.local == local
        assert lc.remote is None

    def test_upload_file_remote_none(self, local_remote_dir: Tuple[str, str]):
        local, remote = local_remote_dir
        filename = 'file.txt'
        local_file_path = os.path.join(local, filename)
        remote_file_path = os.path.join(remote, filename)
        lc = LocalWriter(local=local, remote=None)
        # Creating an empty file at specified location
        with open(local_file_path, 'w') as _:
            pass
        lc.upload_file(filename)
        assert not os.path.exists(remote_file_path)
