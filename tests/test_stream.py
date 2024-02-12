# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shutil
import tempfile
from typing import Any, Optional

import pytest
from _pytest.monkeypatch import MonkeyPatch

from streaming import Stream, StreamingDataset
from streaming.distributed import barrier
from tests.common.utils import convert_to_mds


@pytest.mark.world_size(2)
def test_local_is_none_with_no_split() -> None:
    remote = 'remote_dir'
    data = os.path.abspath(remote).encode('utf-8')
    remote_hash = hashlib.blake2s(data, digest_size=16).hexdigest()
    local = os.path.join(tempfile.gettempdir(), 'streaming', 'local', remote_hash)
    shutil.rmtree(local, ignore_errors=True)
    barrier()
    stream = Stream(remote=remote, local=None)
    stream.apply_defaults(
        split=None,
        allow_schema_mismatch=False,
        allow_unsafe_types=False,
        allow_unchecked_resumption=False,
        download_retry=2,
        download_timeout='1m',
        download_max_size=None,
        validate_hash=None,
        keep_phases=None,
    )
    assert local == stream.local
    shutil.rmtree(local, ignore_errors=True)


@pytest.mark.world_size(2)
def test_local_is_none_with_split() -> None:
    remote = 'remote_dir'
    data = os.path.abspath(remote).encode('utf-8')
    remote_hash = hashlib.blake2s(data, digest_size=16).hexdigest()
    local = os.path.join(tempfile.gettempdir(), 'streaming', 'local', remote_hash)
    shutil.rmtree(local, ignore_errors=True)
    barrier()
    stream = Stream(remote=remote, local=None, split='train')
    stream.apply_defaults(
        allow_schema_mismatch=False,
        allow_unsafe_types=False,
        allow_unchecked_resumption=False,
        download_retry=2,
        download_timeout='1m',
        download_max_size=None,
        validate_hash=None,
        keep_phases=None,
    )
    assert local == stream.local
    shutil.rmtree(local, ignore_errors=True)


@pytest.mark.world_size(2)
@pytest.mark.parametrize('split', [None, 'train'])
def test_local_exists(split: Optional[str]) -> None:
    local = tempfile.mkdtemp()
    remote = 'remote_dir'
    stream = Stream(remote=remote, local=local, split=split)
    assert local == stream.local
    shutil.rmtree(local, ignore_errors=True)


def test_existing_local_raises_exception(monkeypatch: MonkeyPatch) -> None:
    local = tempfile.mkdtemp()
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: local)
    with pytest.raises(ValueError):
        _ = Stream()
    shutil.rmtree(local, ignore_errors=True)


@pytest.mark.usefixtures('local_remote_dir')
def test_missing_index_json_local(local_remote_dir: Any):
    num_samples = 117
    remote_dir, _ = local_remote_dir
    convert_to_mds(out_root=remote_dir, dataset_name='sequencedataset', num_samples=num_samples)
    if os.path.exists(os.path.join(remote_dir, 'index.json')):
        os.remove(os.path.join(remote_dir, 'index.json'))
    else:
        raise Exception(f"Missing {os.path.join(remote_dir, 'index.json')}")
    stream = Stream(remote=None, local=remote_dir)
    with pytest.raises(RuntimeError):
        _ = StreamingDataset(streams=[stream])
