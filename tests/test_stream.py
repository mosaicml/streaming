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
from streaming.base.distributed import barrier
from tests.common.utils import convert_to_mds


@pytest.mark.world_size(2)
def test_local_is_none_with_no_split() -> None:
    remote = 'remote_dir'
    remote_hash = hashlib.blake2s(remote.encode('utf-8'), digest_size=16).hexdigest()
    local = os.path.join(tempfile.gettempdir(), remote_hash) + '/'
    shutil.rmtree(local, ignore_errors=True)
    barrier()
    stream = Stream(remote=remote, local=None)
    assert local == stream.local
    shutil.rmtree(local, ignore_errors=True)


@pytest.mark.world_size(2)
def test_local_is_none_with_split() -> None:
    remote = 'remote_dir'
    remote_hash = hashlib.blake2s(remote.encode('utf-8'), digest_size=16).hexdigest()
    local = os.path.join(tempfile.gettempdir(), remote_hash, 'train')
    shutil.rmtree(local, ignore_errors=True)
    barrier()
    stream = Stream(remote=remote, local=None, split='train')
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
    with pytest.raises(FileExistsError, match=f'Could not create a temporary local directory.*'):
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
    with pytest.raises(RuntimeError, match='No `remote` provided, but local file.*'):
        _ = StreamingDataset(streams=[stream], batch_size=1)
