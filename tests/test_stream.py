# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shutil
import tempfile
from typing import Optional

import pytest
from _pytest.monkeypatch import MonkeyPatch

from streaming import Stream
from streaming.base.distributed import barrier


@pytest.mark.world_size(2)
def test_local_is_none_with_no_split(monkeypatch: MonkeyPatch) -> None:
    remote = 'remote_dir'
    remote_hash = hashlib.blake2s(remote.encode('utf-8'), digest_size=16).hexdigest()
    local = os.path.join(tempfile.gettempdir(), remote_hash)
    shutil.rmtree(local, ignore_errors=True)
    barrier()
    stream = Stream(remote=remote, local=None)
    assert local == stream.local


@pytest.mark.world_size(2)
def test_local_is_none_with_split(monkeypatch: MonkeyPatch) -> None:
    remote = 'remote_dir'
    remote_hash = hashlib.blake2s(remote.encode('utf-8'), digest_size=16).hexdigest()
    local = os.path.join(tempfile.gettempdir(), remote_hash + '_' + 'train')
    shutil.rmtree(local, ignore_errors=True)
    barrier()
    stream = Stream(remote=remote, local=None, split='train')
    assert local == stream.local


@pytest.mark.world_size(2)
@pytest.mark.parametrize('split', [None, 'train'])
def test_local_exists(split: Optional[str]) -> None:
    local = tempfile.mkdtemp()
    remote = 'remote_dir'
    stream = Stream(remote=remote, local=local, split=split)
    assert local == stream.local
