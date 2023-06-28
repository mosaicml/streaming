# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import tempfile
from typing import Optional

import pytest
from _pytest.monkeypatch import MonkeyPatch

from streaming import Stream


@pytest.mark.world_size(2)
def test_local_is_none_with_no_split(monkeypatch: MonkeyPatch) -> None:
    local = 'local_dir'
    remote = 'remote_dir'
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: local)
    remote_hash = hashlib.md5(remote.encode('utf-8')).hexdigest()
    stream = Stream(remote=remote, local=None)
    assert os.path.join(local, remote_hash) == stream.local


@pytest.mark.world_size(2)
def test_local_is_none_with_split(monkeypatch: MonkeyPatch) -> None:
    local = 'local_dir'
    remote = 'remote_dir'
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: local)
    remote_hash = hashlib.md5(remote.encode('utf-8')).hexdigest()
    stream = Stream(remote=remote, local=None, split='train')
    assert os.path.join(local, remote_hash + '_' + 'train') == stream.local


@pytest.mark.world_size(2)
@pytest.mark.parametrize('split', [None, 'train'])
def test_local_exists(split: Optional[str]) -> None:
    local = 'local_dir'
    remote = 'remote_dir'
    stream = Stream(remote=remote, local=local, split=split)
    assert local == stream.local
