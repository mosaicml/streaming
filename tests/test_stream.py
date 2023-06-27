# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import tempfile
from typing import Optional

import pytest
from _pytest.monkeypatch import MonkeyPatch

from streaming import Stream


@pytest.mark.usefixtures('local_remote_dir')
def test_local_is_none_with_no_split(monkeypatch: MonkeyPatch,
                                     local_remote_dir: tuple[str, str]) -> None:
    local, remote = local_remote_dir
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: local)
    remote_hash = hashlib.md5(remote.encode('utf-8')).hexdigest()
    stream = Stream(remote=remote, local=None)
    assert os.path.join(local, remote_hash) == stream.local


@pytest.mark.usefixtures('local_remote_dir')
def test_local_is_none_with_split(monkeypatch: MonkeyPatch, local_remote_dir: tuple[str,
                                                                                    str]) -> None:
    local, remote = local_remote_dir
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: local)
    remote_hash = hashlib.md5(remote.encode('utf-8')).hexdigest()
    stream = Stream(remote=remote, local=None, split='train')
    assert os.path.join(local, remote_hash + '_' + 'train') == stream.local


@pytest.mark.parametrize('split', [None, 'train'])
@pytest.mark.usefixtures('local_remote_dir')
def test_local_exists(local_remote_dir: tuple[str, str], split: Optional[str]) -> None:
    local, remote = local_remote_dir
    stream = Stream(remote=remote, local=local, split=split)
    assert local == stream.local
