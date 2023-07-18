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
    with pytest.raises(
            ValueError,
            match=
            f'Could not create a local directory. Specify a local directory with the `local` value.'
    ):
        _ = Stream()
    shutil.rmtree(local, ignore_errors=True)
