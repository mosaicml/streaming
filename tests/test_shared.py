# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest

from streaming.base.shared import get_shm_prefix
from streaming.base.coord.world import World


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix(local_remote_dir: Tuple[str, str]):
    local, remote = local_remote_dir

    _, _ = get_shm_prefix(streams_local=[local], streams_remote=[remote], world=World())


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix_same_local_dir(local_remote_dir: Tuple[str, str]):
    local, remote = local_remote_dir
    with pytest.raises(ValueError, match='Reused local directory.*Provide a different one.'):
        _, _ = get_shm_prefix(streams_local=[local, local],
                              streams_remote=[remote, remote],
                              world=World())


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix_same_split_dir(local_remote_dir: Tuple[str, str]):
    local, remote = local_remote_dir
    _, _ = get_shm_prefix(streams_local=[local, remote],
                          streams_remote=[local, remote],
                          world=World())
    with pytest.raises(ValueError, match='Reused local directory.*vs.*Provide a different one.'):
        _, _ = get_shm_prefix(streams_local=[local, remote],
                              streams_remote=[local, remote],
                              world=World())


def test_same_local_remote_none(local_remote_dir: Tuple[str, str]):
    local, _ = local_remote_dir
    _, _ = get_shm_prefix(streams_local=[local], streams_remote=[None], world=World())
    _, _ = get_shm_prefix(streams_local=[local], streams_remote=[None], world=World())
