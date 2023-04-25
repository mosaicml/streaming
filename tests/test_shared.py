# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest

from streaming.base.shared import get_shm_prefix
from streaming.base.world import World


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix(local_remote_dir: Tuple[str, str]):
    local, remote = local_remote_dir
    my_locals = [local, remote]

    _, _ = get_shm_prefix(my_locals=my_locals, world=World())


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix_same_local_dir(local_remote_dir: Tuple[str, str]):
    local, _ = local_remote_dir
    my_locals = [local, local]
    with pytest.raises(ValueError, match='Reused local directory.*Provide a different one.'):
        _, _ = get_shm_prefix(my_locals=my_locals, world=World())


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix_same_split_dir(local_remote_dir: Tuple[str, str]):
    local, remote = local_remote_dir
    my_locals = [local, remote]
    _, _ = get_shm_prefix(my_locals=my_locals, world=World())
    with pytest.raises(ValueError, match='Reused local directory.*vs.*Provide a different one.'):
        _, _ = get_shm_prefix(my_locals=my_locals, world=World())
