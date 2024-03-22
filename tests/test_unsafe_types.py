# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest

from streaming import MDSWriter, StreamingDataset


@pytest.mark.usefixtures('local_remote_dir')
def test_do_allow_unsafe_types_safe(local_remote_dir: Tuple[str, str]):
    local, _ = local_remote_dir
    columns = {'num': 'int'}
    with MDSWriter(out=local, columns=columns) as out:
        for num in range(100):
            sample = {'num': num}
            out.write(sample)
    StreamingDataset(local=local, allow_unsafe_types=True, batch_size=1)


@pytest.mark.usefixtures('local_remote_dir')
def test_do_allow_unsafe_types_unsafe(local_remote_dir: Tuple[str, str]):
    local, _ = local_remote_dir
    columns = {'num': 'pkl'}
    with MDSWriter(out=local, columns=columns) as out:
        for num in range(100):
            sample = {'num': num}
            out.write(sample)
    StreamingDataset(local=local, allow_unsafe_types=True, batch_size=1)


@pytest.mark.usefixtures('local_remote_dir')
def test_dont_allow_unsafe_types_safe(local_remote_dir: Tuple[str, str]):
    local, _ = local_remote_dir
    columns = {'num': 'int'}
    with MDSWriter(out=local, columns=columns) as out:
        for num in range(100):
            sample = {'num': num}
            out.write(sample)
    StreamingDataset(local=local, batch_size=1)


@pytest.mark.usefixtures('local_remote_dir')
def test_dont_allow_unsafe_types_unsafe(local_remote_dir: Tuple[str, str]):
    local, _ = local_remote_dir
    columns = {'num': 'pkl'}
    with MDSWriter(out=local, columns=columns) as out:
        for num in range(100):
            sample = {'num': num}
            out.write(sample)
    with pytest.raises(ValueError, match='.*contains an unsafe type.*'):
        StreamingDataset(local=local, batch_size=1)
