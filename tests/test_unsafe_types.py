# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from shutil import rmtree
from tempfile import mkdtemp

import pytest

from streaming import MDSWriter, StreamingDataset


def test_do_allow_unsafe_types_safe():
    local = mkdtemp()
    columns = {'num': 'int'}
    with MDSWriter(out=local, columns=columns) as out:
        for num in range(100):
            sample = {'num': num}
            out.write(sample)
    dataset = StreamingDataset(local=local, allow_unsafe_types=True)
    del dataset
    rmtree(local)


def test_do_allow_unsafe_types_unsafe():
    local = mkdtemp()
    columns = {'num': 'pkl'}
    with MDSWriter(out=local, columns=columns) as out:
        for num in range(100):
            sample = {'num': num}
            out.write(sample)
    dataset = StreamingDataset(local=local, allow_unsafe_types=True)
    del dataset
    rmtree(local)


def test_dont_allow_unsafe_types_safe():
    local = mkdtemp()
    columns = {'num': 'int'}
    with MDSWriter(out=local, columns=columns) as out:
        for num in range(100):
            sample = {'num': num}
            out.write(sample)
    dataset = StreamingDataset(local=local)
    del dataset
    rmtree(local)


def test_dont_allow_unsafe_types_unsafe():
    local = mkdtemp()
    columns = {'num': 'pkl'}
    with MDSWriter(out=local, columns=columns) as out:
        for num in range(100):
            sample = {'num': num}
            out.write(sample)
    with pytest.raises(ValueError, match='.*contains an unsafe type.*'):
        dataset = StreamingDataset(local=local)
        del dataset
    rmtree(local)
