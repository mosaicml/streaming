# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from streaming.base import StreamingDataset
from streaming.base.constant import LOCALS
from streaming.base.shared import SharedArray, get_shm_prefix
from streaming.base.shared.memory import SharedMemory
from streaming.base.shared.prefix import _check_and_find
from streaming.base.util import clean_stale_shared_memory
from streaming.base.world import World
from tests.common.utils import convert_to_mds


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix(local_remote_dir: tuple[str, str]):
    local, remote = local_remote_dir

    _, _ = get_shm_prefix(streams_local=[local], streams_remote=[remote], world=World.detect())


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix_same_local_dir(local_remote_dir: tuple[str, str]):
    local, remote = local_remote_dir
    with pytest.raises(ValueError, match='Reused local directory.*Provide a different one.'):
        _, _ = get_shm_prefix(streams_local=[local, local],
                              streams_remote=[remote, remote],
                              world=World.detect())


@pytest.mark.usefixtures('local_remote_dir')
def test_get_shm_prefix_same_split_dir(local_remote_dir: tuple[str, str]):
    local, remote = local_remote_dir
    _, _ = get_shm_prefix(streams_local=[local, remote],
                          streams_remote=[local, remote],
                          world=World.detect())
    with pytest.raises(ValueError, match='Reused local directory.*vs.*Provide a different one.'):
        _, _ = get_shm_prefix(streams_local=[local, remote],
                              streams_remote=[local, remote],
                              world=World.detect())


def test_same_local_remote_none(local_remote_dir: tuple[str, str]):
    local, _ = local_remote_dir
    _, _ = get_shm_prefix(streams_local=[local], streams_remote=[None], world=World.detect())
    _, _ = get_shm_prefix(streams_local=[local], streams_remote=[None], world=World.detect())


@pytest.mark.parametrize('from_beginning', [True, False])
@pytest.mark.usefixtures('local_remote_dir')
def test_load_get_state_dict_once(local_remote_dir: tuple[str, str], from_beginning: bool):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)
    dataset = StreamingDataset(local=local, remote=remote, batch_size=1)

    # Get the current dataset state dict
    old_state_dict = dataset.state_dict(0, from_beginning)
    assert old_state_dict is not None

    state_keys = list(old_state_dict.keys())

    # Change the state dict and load it back to the dataset.
    new_state_dict = old_state_dict.copy()
    for key in state_keys:
        new_state_dict[key] += 1
    dataset.load_state_dict(new_state_dict)

    new_loaded_state_dict = dataset.state_dict(0, from_beginning)
    assert new_loaded_state_dict is not None
    if from_beginning:
        for key in state_keys:
            if key == 'sample_in_epoch':
                # If `from_beginning` is True, we expect sample_in_epoch to be 0.
                assert new_loaded_state_dict[key] == 0
            else:
                # All other fields in retrieved and loaded state dicts should match.
                assert new_loaded_state_dict[key] == new_state_dict[key]
    else:
        # If `from_beginning` is False, retrieved and loaded state dicts should match completely.
        assert new_loaded_state_dict == new_state_dict

    for key in state_keys:
        if key == 'sample_in_epoch' and from_beginning:
            # If `from_beginning` is True, we expect sample_in_epoch to be the same, 0.
            assert new_loaded_state_dict[key] == old_state_dict[key]
        else:
            assert new_loaded_state_dict[key] == old_state_dict[key] + 1


@pytest.mark.parametrize('iterations', [10])
@pytest.mark.usefixtures('local_remote_dir')
def test_load_get_state_dict_multiple(local_remote_dir: tuple[str, str], iterations: int):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)
    dataset = StreamingDataset(local=local, remote=remote, batch_size=1)

    # Get the current dataset state dict
    old_state_dict = dataset.state_dict(0, False)
    assert old_state_dict is not None

    state_keys = list(old_state_dict.keys())

    for _ in range(iterations):
        # Change the state dict and load it back to the dataset.
        new_state_dict = old_state_dict.copy()
        for key in state_keys:
            # If the epoch from the loaded state dict is -1, make sure that the new epoch
            # is greater than -1. Otherwise, we will assume a stale resumption state, ignoring it.
            if key == 'epoch' and new_state_dict[key] < 0:
                new_state_dict[key] *= -5
            else:
                new_state_dict[key] *= 5

        dataset.load_state_dict(new_state_dict)
        new_loaded_state_dict = dataset.state_dict(0, False)

        assert new_loaded_state_dict is not None
        assert new_loaded_state_dict == new_state_dict
        for key in state_keys:
            # Ensure we check that epoch has been correctly updated, in case it was negative.
            if key == 'epoch' and old_state_dict[key] < 0:
                assert new_loaded_state_dict[key] == old_state_dict[key] * -5
            else:
                assert new_loaded_state_dict[key] == old_state_dict[key] * 5

        old_state_dict = new_loaded_state_dict


@pytest.mark.usefixtures('local_remote_dir')
def test_state_dict_too_large(local_remote_dir: tuple[str, str]):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)
    dataset = StreamingDataset(local=local, remote=remote, batch_size=1)

    # Make a state dict that is too large to fit in the allocated shared memory.
    import mmap
    key = 'a' * mmap.PAGESIZE
    big_state_dict = {key: 1}

    with pytest.raises(ValueError, match='The StreamingDataset state dict*'):
        dataset.load_state_dict(big_state_dict)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
@patch('streaming.base.shared.array.SharedMemory')
def test_shared_array_size_is_integer(mock_shared_memory: MagicMock, dtype: type[np.dtype]):
    SharedArray(3, dtype=dtype, name='test_shared_array')
    mock_shared_memory.assert_called_once()  # pyright: ignore
    size_arg = mock_shared_memory.call_args[1]['size']
    assert isinstance(size_arg, int), 'Size passed to SharedMemory is not an integer'


def test_check_and_find_skips_filelock_conflict():
    """Test _check_and_find skips prefix due to file lock conflict."""
    clean_stale_shared_memory()

    with patch('os.path.exists') as mock_exists, \
         patch('multiprocessing.shared_memory.SharedMemory', side_effect=FileNotFoundError):
        # Simulate that `/000000.barrier_filelock` exists, indicating a lock conflict
        bf_path = os.path.join(tempfile.gettempdir(), '000000_barrier_filelock')
        mock_exists.side_effect = lambda path: path == bf_path

        # Expect _check_and_find to return 1 as the next available prefix
        next_prefix = _check_and_find(['local_dir'], [None], LOCALS)
        assert next_prefix == 1


@patch.object(SharedMemory,
              '__init__',
              side_effect=[
                  PermissionError('Mocked permission error'),
                  FileNotFoundError('Mocked file not found error')
              ])
def test_shared_memory_permission_error(mock_shared_memory_class: MagicMock):
    with patch('os.path.exists', return_value=False):
        next_prefix = _check_and_find(['local'], [None], LOCALS)
        assert next_prefix == 1
