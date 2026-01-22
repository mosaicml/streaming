# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
import shutil
import tempfile
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from typing import Any, Callable
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


# Global counter to track attach attempts (per process)
attach_attempts = 0


def patched_shared_memory_init(original_init: Callable):
    """Wrapper that fails first 3 attach attempts for non-local leaders."""

    def wrapper(self: Any, name: str, create: bool = False, size: int = -1):
        global attach_attempts

        # Only interfere with attach (create=False) for specific shared memory names
        if not create and name and 'locals' in name:
            attach_attempts += 1
            # Fail first 3 attempts to simulate OS propagation delay
            if attach_attempts <= 3:
                print(
                    f'    [Mock] Attach attempt {attach_attempts} - simulating FileNotFoundError')
                raise FileNotFoundError(f'[Mock] Simulating OS propagation delay for {name}')
            else:
                print(f'    [Mock] Attach attempt {attach_attempts} - allowing success')

        # Call original init
        return original_init(self, name, create, size)

    return wrapper


def worker_process(rank: int, world_size: int, dataset_path: str):
    """Worker that creates StreamingDataset with forced race condition."""
    global attach_attempts
    attach_attempts = 0  # Reset counter for this process

    try:
        import torch.distributed as dist

        # Patch SharedMemory BEFORE importing streaming
        # This simulates slow OS propagation
        with patch.object(BuiltinSharedMemory, '__init__',
                          patched_shared_memory_init(BuiltinSharedMemory.__init__)):
            from streaming import StreamingDataset

            # Initialize distributed
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['LOCAL_RANK'] = str(rank)
            os.environ['LOCAL_WORLD_SIZE'] = str(world_size)

            dist.init_process_group(backend='gloo',
                                    init_method=os.environ['MASTER_ADDR'],
                                    rank=rank,
                                    world_size=world_size)

            print(f'[Rank {rank}] Creating StreamingDataset...')

            # On MAIN branch (no retry): Will fail immediately on first FileNotFoundError
            # On FIX branch (with retry): Will retry and succeed after 3 attempts
            dataset = StreamingDataset(local=dataset_path,
                                       remote=None,
                                       shuffle=False,
                                       batch_size=4)

            print(f'[Rank {rank}] ✅ Success! Dataset created with {len(dataset)} samples')
            dist.destroy_process_group()
            return True

    except (FileNotFoundError, RuntimeError) as e:
        if 'shared memory prefix' in str(e) or 'FileNotFoundError' in str(e):
            print(f'[Rank {rank}] ❌ FAILED - Issue #824 (no retry): {e}')
        else:
            print(f'[Rank {rank}] ❌ Unexpected error: {e}')
        return False
    except Exception as e:
        print(f'[Rank {rank}] ❌ Unexpected error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_forced_race():
    """Test with forced race condition."""

    # Create test dataset
    temp_dir = tempfile.mkdtemp(prefix='forced_race_')
    dataset_path = os.path.join(temp_dir, 'dataset')

    try:
        # Create MDS dataset
        from streaming import MDSWriter
        os.makedirs(dataset_path, exist_ok=True)

        with MDSWriter(out=dataset_path, columns={'id': 'int', 'value': 'str'}) as writer:
            for i in range(100):
                writer.write({'id': i, 'value': f'sample_{i}'})

        print(f'Created test dataset at {dataset_path}\n')

        # Clean stale shared memory
        from streaming.base.util import clean_stale_shared_memory
        clean_stale_shared_memory()

        # Setup master address
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ['MASTER_ADDR'] = f'tcp://127.0.0.1:{port}'

        # Launch processes
        ctx = mp.get_context('spawn')
        processes = []

        for rank in range(2):
            p = ctx.Process(target=worker_process, args=(rank, 2, dataset_path))
            p.start()
            processes.append(p)

        # Wait for completion with timeout
        success = True
        timeout_seconds = 60
        for p in processes:
            p.join(timeout=timeout_seconds)
            if p.is_alive():
                # Process hung, terminate it
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
                success = False
                print(f'Process {p.pid} timed out after {timeout_seconds} seconds')
            elif p.exitcode != 0:
                success = False

        assert success, 'Test FAILED - No retry logic to handle the forced race condition'

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        try:
            from streaming.base.util import clean_stale_shared_memory
            clean_stale_shared_memory()
        except:
            pass
