# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
import re
from multiprocessing.managers import ListProxy
from random import random
from time import sleep
from typing import Any

import pytest

from streaming.base.shared import SharedArray, SharedBarrier


class TestSharedBarrier:

    @pytest.mark.parametrize('filelock_path', ['barrier_filelock_path'])
    @pytest.mark.parametrize('shm_name', ['barrier_shm_name'])
    def test_params(self, filelock_path: str, shm_name: str):
        barrier = SharedBarrier(filelock_path, shm_name)
        assert isinstance(barrier._arr, SharedArray)
        assert barrier._arr.shape == (3,)
        assert barrier.num_enter == 0
        assert barrier.num_exit == -1
        assert barrier.flag is True

    @pytest.mark.parametrize('num_enter', [3, 10])
    @pytest.mark.parametrize('num_exit', [4, 9])
    @pytest.mark.parametrize('flag', [True, False])
    def test_setter_getter(self, num_enter: int, num_exit: int, flag: bool):
        barrier = SharedBarrier('/tmp/dir/filelock_path', 'barrier_shm_name')
        barrier.num_enter = num_enter
        assert barrier.num_enter == num_enter
        barrier.num_exit = num_exit
        assert barrier.num_exit == num_exit
        barrier.flag = flag
        assert barrier.flag == flag

    def run(self, num_process: int, barrier: Any, shared_list: ListProxy):
        sleep(random())
        shared_list.append(f'Hit barrier, waiting: {mp.current_process().name}')
        barrier(num_process)
        shared_list.append(f'passed barrier: {mp.current_process().name}')
        barrier(num_process)
        shared_list.append(f'passed barrier again: {mp.current_process().name}')

    @pytest.mark.parametrize('num_process', [2, 3])
    @pytest.mark.parametrize('filelock_root', ['/tmp/dir/'])
    def test_barrier(self, num_process: int, filelock_root: str):
        mp.set_start_method('fork', force=True)
        manager = mp.Manager()
        shared_list = manager.list()
        os.makedirs(filelock_root, exist_ok=True)
        barrier = SharedBarrier(os.path.join(filelock_root, 'filelock_path'), 'barrier_shm_name')
        processes = [
            mp.Process(target=self.run, args=(num_process, barrier, shared_list))
            for _ in range(num_process)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        expected_log_message = [''] * 3 * num_process
        for i in range(num_process):
            expected_log_message[i] = f'Hit barrier, waiting: Process-\\d+'
            expected_log_message[i + num_process] = f'passed barrier: Process-\\d+'
            expected_log_message[i + (2 * num_process)] = f'passed barrier again: Process-\\d+'

        assert re.fullmatch(' '.join(expected_log_message), ' '.join(shared_list))
