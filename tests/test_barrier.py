# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import re
from multiprocessing.managers import ListProxy
from random import random
from time import sleep
from typing import Any

import numpy as np
import pytest

from streaming.base.shared import SharedBarrier


class TestSharedBarrier:

    @pytest.mark.parametrize('filelock_path', ['/tmp/dir/file_path'])
    @pytest.mark.parametrize('shm_path', ['barrier_shm_path'])
    def test_params(self, filelock_path: str, shm_path: str):
        barrier = SharedBarrier(filelock_path, shm_path, True)
        assert barrier.filelock_path == filelock_path
        assert barrier.shm_path == shm_path
        assert isinstance(barrier._arr, np.ndarray)
        assert barrier._arr.shape == (3,)
        assert barrier.num_enter == 0
        assert barrier.num_exit == -1
        assert barrier.flag is True

    @pytest.mark.parametrize('num_enter', [3, 10])
    @pytest.mark.parametrize('num_exit', [4, 9])
    @pytest.mark.parametrize('flag', [True, False])
    def test_setter_getter(self, num_enter: int, num_exit: int, flag: bool):
        barrier = SharedBarrier('/tmp/dir/file_path', 'barrier_shm_path', True)
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
    def test_barrier(self, num_process: int):
        mp.set_start_method('fork', force=True)
        manager = mp.Manager()
        shared_list = manager.list()
        barrier = SharedBarrier('/tmp/dir/file_path', 'barrier_shm_path', True)
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
