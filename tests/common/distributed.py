# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
import logging
import os
import socket
import time
from typing import Any, Dict, List

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from _pytest.fixtures import FixtureRequest
from _pytest.outcomes import Skipped
from torch.multiprocessing import Process

logger = logging.getLogger(__name__)

# Worker timeout after the first worker has completed.
WORKER_TIMEOUT_IN_SECS = 120

# Local host
MASTER_ADDRESS = '127.0.0.1'


class DistributedTest:
    """A simulation of distributed behavior on a single machine."""
    is_dist_test = True
    # Number of ranks
    world_size = 1
    current_test = None
    test_kwargs = None
    port = None

    # Only `gloo` is supported for now
    __backend = 'gloo'

    def _run_test(self, request: FixtureRequest) -> None:
        """Run the current test.

        Args:
            request (FixtureRequest): A request object with a test context

        Raises:
            ValueError: `world_size` should be of type `int`
        """
        # Override the marker if any
        self._set_custom_markers(request)

        # Get a free TCP Port to listen on
        self.port = self.get_free_tcp_port()

        self.current_test = self._get_current_test_func(request)
        self.test_kwargs = self._get_test_kwargs(request)
        if not isinstance(self.world_size, int):
            raise ValueError('`world_size` should be of type `int`')
        self._launch_processes(self.world_size)
        time.sleep(0.5)

    def _get_current_test_func(self, request: FixtureRequest) -> Any:
        """Get the current test function.

        Args:
            request (FixtureRequest): A request object with a test context

        Returns:
            A test function
        """
        # Get the test method
        func_name = request.function.__name__
        return getattr(self, func_name)

    def _get_test_kwargs(self, request: FixtureRequest) -> Dict[str, Any]:
        """Get the test arguments.

        Args:
            request (FixtureRequest): A request object with a test context

        Returns:
            Dict[str, Any]: A dictionary of arguments
        """
        # Get fixture / parametrize kwargs from pytest request object
        test_kwargs = {}
        params = inspect.getfullargspec(self.current_test).args
        params.remove('self')
        for p in params:
            test_kwargs[p] = request.getfixturevalue(p)
        return test_kwargs

    def _set_custom_markers(self, request: FixtureRequest) -> None:
        """Get the custom markers and overrides it.

        Args:
            request (FixtureRequest): A request object with a test context
        """
        # Fetch custom pytest marker `world_size`
        # and override with a class parameter
        for mark in getattr(request.function, 'pytestmark', []):
            if mark.name == 'world_size':
                self.world_size = mark.args[0]

    def get_free_tcp_port(self) -> int:
        """Get a free socket port to listen on."""
        tcp = socket.socket()
        tcp.bind(('', 0))
        _, port = tcp.getsockname()
        tcp.close()
        return port

    @contextlib.contextmanager
    def _patch_env(self, **environs: str):
        """Returns a context manager that patches ``os.environ`` with ``environs``.

        The original ``os.environ`` values are restored at the end.
        """
        # Capture the original environ values
        original_environs = {k: os.environ.get(k) for k in environs}

        # Patch the environment
        for k, v in environs.items():
            os.environ[k] = v
        try:
            # Run the context manager
            yield
        finally:
            # Restore the original environ values
            for k, v in original_environs.items():
                if v is None:
                    del os.environ[k]
                else:
                    os.environ[k] = v

    def _launch_processes(self, nproc: int) -> None:
        """Launch and monitor the processes.

        Args:
            nproc (int): Total number of processes
        """
        mp.set_start_method('fork', force=True)
        skip_msg = mp.Queue()  # Allows forked processes to share pytest.skip reason
        processes = []
        for global_rank in range(nproc):
            p = Process(target=self._dist_init, args=(global_rank, nproc, skip_msg))
            p.start()
            processes.append(p)

        self._monitor_processes(processes)

        if not skip_msg.empty():
            # Skip the test if there is a pytest skip marker
            pytest.skip(skip_msg.get())

    def _monitor_processes(self, processes: List[Process]) -> None:
        """Monitor all the process to ensure it finishes. If any process fails unexpectedly, then
        terminate the test execution.

        Args:
            processes (List[Process]): A list of process object
        """
        # Loop through all processes and wait for it to complete.
        # Fail the test if any process died unexpectedly.
        any_done = False
        while not any_done:
            for p in processes:
                if not p.is_alive():
                    any_done = True
                    break

        # Wait for all other processes to complete
        for p in processes:
            p.join(WORKER_TIMEOUT_IN_SECS)

        failed = [(rank, p) for rank, p in enumerate(processes) if p.exitcode != 0]
        for rank, p in failed:
            # If it still hasn't terminated, kill it because it hung.
            if p.exitcode is None:
                p.terminate()
                p.close()
                pytest.fail(f'Worker {rank} hung.', pytrace=False)
            if p.exitcode < 0:
                pytest.fail(f'Worker {rank} killed by signal {-p.exitcode}', pytrace=False)
            if p.exitcode > 0:
                pytest.fail(f'Worker {rank} exited with code {p.exitcode}', pytrace=False)

    def _dist_init(self, global_rank: int, num_procs: int, skip_msg: mp.Queue):
        """Initialize distributed process group and execute the user function.

        Args:
            global_rank (int): A global rank number
            num_procs (int): Total number of processes
            skip_msg (mp.Queue): A multiprocessing queue

        Raises:
            e: Exception if test execution fails
        """
        with self._patch_env(
                RANK=str(global_rank),
                WORLD_SIZE=str(num_procs),
                LOCAL_RANK=str(0),
                MASTER_ADDR=MASTER_ADDRESS,
                MASTER_PORT=str(self.port),
                PYTHONUNBUFFERED='1',
        ):
            # Initializes the default distributed process group
            dist.init_process_group(backend=self.__backend, rank=global_rank, world_size=num_procs)
            # Synchronizes all processes
            dist.barrier()

            try:
                self.current_test(**self.test_kwargs)  # pyright: ignore
            except BaseException as e:
                if isinstance(e, Skipped):
                    skip_msg.put(e.msg)
                else:
                    raise e

            self._cleanup(dist)

    def _cleanup(self, dist: Any):
        """Clean up distributed process group.

        Args:
            dist (Any): A distributed process group
        """
        # Synchronizes all processes
        dist.barrier()
        # tear down after test completes
        dist.destroy_process_group()
