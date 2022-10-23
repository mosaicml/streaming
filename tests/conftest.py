# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)  # pyright: ignore
def pytest_runtest_call(item: Any):
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, 'is_dist_test', False):
        dist_test_class = item.cls()
        dist_test_class._run_test(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice
