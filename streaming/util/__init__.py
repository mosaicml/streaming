# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for streaming."""

from streaming.util.auto import Auto
from streaming.util.importing import get_import_exception_message, redirect_imports
from streaming.util.merging import merge_index
from streaming.util.retrying import retry
from streaming.util.shared import clean_stale_shared_memory
from streaming.util.shorthand import (get_list_arg, get_str2str_arg, normalize_bin_bytes,
                                      normalize_bytes, normalize_count, normalize_dec_bytes,
                                      normalize_duration)
from streaming.util.tabulation import Tabulator
from streaming.util.waiting import wait_for, wait_for_creation, wait_for_deletion

__all__ = [
    'Auto', 'get_import_exception_message', 'redirect_imports', 'merge_index', 'retry',
    'clean_stale_shared_memory', 'get_list_arg', 'get_str2str_arg', 'normalize_dec_bytes',
    'normalize_bin_bytes', 'normalize_bytes', 'normalize_count', 'normalize_duration', 'Tabulator',
    'wait_for', 'wait_for_creation', 'wait_for_deletion'
]
