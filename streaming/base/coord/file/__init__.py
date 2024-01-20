# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinating processes using files."""

from streaming.base.coord.file.lock import SoftFileLock
from streaming.base.coord.file.waiting import create_file, wait_for_creation, wait_for_deletion

__all__ = ['create_file', 'wait_for_creation', 'wait_for_deletion', 'SoftFileLock']
