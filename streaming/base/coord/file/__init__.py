# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinating processes using files."""

from streaming.base.coord.file.barrier import create_file, wait_for_creation, wait_for_deletion
from streaming.base.coord.file.lock import SoftFileLock

__all__ = ['wait_for_creation', 'wait_for_deletion', 'create_file', 'SoftFileLock']
