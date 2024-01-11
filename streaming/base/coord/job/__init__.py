# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Handling for jobs, which are collections of StreamingDataset replicas with the same config."""

from streaming.base.coord.job.directory import JobDirectory
from streaming.base.coord.job.registry import JobRegistry

__all__ = ['JobDirectory', 'JobRegistry']
