# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Handling for jobs, which are collections of StreamingDataset replicas with the same config."""

from streaming.base.coord.job.dir import JobDir
from streaming.base.coord.job.registry import JobRegistry

__all__ = ['JobDir', 'JobRegistry']
