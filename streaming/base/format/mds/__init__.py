# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Module to write and read the dataset in MDS format."""

from streaming.base.format.mds.reader import MDSReader
from streaming.base.format.mds.writer import MDSWriter

__all__ = ['MDSReader', 'MDSWriter']
