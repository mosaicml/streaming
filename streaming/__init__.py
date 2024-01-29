# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""MosaicML Streaming Datasets for cloud-native model training."""

from streaming._version import __version__
from streaming.dataloader import StreamingDataLoader
from streaming.dataset import StreamingDataset
from streaming.format import CSVWriter, JSONLWriter, MDSWriter, TSVWriter, XSVWriter
from streaming.local import LocalDataset
from streaming.stream import Stream
from streaming.util import clean_stale_shared_memory

__all__ = [
    'StreamingDataLoader', 'Stream', 'StreamingDataset', 'CSVWriter', 'JSONLWriter',
    'LocalDataset', 'MDSWriter', 'TSVWriter', 'XSVWriter', 'clean_stale_shared_memory'
]
