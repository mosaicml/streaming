# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""MosaicML Streaming Datasets for cloud-native model training."""

import multiprocessing as mp
import sys

import streaming.text as text
import streaming.vision as vision
from streaming._version import __version__
from streaming.base import (CSVWriter, JSONWriter, LocalDataset, MDSWriter, StreamingDataLoader,
                            StreamingDataset, TSVWriter, XSVWriter)

__all__ = [
    'StreamingDataLoader', 'StreamingDataset', 'CSVWriter', 'JSONWriter', 'MDSWriter', 'TSVWriter',
    'XSVWriter', 'LocalDataset', 'vision', 'text'
]

IS_MACOS = sys.platform == 'darwin'

# Set the multiprocessing start method to `fork` for MAC OS since
# streaming uses a FileLock for sharing the resources between ranks
# and workers which is Unpickleable except method `fork`.
if IS_MACOS:
    mp.set_start_method('fork', force=True)
