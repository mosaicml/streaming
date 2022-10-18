# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""MosaicML Streaming Datasets for cloud-native model training."""

import streaming.text as text
import streaming.vision as vision
from streaming._version import __version__
from streaming.base import (CSVWriter, JSONWriter, LocalIterableDataset, LocalMapDataset,
                            MDSWriter, StreamingDataset, TSVWriter, XSVWriter)

Dataset = StreamingDataset

__all__ = [
    'CSVWriter', 'JSONWriter', 'MDSWriter', 'TSVWriter', 'XSVWriter', 'LocalIterableDataset',
    'LocalMapDataset', 'StreamingDataset', 'vision', 'text', 'Dataset'
]
