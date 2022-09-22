# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Module to write and read the dataset in JSON format."""

from streaming.base.format.json.reader import JSONReader
from streaming.base.format.json.writer import JSONWriter

__all__ = ['JSONReader', 'JSONWriter']
