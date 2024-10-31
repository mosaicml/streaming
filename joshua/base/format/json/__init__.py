# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Module to write and read the dataset in JSON format."""

from joshua.base.format.json.reader import JSONReader
from joshua.base.format.json.writer import JSONWriter

__all__ = ['JSONReader', 'JSONWriter']
