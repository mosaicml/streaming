# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""JSON typing support."""

from typing import Dict, List, Union

JSON = Union[Dict[str, 'JSON'], List['JSON'], str, float, int, bool, None]
JSONDict = Dict[str, JSON]
