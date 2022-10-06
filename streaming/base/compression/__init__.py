# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of compression algorithm."""

from streaming.base.compression.compression import (compress, decompress,
                                                    get_compression_extension, get_compressions,
                                                    is_compression)

__all__ = [
    'compress', 'decompress', 'get_compression_extension', 'get_compressions', 'is_compression'
]
