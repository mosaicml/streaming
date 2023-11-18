# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Methods having to do with streaming dataset indexes."""

__all__ = ['get_index_basename']


def get_index_basename() -> str:
    """Get the canonical index file basename.

    Returns:
        str: Index basename.
    """
    return 'index.json'
