# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported NLP datasets."""

from streaming.text.c4 import C4 as C4
from streaming.text.enwiki import EnWiki as EnWiki

__all__ = ['C4', 'EnWiki']
