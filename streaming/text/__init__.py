# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported NLP datasets."""

from streaming.text.c4 import StreamingC4 as StreamingC4
from streaming.text.enwiki import StreamingEnWiki as StreamingEnWiki
from streaming.text.pile import StreamingPile as StreamingPile

__all__ = ['StreamingPile', 'StreamingC4', 'StreamingEnWiki']
