# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported NLP datasets."""

from joshua.text.c4 import StreamingC4 as StreamingC4
from joshua.text.enwiki import StreamingEnWiki as StreamingEnWiki
from joshua.text.pile import StreamingPile as StreamingPile

__all__ = ['StreamingPile', 'StreamingC4', 'StreamingEnWiki']
