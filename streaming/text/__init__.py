# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported NLP datasets."""

from streaming.text.c4 import StreamingC4 as StreamingC4
from streaming.text.enwiki import StreamingEnWiki as StreamingEnWiki

__all__ = ['StreamingC4', 'StreamingEnWiki']
