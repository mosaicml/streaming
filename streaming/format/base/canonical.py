# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Algorithms to canonicalize native-form shards into a form we can random access."""


def canonicalize(can_algo: str, raw_filename: str, can_filename: str):
    """Canonicalize a shard file from ``raw`` phase to ``can`` phase.

    Args:
        can_algo (str): Canonicalizatino algorithm.
        raw_filename (str): Path to input ``raw`` phase.
        can_filename (str): Path to output ``can`` phase.
    """
    raise NotImplementedError
