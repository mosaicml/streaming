# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Setter and Getter for support hashing algorithms."""

import hashlib
from typing import Any, Callable, Dict, Set

import xxhash

__all__ = ['get_hash', 'get_hashes', 'is_hash']


def _collect() -> Dict[str, Callable[[bytes], Any]]:
    """Get all supported hash algorithms.

    Returns:
        Dict[str, Callable[[bytes], Any]]: Mapping of name to hash.
    """
    hashes = {}
    for algo in hashlib.algorithms_available:
        if hasattr(hashlib, algo) and not algo.startswith('shake_'):
            hashes[algo] = getattr(hashlib, algo)
    for algo in xxhash.algorithms_available:  # pyright: ignore
        assert algo not in hashes
        hashes[algo] = getattr(xxhash, algo)
    return hashes


# Hash algorithms (name -> function).
_hashes = _collect()


def get_hashes() -> Set[str]:
    """List supported hash algorithms.

    Returns:
        Set[str]: Hash algorithm names.
    """
    return set(_hashes)


def is_hash(algo: str) -> bool:
    """Get whether this is a supported hash algorithm.

    Args:
        algo (str): Hash algorithm.

    Returns:
        bool: Whether supported.
    """
    return algo in _hashes


def get_hash(algo: str, data: bytes) -> str:
    """Apply the hash algorithm to the data.

    Args:
        algo (str): Hash algorithm.
        data (bytes): Data to hash.

    Returns:
        str: Hex digest.
    """
    if not is_hash(algo):
        raise ValueError(f'{algo} is not a supported hash algorithm.')
    func = _hashes[algo]
    return func(data).hexdigest()
