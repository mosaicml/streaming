# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Handling for the phasing out of old forms of shard files."""

from typing import Optional, Set

# TODO
_keep_all_phases2phase_out = {
    'origin': None,
    'active': None,
    'both': None,
    'all': None,
}

# TODO
_keep_old_phases2phase_out = {
    'nil': None,
    'src': None,
    'all': None,
}


def get_phasings() -> Set[str]:
    """Get all possible values of phasing.

    Returns:
        Set[str]: All phasings.
    """
    return set(_keep_old_phases2phase_out)


def is_phasing(phasing: str) -> bool:
    """Determine whether the given str is a valid phasing.

    Args:
        phasing (Str): The purported phasing.

    Returns:
        bool: Whether it is a phasing.
    """
    return phasing in _keep_old_phases2phase_out


def get_safe_phasing(phasing: str, remote: Optional[str], local: str) -> str:
    """Get a phasing value which protects against destroying a dataset in-place.

    That is, you need the source form to be able to stream from it, but the final form to be able
    to use it. Do you drop the source phase (``nil``), keep the source (``src``), or keep all
    phases (``all``)?

    Args:
        phasing (str): Unsafe phasing.
        remote (str, optional): Remote path.
        local (str): Local dirname.

    Returns:
        str: Safe phasing.
    """
    if remote not in {None, local}:
        return phasing

    if phasing != 'nil':
        return phasing

    return 'src'
