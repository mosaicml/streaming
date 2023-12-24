# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A magical argument keyword that means derive this argument's value automatically."""

__all__ = ['Auto', 'auto']


class Auto:
    """A magical argument keyword that means derive this argument's value automatically.

    This is useful when your argument's type doesn't have any blank space like ``0`` or ``''`` in
    this method's usage, ``None`` has its own productive meaning, and using a different type would
    be ugly and hard to follow.
    """
    pass


# The singleton instance of this class.
auto = Auto()
