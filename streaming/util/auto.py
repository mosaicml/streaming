# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A magical argument keyword that means derive this argument's value automatically."""

__all__ = ['Auto', 'auto']


class Auto:
    """Keyword that tells the argument to take its default value.

    This is useful when your argument's type doesn't have a canonical blank space which you can
    repurpose to mean "default" (such as ``-1`` if non-negative ``int``, ``''`` if ``str``, etc.),
    and ``None`` has its own productive meaning, and using a different type would not make a lot of
    semantic sense.
    """

    pass


# The singleton instance of this class.
auto = Auto()
