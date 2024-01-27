# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A magical argument keyword that means derive this argument's value automatically."""

from typing import Any

__all__ = ['Auto', 'auto', 'is_auto']


class Auto:
    """A magical argument keyword that means derive this argument's value automatically.

    This is useful when your argument's type doesn't have any blank space like ``0`` or ``''`` in
    this method's usage, ``None`` has its own productive meaning, and using a different type would
    be ugly and hard to follow.
    """
    pass


# The singleton instance of this class.
auto = Auto()


def is_auto(arg: Any) -> bool:
    """Wrap the is-auto checking hack.

    Typechecking is not satisfied with `is auto`, you have to do `isinstance(Auto)`.

    Args:
        arg (Any): The argument.

    Returns:
        bool: Whether the argument is auto.
    """
    return isinstance(arg, Auto)
