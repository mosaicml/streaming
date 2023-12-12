# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for importing."""

import inspect
import sys
from importlib import import_module
from warnings import warn

__all__ = ['get_import_exception_message', 'redirect_imports']


def get_import_exception_message(package_name: str, extra_deps: str) -> str:
    """Get import exception message.

    Args:
        package_name (str): Package name.

    Returns:
        str: Exception message.
    """
    return f'Streaming was installed without {package_name} support. ' + \
            f'To use {package_name} related packages with Streaming, run ' + \
           f'`pip install \'mosaicml-streaming[{package_name}]\'`.'


def redirect_imports(new_fqdn: str) -> None:
    """Overlay the members of the target module onto the module of the caller.

    Args:
        new_fqdn (str): Fully-qualified dot-separated target module path.
    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    if module is None:
        raise RuntimeError('Module was None.')
    old_fqdn = module.__name__

    # old = sys.modules[old_fqdn]
    new = import_module(new_fqdn)
    sys.modules[old_fqdn].__dict__.update(new.__dict__)

    warn(f'Please update your imports: {old_fqdn} has moved to {new_fqdn}.',
         DeprecationWarning,
         stacklevel=2)
