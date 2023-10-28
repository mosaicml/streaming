# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""User-friendly import exception message."""

__all__ = ['get_import_exception_message']


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
