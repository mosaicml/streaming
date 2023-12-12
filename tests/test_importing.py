# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0


def test_redirect_imports():
    from streaming.base.util import get_import_exception_message  # pyright: ignore

    # Import successful.
