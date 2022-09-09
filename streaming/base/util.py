# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import List


def get_list_arg(text: str) -> List[str]:
    """Pass a list as a commandline flag.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: Splits, if any.
    """
    return text.split(',') if text else []
