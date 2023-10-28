# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Graceful migration of StreamingDataset arguments."""

import logging
from typing import Optional

__all__ = ['get_keep_packed']

logger = logging.getLogger(__name__)


def get_keep_packed(keep_packed: Optional[bool], keep_zip: Optional[bool]) -> bool:
    """Get the value of ``keep_packed`` given both old aand new arguments.

    Warns if the deprecated argument ``keep_zip`` is used.

    Args:
        keep_packed (bool, optinoal): New argument.
        keep_zip (bool, optional): Old argument.

    Returns:
        bool: Normalized argument.
    """
    if keep_zip is not None:
        logger.warning('StreamingDataset/Stream argument `keep_zip` is deprecated, please use ' +
                       'the new `keep_packed` argument instead, which is more general.')

    if keep_packed is not None:
        return keep_packed

    if keep_zip is not None:
        return keep_zip

    return False
