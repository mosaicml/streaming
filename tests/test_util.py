# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import pytest

from streaming.base.util import get_list_arg


@pytest.mark.parametrize(('text', 'expected_output'), [('hello,world', ['hello', 'world']),
                                                       ('hello', ['hello']), ('', [])])
def test_get_list_arg(text: str, expected_output: List[Optional[str]]):
    output = get_list_arg(text)
    assert output == expected_output
