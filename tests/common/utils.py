# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
from typing import List, Optional, Tuple

import pytest


@pytest.fixture
def remote_local(tmp_path: pathlib.Path) -> Tuple[str, str]:
    remote = tmp_path.joinpath('remote')
    local = tmp_path.joinpath('local')
    return str(remote), str(local)


@pytest.fixture
def compressed_remote_local(tmp_path: pathlib.Path) -> Tuple[str, str, str]:
    compressed = tmp_path.joinpath('compressed')
    remote = tmp_path.joinpath('remote')
    local = tmp_path.joinpath('local')
    return tuple(str(x) for x in [compressed, remote, local])


def get_config_in_bytes(format: str,
                        size_limit: int,
                        column_names: List[str],
                        column_encodings: List[str],
                        column_sizes: List[str],
                        compression: Optional[str] = None,
                        hashes: Optional[List[str]] = []):
    config = {
        'version': 2,
        'format': format,
        'compression': compression,
        'hashes': hashes,
        'size_limit': size_limit,
        'column_names': column_names,
        'column_encodings': column_encodings,
        'column_sizes': column_sizes
    }
    return json.dumps(config, sort_keys=True).encode('utf-8')
