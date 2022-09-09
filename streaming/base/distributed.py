# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os

__all__ = ['get_global_rank', 'get_local_rank', 'get_local_world_size', 'get_world_size']


def get_global_rank():
    return int(os.environ.get('RANK', 0))


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def get_local_world_size():
    return int(os.environ.get('LOCAL_WORLD_SIZE', 1))
