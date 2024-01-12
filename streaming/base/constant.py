# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Constants."""

# Shared Memory names
LOCALS = 'locals'
BARRIER = 'barrier'
NEXT_EPOCH = 'next_epoch'
CACHE_USAGE = 'cache_usage'
SHARD_STATES = 'shard_states'
SHARD_ACCESS_TIMES = 'shard_access_times'
RESUME = 'resume'
EPOCH_SHAPE = 'epoch_shape'
EPOCH_DATA = 'epoch_data'
SHM_TO_CLEAN = [
    LOCALS,
    BARRIER,
    NEXT_EPOCH,
    CACHE_USAGE,
    SHARD_STATES,
    SHARD_ACCESS_TIMES,
    RESUME,
    EPOCH_SHAPE,
    EPOCH_DATA,
]

# filelock names
BARRIER_FILELOCK = 'barrier_filelock'
CACHE_FILELOCK = 'cache_filelock'

# Time to wait, in seconds.
TICK = 0.007
