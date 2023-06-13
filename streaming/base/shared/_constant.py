# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Constants."""

# Shared Memory names
LOCALS = '_locals'
BARRIER = '_barrier'
NEXT_EPOCH = '_next_epoch'
CACHE_USAGE = '_cache_usage'
SHARD_STATES = '_shard_states'
SHARD_ACCESS_TIMES = '_shard_access_times'
RESUME = '_resume'
EPOCH_SHAPE = '_epoch_shape'
EPOCH_DATA = '_epoch_data'

# filelock names
BARRIER_FILELOCK = '_barrier_filelock'
CACHE_FILELOCK = '_cache_filelock'
