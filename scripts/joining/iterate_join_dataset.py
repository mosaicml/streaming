# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A sample script to iterate over joinable datasets."""

from streaming.base.join_dataset import StreamingJoinDataset
from streaming.base import Stream
from torch.utils.data import DataLoader
import os

# Create separate streams for each sub dataset
all_columns = {'_id': 'int32', 'x': 'ndarray:float64', 'y': 'int64', 'pet': 'str', 'color': 'str'}
dataset_dir = 'sample_dataset'

streams = []
for colname in list(all_columns.keys())[1:]:
    sub_dataset_path = os.path.join(dataset_dir, colname)
    streams.append(Stream(local=sub_dataset_path))

dataset = StreamingJoinDataset(streams=streams, batch_size=1)
dataloader = DataLoader(dataset, batch_size=1)

# Inspect the dataset samples, joined on '_id'.
for i, sample in enumerate(dataloader):
    print(f'Batch {i}, sample is: {sample}')


