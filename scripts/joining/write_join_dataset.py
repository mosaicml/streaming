# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A sample script to write out a joinable dataset."""

import numpy as np
from streaming.base import MDSWriter
import os
from contextlib import ExitStack

class FunClassificationDataset:
    """Classification dataset with multiple different input features.

    Args:
        shape: data sample dimensions (default: (10,))
        size: number of samples (default: 100)
        num_classes: number of classes (default: 2)
    """

    def __init__(self, shape=(3,), size=100, num_classes=2):
        self.size = size
        self._id = np.arange(size)
        self.x = np.random.randn(size, *shape)
        self.y = np.random.randint(0, num_classes, size)
        self.pet = np.random.choice(['cat', 'dog', 'bird', 'fish'], size)
        self.color = np.random.choice(['red', 'green', 'blue', 'yellow'], size)

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self._id[index], self.x[index], self.y[index], self.pet[index], self.color[index]

all_columns = {'_id': 'int32', 'x': 'ndarray:float64', 'y': 'int64', 'pet': 'str', 'color': 'str'}
output_dir = 'sample_dataset'

dataset = FunClassificationDataset()

with ExitStack() as stack:
    writers = []
    for colname in list(all_columns.keys())[1:]:
        sub_dataset_columns = {'_id': all_columns['_id'], colname: all_columns[colname]}
        sub_dataset_path = os.path.join(output_dir, colname)
        writers.append(stack.enter_context(MDSWriter(out=sub_dataset_path, columns=sub_dataset_columns)))

    for sample in dataset:
        _id = sample[0]
        for writer, value, colname in zip(writers, sample[1:], list(all_columns.keys())[1:]):
            writer.write({'_id': _id, colname: value})

print(f"Joinable datasets written to folder: {output_dir}")

