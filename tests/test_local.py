# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from streaming import MDSWriter
from streaming.base.local import LocalDataset


def test_local_dataset():
    columns = {'value': 'int'}
    num_samples = 100
    with TemporaryDirectory() as dirname:
        with MDSWriter(out=dirname, columns=columns) as out:
            for i in range(num_samples):
                out.write({'value': i})

        dataset = LocalDataset(dirname)
        for sample_id in range(num_samples):
            sample = dataset[sample_id]
            assert sample['value'] == sample_id


def test_local_dataloader():
    columns = {'value': 'int'}
    num_samples = 100
    with TemporaryDirectory() as dirname:
        with MDSWriter(out=dirname, columns=columns) as out:
            for i in range(num_samples):
                out.write({'value': i})

        dataset = LocalDataset(dirname)
        loader = DataLoader(dataset, batch_size=1)
        for sample_id, batch in enumerate(loader):
            assert batch['value'] == torch.tensor([sample_id])
