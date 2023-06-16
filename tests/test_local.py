# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory

from streaming import MDSWriter
from streaming.base.local import LocalDataset


def test_local():
    columns = {'value': 'int'}
    num_samples = 100
    with TemporaryDirectory() as dirname:
        with MDSWriter(out=dirname, columns=columns) as out:
            for i in range(num_samples):
                out.write({'value': i})

        dataset = LocalDataset(dirname)
        for sample_id in range(num_samples):
            dataset[sample_id]
