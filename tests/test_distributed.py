# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
from typing import Any, Tuple
from unittest import mock

import numpy as np
import pytest
import torch.distributed as dist
from torch.utils.data import DataLoader

import streaming.base.distributed as ms_dist
from streaming.base import StreamingDataset
from tests.common.datasets import SequenceDataset, write_mds_dataset
from tests.common.distributed import DistributedTest

logger = logging.getLogger(__name__)


class TestWorldSize(DistributedTest):
    world_size = 3

    def test_class_variable(self):
        assert dist.is_initialized()
        assert dist.get_world_size() == 3
        assert dist.get_rank() < 3

    @pytest.mark.world_size(4)
    def test_parameterize(self):
        assert dist.is_initialized()
        assert dist.get_world_size() == 4
        assert dist.get_rank() < 4


class TestAllgatherObject(DistributedTest):

    @pytest.mark.world_size(2)
    @pytest.mark.parametrize(('data', 'expected_data'),
                             [(5, [5, 5]),
                              (np.array(10), [np.array(10), np.array(10)])])
    def test_all_gather_object(self, data: Any, expected_data: Any):
        output = ms_dist.all_gather_object(data)
        assert output == expected_data

    @pytest.mark.world_size(1)
    @pytest.mark.parametrize(('data', 'expected_data'), [(5, [5]), (np.array(10), [np.array(10)])])
    def test_all_gather_object_non_dist(self, data: Any, expected_data: Any):
        output = ms_dist.all_gather_object(data)
        assert output == expected_data


@mock.patch.dict(os.environ, {'WORLD_SIZE': '2'})
def test_all_gather_object_non_dist_exception():
    with pytest.raises(RuntimeError):
        _ = ms_dist.all_gather_object(5)


@pytest.mark.skip(
    'Fails due to new shared Filelock. See https://mosaicml.atlassian.net/browse/CO-1403')
class TestInit(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('batch_size', [128])
    @pytest.mark.parametrize('drop_last', [False, True])
    @pytest.mark.parametrize('num_workers', [0, 1, 8])
    @pytest.mark.parametrize('num_samples', [9867])
    @pytest.mark.parametrize('size_limit', [8_192])
    def test_dataloader_multi_device(self, remote_local: Tuple[str, str], batch_size: int,
                                     drop_last: bool, num_workers: int, num_samples: int,
                                     size_limit: int):

        global_rank = ms_dist.get_rank()
        global_num_ranks = ms_dist.get_world_size()
        node_rank = ms_dist.get_local_rank()

        assert batch_size % global_num_ranks == 0
        per_rank_batch_size = batch_size // global_num_ranks

        # Create globally shared remote, and node-local folders
        remote_local_list = list(remote_local)
        dist.broadcast_object_list(remote_local_list)
        remote, local = remote_local_list
        node_local = os.path.join(local, str(node_rank))

        dataset = SequenceDataset(num_samples)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        if global_rank == 0:
            write_mds_dataset(dirname=remote,
                              columns=columns,
                              samples=dataset,
                              size_limit=size_limit)
        dist.barrier()

        # Build a StreamingDataset
        dataset = StreamingDataset(local=node_local,
                                   remote=remote,
                                   shuffle=True,
                                   batch_size=per_rank_batch_size)

        # Build DataLoader
        dataloader = DataLoader(dataset=dataset,
                                batch_size=per_rank_batch_size,
                                num_workers=num_workers,
                                drop_last=drop_last)

        # Expected number of batches based on batch_size and drop_last
        device_compatible_num_samples = (global_num_ranks) * math.ceil(num_samples /
                                                                       (global_num_ranks))
        expected_num_batches = (device_compatible_num_samples //
                                batch_size) if drop_last else math.ceil(
                                    device_compatible_num_samples / batch_size)
        expected_num_samples = expected_num_batches * batch_size if drop_last else device_compatible_num_samples

        # Iterate over DataLoader
        rcvd_batches = 0
        sample_order = []

        for batch_ix, batch in enumerate(dataloader):
            rcvd_batches += 1

            # Every batch should be complete except (maybe) final one
            if batch_ix + 1 < expected_num_batches:
                assert len(batch['id']) == per_rank_batch_size
            else:
                if drop_last:
                    assert len(batch['id']) == per_rank_batch_size
                else:
                    assert len(batch['id']) <= per_rank_batch_size
            device_batch_ids = [int(uid) for uid in batch['id']]
            all_device_batch_ids = ms_dist.all_gather_object(device_batch_ids)

            for ids in all_device_batch_ids:
                sample_order += ids

        # Test dataloader length
        assert len(dataloader) == expected_num_batches
        assert rcvd_batches == expected_num_batches

        # Test that all samples arrived with no duplicates
        assert len(sample_order) == expected_num_samples
        if not drop_last:
            assert len(set(sample_order)) == num_samples
