# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
import shutil
from multiprocessing import Process
from typing import Any, Callable, Tuple

import pytest
from torch.utils.data import DataLoader

from streaming.base import Stream, StreamingDataLoader, StreamingDataset
from streaming.base.batching import generate_work
from streaming.base.util import clean_stale_shared_memory
from streaming.base.world import World
from tests.common.utils import convert_to_mds


@pytest.mark.usefixtures('local_remote_dir')
def test_tiny_dataset_exception(local_remote_dir: Tuple[str, str]):

    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=1,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(
        local=local_dir,
        remote=remote_dir,
        shuffle=True,
        num_canonical_nodes=2,
        batch_size=1,
    )

    with pytest.raises(ValueError, match=f'The number of samples assigned to a canonical node*'):
        # When we iterate through the dataset, we should throw an error because
        # the number of samples is greater than num_canonical_nodes.
        for _ in dataset:
            pass


@pytest.mark.usefixtures('local_remote_dir')
def test_no_batch_size_exception(local_remote_dir: Tuple[str, str]):

    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=200,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir, remote=remote_dir)
    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset, batch_size=2, num_workers=2, drop_last=True)

    with pytest.raises(ValueError, match=f'Please pass `batch_size` to StreamingDataset*'):
        # When we iterate through the dataloader, we should throw an error because
        # we have not passed in batch size to the StreamingDataset. Instantiation of
        # StreamingDataset is still fine though.
        for _ in dataloader:
            pass


@pytest.mark.usefixtures('local_remote_dir')
def test_new_defaults_warning(local_remote_dir: Tuple[str, str], caplog: Callable):
    caplog.set_level(logging.WARNING)
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=100,
                   size_limit=1 << 8)

    # Build a StreamingDataset with new defaults. Should warn about the new defaults changes.
    dataset = StreamingDataset(local=local, remote=remote, shuffle=True, batch_size=4)
    dataloader = StreamingDataLoader(dataset=dataset, batch_size=4)
    for _ in dataloader:
        pass

    assert 'Because `predownload` was not specified,' in caplog.text
    assert 'Because `shuffle_block_size` was not specified,' in caplog.text
    assert 'Because `num_canonical_nodes` was not specified,' in caplog.text


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [8])
@pytest.mark.parametrize('epoch_size', [10, 200])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_epoch_size_no_streams(local_remote_dir: Tuple[str,
                                                                  str], batch_size: int, seed: int,
                                          shuffle: bool, drop_last: bool, num_workers: int,
                                          num_canonical_nodes: int, epoch_size: int):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local,
                               remote=remote,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               epoch_size=epoch_size)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=drop_last)

    samples_seen = 0
    for batch in dataloader:
        samples_seen += batch['sample'].size(dim=0)

    if epoch_size % num_canonical_nodes != 0:
        assert samples_seen == math.ceil(epoch_size / num_canonical_nodes) * num_canonical_nodes
    else:
        if drop_last:
            assert samples_seen == epoch_size - (epoch_size % batch_size)
        else:
            assert samples_seen == epoch_size


@pytest.mark.parametrize('batch_size', [4, 7])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [8])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_per_stream_batching(local_remote_dir: Tuple[str, str], batch_size: int,
                                        seed: int, shuffle: bool, drop_last: bool,
                                        num_workers: int, num_canonical_nodes: int):
    # create mock datasets for 2 streams. Second one has 1.5x the samples
    local, remote = local_remote_dir
    local1 = os.path.join(local, 'stream1')
    local2 = os.path.join(local, 'stream2')
    remote1 = os.path.join(remote, 'stream1')
    remote2 = os.path.join(remote, 'stream2')

    # stream 1 has samples 0->600
    convert_to_mds(out_root=remote1,
                   dataset_name='sequencedataset',
                   num_samples=200,
                   size_limit=1 << 8)
    # stream 2 has samples 600 and above.
    # This lets us differentiate between the samples from each stream
    convert_to_mds(out_root=remote2,
                   dataset_name='sequencedataset',
                   num_samples=300,
                   offset=600,
                   size_limit=1 << 8)

    stream1 = Stream(local=local1, remote=remote1)
    stream2 = Stream(local=local2, remote=remote2)

    # Build StreamingDataset
    dataset = StreamingDataset(streams=[stream1, stream2],
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               batching_method='per_stream')

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=drop_last)

    # Ensure that the samples seen in each batch are from the same stream.
    # Stream 1 has samples 0 to 600, stream 2 has samples 600 and above.
    # When we have 1 batch and 1 device, we will repeat samples until we are divisible by NCN.
    # This logic takes care of that case.
    total_batches_stream_1 = 200 // batch_size if 200 % num_canonical_nodes == 0 else (
        200 + (num_canonical_nodes - 200 % num_canonical_nodes)) // batch_size
    total_batches_stream_2 = 300 // batch_size if 300 % num_canonical_nodes == 0 else (
        300 + (num_canonical_nodes - 300 % num_canonical_nodes)) // batch_size
    total_batches = total_batches_stream_1 + total_batches_stream_2

    batches_seen = 0
    for batch in dataloader:
        batches_seen += 1
        samples = batch['sample']
        first_sample = samples[0]
        if first_sample >= 600:
            assert (samples >= 600).all()
        else:
            assert (samples < 600).all()

    # Make sure that we see the expected number of batches, accounting for sample drops
    assert batches_seen == total_batches


@pytest.mark.parametrize('batch_size', [4, 7])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('physical_nodes', [2, 8])
@pytest.mark.parametrize('ranks_per_node', [4, 8])
@pytest.mark.parametrize('workers_per_rank', [4, 8])
@pytest.mark.parametrize('num_canonical_nodes', [8])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_device_per_stream_batching(local_remote_dir: Tuple[str, str], batch_size: int,
                                               seed: int, shuffle: bool, physical_nodes: int,
                                               ranks_per_node: int, workers_per_rank: int,
                                               num_canonical_nodes: int):
    # create mock datasets for 2 streams. Second one has 1.5x the samples
    local, remote = local_remote_dir
    local1 = os.path.join(local, 'stream1')
    local2 = os.path.join(local, 'stream2')
    remote1 = os.path.join(remote, 'stream1')
    remote2 = os.path.join(remote, 'stream2')

    # stream 1 has samples 0->600, sample ids 0->200
    convert_to_mds(out_root=remote1,
                   dataset_name='sequencedataset',
                   num_samples=200,
                   size_limit=1 << 8)
    # stream 2 has samples 600 and above, sample ids 200 and above.
    # This lets us differentiate between the samples from each stream
    convert_to_mds(out_root=remote2,
                   dataset_name='sequencedataset',
                   num_samples=300,
                   offset=600,
                   size_limit=1 << 8)

    stream1 = Stream(local=local1, remote=remote1)
    stream2 = Stream(local=local2, remote=remote2)

    # Build StreamingDataset
    dataset = StreamingDataset(streams=[stream1, stream2],
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               batching_method='per_stream',
                               shuffle_block_size=100)

    # Get sample partition
    fake_world = World(
        num_nodes=physical_nodes,
        ranks_per_node=ranks_per_node,
        workers_per_rank=workers_per_rank,
        worker=0,
    )
    sample_partition = generate_work(batching_method='device_per_stream',
                                     dataset=dataset,
                                     world=fake_world,
                                     epoch=0,
                                     sample_in_epoch=0)

    # Partition shape should be:
    # (physical nodes, ranks per node, workers per rank, batches per worker, batch size)
    assert sample_partition.shape[0] == physical_nodes
    assert sample_partition.shape[1] == ranks_per_node
    assert sample_partition.shape[2] == workers_per_rank
    assert sample_partition.shape[4] == batch_size

    # Transpose and reshape sample partition to get device batches, in training traversal order.
    sample_partition = sample_partition.transpose(3, 2, 0, 1, 4).flatten().reshape(-1, batch_size)

    for device_batch in sample_partition:
        if device_batch[0] < 200:
            # Ensure all samples are from stream 1
            assert (device_batch < 200).all()
        else:
            # Ensure all samples are from stream 2
            assert (device_batch >= 200).all()


@pytest.mark.parametrize('batch_size', [4, 7])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('drop_last', [True])
@pytest.mark.parametrize('num_workers', [4])
@pytest.mark.parametrize('num_canonical_nodes', [8])
@pytest.mark.parametrize('num_stream_1_samples', [200, 255])
@pytest.mark.parametrize('num_stream_2_samples', [342, 557])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_stratified_batching(local_remote_dir: Tuple[str, str], batch_size: int,
                                        seed: int, shuffle: bool, drop_last: bool,
                                        num_workers: int, num_canonical_nodes: int,
                                        num_stream_1_samples: int, num_stream_2_samples: int):
    # create mock datasets for 2 streams. Second one has 1.5x the samples
    local, remote = local_remote_dir
    local1 = os.path.join(local, 'stream1')
    local2 = os.path.join(local, 'stream2')
    remote1 = os.path.join(remote, 'stream1')
    remote2 = os.path.join(remote, 'stream2')

    # stream 1 has samples 0->num_stream_1_samples*3
    convert_to_mds(out_root=remote1,
                   dataset_name='sequencedataset',
                   num_samples=num_stream_1_samples,
                   size_limit=1 << 8)
    # stream 2 has samples num_stream_1_samples*3 and above.
    # This lets us differentiate between the samples from each stream
    convert_to_mds(out_root=remote2,
                   dataset_name='sequencedataset',
                   num_samples=num_stream_2_samples,
                   offset=num_stream_1_samples * 3,
                   size_limit=1 << 8)

    stream1 = Stream(local=local1, remote=remote1)
    stream2 = Stream(local=local2, remote=remote2)

    # Build StreamingDataset
    dataset = StreamingDataset(streams=[stream1, stream2],
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               batching_method='stratified')

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=drop_last)

    # Ensure that the samples seen in each batch are proportional to the stream sizes.
    total_samples = num_stream_1_samples + num_stream_2_samples
    stream_1_batch_part = round(batch_size * (num_stream_1_samples / total_samples))
    stream_2_batch_part = batch_size - stream_1_batch_part

    # The total number of possible batches is the minimum of the batch parts from each stream.
    # Total number of samples will be padded to be divisible by NCN.
    total_stream_1_batches = num_stream_1_samples // stream_1_batch_part \
        if num_stream_1_samples % num_canonical_nodes == 0 else (
        num_stream_1_samples +
        (num_canonical_nodes - num_stream_1_samples % num_canonical_nodes)) // stream_1_batch_part
    total_stream_2_batches = num_stream_2_samples // stream_2_batch_part \
        if num_stream_2_samples % num_canonical_nodes == 0 else (
        num_stream_2_samples +
        (num_canonical_nodes - num_stream_2_samples % num_canonical_nodes)) // stream_2_batch_part
    total_batches = min(total_stream_1_batches, total_stream_2_batches)
    batches_seen = 0
    for batch in dataloader:
        batches_seen += 1
        samples = batch['sample']
        # Check if constructed batch is the correct size
        assert len(samples) == batch_size
        stream_1_samples = 0
        stream_2_samples = 0
        for sample in samples:
            # stream 1 goes until num_stream_1_samples*3, stream 2 is everything after
            if sample < num_stream_1_samples * 3:
                stream_1_samples += 1
            else:
                stream_2_samples += 1
        # check that the batch is consistently composed of the correct number of samples
        # from each stream
        assert stream_1_samples == stream_1_batch_part
        assert stream_2_samples == stream_2_batch_part

    # Check if the number of batches seen is correct
    assert batches_seen == total_batches


@pytest.mark.parametrize('batch_size', [4, 7])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('drop_last', [True])
@pytest.mark.parametrize('num_workers', [4])
@pytest.mark.parametrize('num_canonical_nodes', [8])
@pytest.mark.parametrize('stream_1_proportion', [2, 5])
@pytest.mark.parametrize('stream_2_proportion', [2, 5])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_stratified_batching_user_set(local_remote_dir: Tuple[str,
                                                                         str], batch_size: int,
                                                 seed: int, shuffle: bool, drop_last: bool,
                                                 num_workers: int, num_canonical_nodes: int,
                                                 stream_1_proportion: int,
                                                 stream_2_proportion: int):
    # create mock datasets for 2 streams. Second one has 1.5x the samples
    local, remote = local_remote_dir
    local1 = os.path.join(local, 'stream1')
    local2 = os.path.join(local, 'stream2')
    remote1 = os.path.join(remote, 'stream1')
    remote2 = os.path.join(remote, 'stream2')

    # stream 1 has samples 0->600
    convert_to_mds(out_root=remote1,
                   dataset_name='sequencedataset',
                   num_samples=200,
                   size_limit=1 << 8)
    # stream 2 has samples 600 and above.
    # This lets us differentiate between the samples from each stream
    convert_to_mds(out_root=remote2,
                   dataset_name='sequencedataset',
                   num_samples=300,
                   offset=600,
                   size_limit=1 << 8)

    stream1 = Stream(local=local1, remote=remote1, proportion=stream_1_proportion)
    stream2 = Stream(local=local2, remote=remote2, proportion=stream_2_proportion)

    # Build StreamingDataset
    dataset = StreamingDataset(streams=[stream1, stream2],
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               batching_method='stratified')

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=drop_last)

    # Ensure that the samples seen in each batch match what the user set.
    total_proportions = stream_1_proportion + stream_2_proportion
    stream_2_batch_part = round((stream_2_proportion / total_proportions) * batch_size)
    stream_1_batch_part = batch_size - stream_2_batch_part
    for batch in dataloader:
        samples = batch['sample']
        # Check if constructed batch is the correct size
        assert len(samples) == batch_size
        stream_1_samples = 0
        stream_2_samples = 0
        for sample in samples:
            # stream 1 goes until 600, stream 2 is everything after
            if sample < 600:
                stream_1_samples += 1
            else:
                stream_2_samples += 1
        # check that the batch is consistently composed of the correct number of samples
        # from each stream
        assert stream_1_samples == stream_1_batch_part
        assert stream_2_samples == stream_2_batch_part


@pytest.mark.parametrize('stream_2_size', list(range(1, 65, 10)))
@pytest.mark.usefixtures('local_remote_dir')
def test_stratified_batching_Exception(local_remote_dir: Tuple[str, str], stream_2_size: int):

    local, remote = local_remote_dir
    local1 = os.path.join(local, 'stream1')
    local2 = os.path.join(local, 'stream2')
    remote1 = os.path.join(remote, 'stream1')
    remote2 = os.path.join(remote, 'stream2')

    # With a batch size of 8, stream 1 of size 1000, and stream 2 anywhere between 1 and 65,
    # We expect stream 2 to be too small to be included in each batch,
    # which should raise ValueError.
    stream_1_size = 1000
    batch_size = 8

    # Make stream 1 with stream_1_size samples
    convert_to_mds(out_root=remote1,
                   dataset_name='sequencedataset',
                   num_samples=stream_1_size,
                   size_limit=1 << 8)
    # Make stream 2 with stream_2_size samples
    convert_to_mds(out_root=remote2,
                   dataset_name='sequencedataset',
                   num_samples=stream_2_size,
                   offset=stream_1_size * 3,
                   size_limit=1 << 8)

    stream1 = Stream(local=local1, remote=remote1)
    stream2 = Stream(local=local2, remote=remote2)
    dataset = StreamingDataset(streams=[stream1, stream2],
                               batch_size=batch_size,
                               batching_method='stratified')

    dataloader = StreamingDataLoader(dataset=dataset, batch_size=batch_size, drop_last=False)

    with pytest.raises(ValueError, match=f'Number of samples for stream*'):
        # When we iterate through the dataloader, the samples will be partitioned.
        # This should thow ValueError since stream 2 is too small to be included in each batch.
        for _ in dataloader:
            continue


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [8])
@pytest.mark.parametrize('epoch_size', [10, 200])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_epoch_size_multiple_streams_default(local_remote_dir: Tuple[str, str],
                                                        batch_size: int, seed: int, shuffle: bool,
                                                        drop_last: bool, num_workers: int,
                                                        num_canonical_nodes: int, epoch_size: int):
    # create mock datasets for 2 streams. Second one has 1.5x the samples
    local, remote = local_remote_dir
    local1 = os.path.join(local, 'stream1')
    local2 = os.path.join(local, 'stream2')
    remote1 = os.path.join(remote, 'stream1')
    remote2 = os.path.join(remote, 'stream2')

    # stream 1 has samples 0->600
    convert_to_mds(out_root=remote1,
                   dataset_name='sequencedataset',
                   num_samples=200,
                   size_limit=1 << 8)
    # stream 2 has samples 600 and above.
    # This lets us differentiate between the samples from each stream
    convert_to_mds(out_root=remote2,
                   dataset_name='sequencedataset',
                   num_samples=300,
                   offset=600,
                   size_limit=1 << 8)

    stream1 = Stream(local=local1, remote=remote1)
    stream2 = Stream(local=local2, remote=remote2)

    # Build StreamingDataset
    dataset = StreamingDataset(streams=[stream1, stream2],
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               epoch_size=epoch_size)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=drop_last)

    # track the number of samples seen overall in the epoch,
    # and also track the number of samples seen from each stream.
    # we expect the number of samples from each stream in the epoch
    # to be proportional to the number of total samples in the stream,
    # in the case when proportion, repeat, and choose are all unspecified.
    samples_seen_stream1 = 0
    samples_seen_stream2 = 0
    samples_seen = 0
    for batch in dataloader:
        samples = batch['sample']
        samples_seen += samples.size(dim=0)
        stream1_seen = (samples < 600).sum().item()
        stream2_seen = (samples > 600).sum().item()
        samples_seen_stream1 += stream1_seen
        samples_seen_stream2 += stream2_seen

    # if epoch size is not divisible by canonical nodes the partition algorithm will have
    # some repeated samples. This means the number of samples seen will be within some
    # tolerance of the epoch size. In all cases though, stream 1 and stream 2 samples
    # should be approximately in a 2:3 ratio, in accordance with the number of samples
    # each stream has (stream 1: 200, stream 2: 300).
    if epoch_size % num_canonical_nodes != 0:
        assert samples_seen == (math.ceil(epoch_size / num_canonical_nodes) * num_canonical_nodes)
        assert samples_seen_stream1 == int(
            samples_seen * 0.4) or samples_seen_stream1 == int(samples_seen * 0.4) + 1
        assert samples_seen_stream2 == int(
            samples_seen * 0.6) or samples_seen_stream2 == int(samples_seen * 0.6) + 1
    else:
        # if drop_last is True, we will drop incomplete batches, so samples_seen can
        # be less than epoch_size
        if drop_last:
            assert samples_seen == epoch_size - (epoch_size % batch_size)
            assert samples_seen_stream1 == int(
                samples_seen * 0.4) or samples_seen_stream1 == int(samples_seen * 0.4) + 1
            assert samples_seen_stream2 == int(
                samples_seen * 0.6) or samples_seen_stream2 == int(samples_seen * 0.6) + 1
        # drop_last is false, and epoch_size is divisible by num_canonical_nodes, so samples_seen
        # should be the same as epoch_size
        else:
            assert samples_seen == epoch_size
            assert samples_seen_stream1 == int(
                samples_seen * 0.4) or samples_seen_stream1 == int(samples_seen * 0.4) + 1
            assert samples_seen_stream2 == int(
                samples_seen * 0.6) or samples_seen_stream2 == int(samples_seen * 0.6) + 1


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('drop_last', [False])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [4, 8])
@pytest.mark.parametrize('epoch_size', [16, 200])
@pytest.mark.parametrize('sampling_method', ['fixed', 'balanced'])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_fixed_balanced_sampling(local_remote_dir: Any, batch_size: int, seed: int,
                                            shuffle: bool, drop_last: bool, num_workers: int,
                                            num_canonical_nodes: int, epoch_size: int,
                                            sampling_method: str):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               epoch_size=epoch_size,
                               sampling_method=sampling_method)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=drop_last)

    # check 2 more epochs to see if samples are the same
    first_samples_seen = {}
    for epoch in range(3):
        samples_seen = first_samples_seen if epoch == 0 else {}
        for batch in dataloader:
            for sample_id in batch['id']:
                if sample_id in samples_seen:
                    samples_seen[sample_id] += 1
                else:
                    samples_seen[sample_id] = 1

        if epoch > 0 and sampling_method == 'fixed':
            assert samples_seen == first_samples_seen
        if epoch > 0 and sampling_method == 'balanced':
            assert samples_seen != first_samples_seen


@pytest.mark.parametrize('num_samples', [9867])
@pytest.mark.parametrize('seed', [1234])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_mid_epoch_exit(local_remote_dir: Tuple[str, str], num_samples: int, seed: int):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote, dataset_name='sequencedataset', num_samples=num_samples)

    def run_one_iter(local: str, remote: str, seed: int) -> None:
        # Build a StreamingDataset
        dataset = StreamingDataset(local=local, remote=remote, shuffle_seed=seed, batch_size=1)

        # Do one iteration
        it = iter(dataset)
        next(it)
        # Test if we can exit...

    p = Process(target=run_one_iter, args=(local, remote, seed))
    p.start()
    p.join(5)
    p.terminate()

    result = p.exitcode

    assert result == 0


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('size_limit', [256, 512])
@pytest.mark.parametrize('seed', [1111])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('shuffle_block_size', [50, 100])
@pytest.mark.usefixtures('local_remote_dir')
def test_py1e_shuffle_block_warning(local_remote_dir: Any, batch_size: int, size_limit: int,
                                    seed: int, shuffle: bool, shuffle_block_size: int):
    remote_dir, local_dir = local_remote_dir
    # Here, size_limit is in bytes. Each SequenceDataset sample is around 10 bytes, but the header
    # will also take up some space.
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=1000,
                   size_limit=(size_limit * 10) + 1000)

    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               num_canonical_nodes=1,
                               shuffle_seed=seed,
                               shuffle_algo='py1e',
                               shuffle_block_size=shuffle_block_size)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    with pytest.warns(UserWarning, match=f'Shuffle block size was smaller than shard size*'):
        for _ in dataloader:
            pass


@pytest.mark.parametrize('batch_size', [128])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('num_workers', [0, 4])
@pytest.mark.parametrize('num_samples', [9867, 30_000])
@pytest.mark.parametrize('size_limit', [8_192])
@pytest.mark.parametrize('seed', [1234])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_single_device(local_remote_dir: Tuple[str, str], batch_size: int,
                                  drop_last: bool, shuffle: bool, num_workers: int,
                                  num_samples: int, size_limit: int, seed: int):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=num_samples,
                   size_limit=size_limit)

    # Build a StreamingDataset
    dataset = StreamingDataset(local=local,
                               remote=remote,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=drop_last)

    # Expected number of batches based on batch_size and drop_last
    expected_num_batches = (num_samples // batch_size) if drop_last else math.ceil(num_samples /
                                                                                   batch_size)
    expected_num_samples = expected_num_batches * batch_size if drop_last else num_samples

    # Iterate over DataLoader
    rcvd_batches = 0
    sample_order = []

    for batch_ix, batch in enumerate(dataloader):
        rcvd_batches += 1

        # Every batch should be complete except (maybe) final one
        if batch_ix + 1 < expected_num_batches:
            assert len(batch['id']) == batch_size
        else:
            if drop_last:
                assert len(batch['id']) == batch_size
            else:
                assert len(batch['id']) <= batch_size

        sample_order.extend(batch['id'][:])

    # Test dataloader length
    assert len(dataloader) == expected_num_batches
    assert rcvd_batches == expected_num_batches

    # Test that all samples arrived with no duplicates
    assert len(set(sample_order)) == expected_num_samples
    if not drop_last:
        assert len(set(sample_order)) == num_samples


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [1111])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('sampling_method', ['balanfixed', 'fixedd', '', 'random', 'ayo'])
@pytest.mark.usefixtures('local_remote_dir')
def test_sampling_method_invalid_Exception(local_remote_dir: Any, batch_size: int, seed: int,
                                           shuffle: bool, sampling_method: str):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    with pytest.raises(ValueError, match=f'Invalid sampling method:*'):
        _ = StreamingDataset(local=local_dir,
                             remote=remote_dir,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             shuffle_seed=seed,
                             sampling_method=sampling_method)


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seed', [1111])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('num_workers', [0, 8])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_determinism(local_remote_dir: Any, batch_size: int, seed: int, shuffle: bool,
                                num_workers: int):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    # Append sample ID
    sample_order = []
    for batch in dataloader:
        sample_order.extend(batch['id'][:])

    del dataloader
    del dataset

    # Build StreamingDataset again to test deterministic sample ID
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    # Append sample ID
    second_sample_order = []
    for batch in dataloader:
        second_sample_order.extend(batch['id'][:])

    assert len(sample_order) == len(second_sample_order)
    assert sample_order == second_sample_order


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [0, 8])
@pytest.mark.parametrize('num_canonical_nodes', [1])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_sample_order(local_remote_dir: Any, batch_size: int, seed: int, shuffle: bool,
                                 drop_last: bool, num_workers: int, num_canonical_nodes: int):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=drop_last)

    if drop_last:
        num_samples = (len(dataset) // batch_size) * batch_size
        expected_sample_order = [f'{value:06}' for value in range(num_samples)]
    else:
        expected_sample_order = [f'{value:06}' for value in range(len(dataset))]

    # Append sample ID
    sample_order = []
    for batch in dataloader:
        sample_order.extend(batch['id'][:])

    assert expected_sample_order == sample_order


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seed', [3456])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('num_workers', [0, 4])
@pytest.mark.parametrize('num_canonical_nodes', [1])
@pytest.mark.usefixtures('local_remote_dir')
def test_streamingdataloader_mid_epoch_resumption(local_remote_dir: Any, batch_size: int,
                                                  seed: int, shuffle: bool, num_workers: int,
                                                  num_canonical_nodes: int):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=False)

    expected_sample_order = [f'{ix:06}' for ix in range(len(dataset))]

    sample_order = []
    for idx, batch in enumerate(dataloader):
        if idx == len(dataset) // (batch_size * 2):
            sample_order.extend(batch['id'][:])
            state_dict = dataloader.state_dict()
            assert state_dict is not None
            break
        sample_order.extend(batch['id'][:])

    del dataloader
    del dataset

    clean_stale_shared_memory()

    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=False)

    dataloader.load_state_dict(state_dict)  # pyright: ignore
    for idx, batch in enumerate(dataloader):
        sample_order.extend(batch['id'][:])

    # sort the sample to check for missing and duplicate samples
    sample_order.sort()
    assert len(sample_order) == len(expected_sample_order), 'Missing samples'
    assert len(set(sample_order)) == len(set(expected_sample_order)), 'Duplicate samples'
    assert sample_order == expected_sample_order, 'Incorrect sample order'


@pytest.mark.parametrize('shuffle_seed', [(9876, 9876), (12345, 1567)])
@pytest.mark.usefixtures('local_remote_dir')
def test_multiple_dataset_instantiation(local_remote_dir: Any, shuffle_seed: tuple):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)
    batch_size = 8
    num_workers = 2

    shuffle_seed_train, shuffle_seed_val = shuffle_seed

    # Build train StreamingDataset
    train_dataset = StreamingDataset(local=local_dir,
                                     remote=remote_dir,
                                     batch_size=batch_size,
                                     shuffle_seed=shuffle_seed_train)

    # Build train DataLoader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

    # Append train sample ID
    train_sample_order = []
    for batch in train_dataloader:
        train_sample_order.extend(batch['id'][:])

    shutil.rmtree(local_dir, ignore_errors=True)
    del train_dataloader
    del train_dataset

    # Build val StreamingDataset
    val_dataset = StreamingDataset(local=local_dir,
                                   remote=remote_dir,
                                   batch_size=batch_size,
                                   shuffle_seed=shuffle_seed_val)

    # Build val DataLoader
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers)

    # Append val sample ID
    val_sample_order = []
    for batch in val_dataloader:
        val_sample_order.extend(batch['id'][:])

    assert len(train_sample_order) == len(val_sample_order), 'Missing samples'
    assert len(set(train_sample_order)) == len(set(val_sample_order)), 'Duplicate samples'


def test_same_local_no_remote(local_remote_dir: Tuple[str, str]):
    local_0, _ = local_remote_dir
    convert_to_mds(out_root=local_0,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)
    # Build StreamingDataset
    dataset_0 = StreamingDataset(local=local_0, remote=None, batch_size=4, num_canonical_nodes=1)
    # Build StreamingDataset
    dataset_1 = StreamingDataset(local=local_0, remote=None, batch_size=2, num_canonical_nodes=1)
    samples_seen_0 = set()
    for sample in dataset_0:
        samples_seen_0.add(sample['sample'])

    samples_seen_1 = set()
    for sample in dataset_1:
        samples_seen_1.add(sample['sample'])

    assert samples_seen_0 == samples_seen_1
    assert len(samples_seen_0) == len(samples_seen_1) == 117


def test_same_local_diff_remote(local_remote_dir: Tuple[str, str]):
    local_0, remote_0 = local_remote_dir
    _, remote_1 = local_remote_dir
    convert_to_mds(out_root=local_0,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)
    # Build StreamingDataset
    _ = StreamingDataset(local=local_0, remote=remote_0, batch_size=4, num_canonical_nodes=1)
    # Build StreamingDataset
    with pytest.raises(ValueError, match='Reused local directory.*vs.*Provide a different one.'):
        _ = StreamingDataset(local=local_0, remote=remote_1, batch_size=2, num_canonical_nodes=1)
