# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import math
import os
import shutil
from typing import Any, Tuple

import pytest
from torch.utils.data import DataLoader

from streaming.base import Stream, StreamingDataLoader, StreamingDataset
from tests.common.utils import convert_to_mds


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [4, 8])
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
    dataloader = StreamingDataLoader(dataset=dataset,
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


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [4, 8])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_uniform_global_batch_sampling(local_remote_dir: Tuple[str,
                                                                          str], batch_size: int,
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
    # stream 2 has samples 600 and above. This lets us differentiate between the samples from each stream
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
                               sampling_method='uniform_global_batch')

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=drop_last)

    # Ensure that the samples seen in each batch are from the same stream.
    # Stream 1 has samples 0 to 600, stream 2 has samples 600 and above.
    for batch in dataloader:
        samples = batch['sample']
        first_sample = samples[0]
        print(first_sample)
        print(samples)
        if first_sample >= 600:
            assert (samples >= 600).all()
        else:
            assert (samples < 600).all()


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [4, 8])
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
    # stream 2 has samples 600 and above. This lets us differentiate between the samples from each stream
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

    # if epoch size is not divisible by canonical nodes the partition algorithm will have some repeated samples
    # so the number of samples seen will be within some tolerance of the epoch size
    # in all cases though, stream 1 and stream 2 samples should be approximately in a 2:3 ratio
    # in accordance with the number of samples each stream has (stream 1: 200, stream 2: 300)
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
