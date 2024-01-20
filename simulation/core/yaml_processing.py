# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Ingest yaml and create SimulationDataset."""

from typing import Optional, Tuple

from core.sim_dataset import SimulationDataset
from core.sim_time import Time, TimeUnit, ensure_time
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from streaming.base import Stream


def ingest_yaml(yaml_dict: Optional[dict] = None,
                filepath: Optional[str] = None) -> Tuple[Optional[int], int, Time, int, dict]:
    """Create SimulationDataset from yaml file and other needed args.

    Args:
        yaml_dict (Optional[dict]): yaml file already converted to a dictionary
        filepath (Optional[str]): path to yaml file

    Returns:
        Tuple[Optional[int], Optional[int], Time, Optional[int], Optional[dict]]: total_devices,
            workers, max_duration, global_batch_size, train_dataset parameters from yaml
    """
    config = None
    # Read in the yaml file
    if filepath is not None:
        with open(filepath) as f:
            config = om.load(f)
            if not isinstance(config, DictConfig):
                raise ValueError('Yaml file must be a dictionary, not a list.')
    elif yaml_dict is not None:
        config = om.create(yaml_dict)
    else:
        raise ValueError('Must specify either filepath or yaml_dict.')

    # Get the number of devices (GPUs)
    if 'compute' in config and 'gpus' in config['compute']:
        total_devices = int(config['compute']['gpus'])
    else:
        total_devices = None

    workers = None
    train_dataset = None
    max_duration = None
    global_batch_size = None
    # Get the training and dataset params
    if 'parameters' in config:
        config = om.create(config['parameters'])

    om.resolve(config)

    if not isinstance(config, DictConfig):
        raise TypeError(f'`config` must be of type DictConfig. Got type {type(config)}.')

    # get global batch size
    if 'global_train_batch_size' in config:
        global_batch_size = config['global_train_batch_size']
    elif 'batch_size' in config:
        global_batch_size = config['batch_size']
    elif 'dataset' in config and 'train_batch_size' in config['dataset']:
        global_batch_size = config['dataset']['train_batch_size']

    # get number of workers and training dataset params
    if 'train_loader' in config:
        train_loader = config['train_loader']
        if 'num_workers' in train_loader:
            workers = train_loader['num_workers']
        else:
            workers = 1
        if 'dataset' in train_loader:
            train_dataset = train_loader['dataset']
        else:
            raise ValueError('dataset must be specified in the yaml file.')
    elif 'dataset' in config:
        dataset = config['dataset']
        if 'train_dataset' in dataset:
            train_dataset = dataset['train_dataset']
            if 'streaming_kwargs' in train_dataset:
                # Merge streaming kwargs, if present, into train_dataset
                train_dataset.update(train_dataset['streaming_kwargs'])
            if 'dataloader_kwargs' in train_dataset and 'num_workers' in train_dataset[
                    'dataloader_kwargs']:
                workers = train_dataset['dataloader_kwargs']['num_workers']
            else:
                workers = 1
        else:
            raise ValueError('train_dataset must be specified in the yaml file.')
    elif 'train_dataset' in config:
        train_dataset = config['train_dataset']
        if 'streaming_kwargs' in train_dataset:
            # Merge streaming kwargs, if present, into train_dataset
            train_dataset.update(train_dataset['streaming_kwargs'])
        if 'dataloader_kwargs' in train_dataset and 'num_workers' in train_dataset[
                'dataloader_kwargs']:
            workers = train_dataset['dataloader_kwargs']['num_workers']
        else:
            workers = 1
    else:
        raise ValueError('train_loader or dataset must be specified in the yaml file.')

    # Get duration of training from config
    if 'max_duration' in config:
        max_duration = config['max_duration']
    elif 'trainer' in config and 'max_duration' in config['trainer']:
        max_duration = config['trainer']['max_duration']
    else:
        raise ValueError('max_duration must be specified in the yaml file.')

    # convert max_duration to epochs or batches.
    max_duration = ensure_time(max_duration, TimeUnit.EPOCH)
    time_unit = max_duration.unit
    if time_unit != TimeUnit.EPOCH and time_unit != TimeUnit.BATCH:
        raise ValueError('Simulator currently only supports max_duration in epochs or batches.')

    # convert train_dataset to dict, if it isn't already
    if isinstance(train_dataset, DictConfig):
        train_dataset = om.to_container(train_dataset)

    if not isinstance(workers, int):
        raise ValueError(f'`workers` must be an int. Instead, got {type(workers)}.')
    if not isinstance(global_batch_size, int):
        raise ValueError(f'`global_batch_size` must be an int. Instead, got ' +
                         f'{type(global_batch_size)}.')
    if not isinstance(train_dataset, dict):
        raise ValueError(f'`train_dataset` must be a dict. Instead, got ' +
                         f'{type(train_dataset)}.')

    return total_devices, workers, max_duration, global_batch_size, train_dataset


def create_simulation_dataset(nodes: int, devices: int, workers: int, global_batch_size: int,
                              train_dataset: dict) -> SimulationDataset:
    """Create SimulationDataset from input information.

    Args:
        nodes (int): number of physical nodes
        devices (int): number of devices per node
        workers (int): number of workers per device
        global_batch_size (int): global batch size (samples)
        train_dataset (dict): train_dataset parameters from yaml file

    Returns:
        SimulationDataset: SimulationDataset created from input information.
    """
    streams = None
    # Check for cases where local and remote may be lists and turn those into streams.
    if 'local' in train_dataset and 'remote' in train_dataset:
        if isinstance(train_dataset['local'], list) \
            and isinstance(train_dataset['remote'], list):
            if len(train_dataset['local']) != len(train_dataset['remote']):
                raise ValueError('local and remote must be the same length in the yaml file.')
            streams = []
            for local, remote in zip(train_dataset['local'], train_dataset['remote']):
                streams.append(Stream(local=local, remote=remote, split=train_dataset['split'] \
                                      if 'split' in train_dataset else None))
            del train_dataset['local']
            del train_dataset['remote']

    # Don't re-retrieve streams if we have built it from local and remote lists.
    if not isinstance(streams, list):
        streams_dict = train_dataset.get('streams', None)
        if streams_dict is not None:
            streams = []
            if not isinstance(streams_dict, dict):
                raise TypeError(f'`streams` must be of type dict, if not a list. ' +
                                f'Got type {type(streams_dict)}.')
            for stream in streams_dict.values():
                if 'path' in stream:
                    del stream['path']
                # Create Stream object from each dictionary entry
                streams.append(Stream(**stream))

    remote = train_dataset.get('remote', None)
    local = train_dataset.get('local', None)
    split = train_dataset.get('split', None)
    download_retry = train_dataset.get('download_retry', 2)
    download_timeout = train_dataset.get('download_timeout', 60)
    validate_hash = train_dataset.get('validate_hash', None)
    keep_zip = train_dataset.get('keep_zip', False)
    epoch_size = train_dataset.get('epoch_size', None)
    predownload = train_dataset.get('predownload', None)
    cache_limit = train_dataset.get('cache_limit', None)
    partition_algo = train_dataset.get('partition_algo', 'relaxed')
    num_canonical_nodes = train_dataset.get('num_canonical_nodes', None)
    if global_batch_size % (devices * nodes) != 0:
        raise ValueError('global_batch_size must be divisible by total number of devices.')
    batch_size = global_batch_size // (devices * nodes)
    shuffle = train_dataset.get('shuffle', False)
    shuffle_algo = train_dataset.get('shuffle_algo', 'py1e')
    shuffle_seed = train_dataset.get('shuffle_seed', 9176)
    shuffle_block_size = train_dataset.get('shuffle_block_size', None)
    sampling_method = train_dataset.get('sampling_method', 'balanced')
    sampling_granularity = train_dataset.get('sampling_granularity', 1)
    batching_method = train_dataset.get('batching_method', 'random')

    dataset = SimulationDataset(nodes, devices, workers, streams, remote, local, split,
                                download_retry, download_timeout, validate_hash, keep_zip,
                                epoch_size, predownload, cache_limit, partition_algo,
                                num_canonical_nodes, batch_size, shuffle, shuffle_algo,
                                shuffle_seed, shuffle_block_size, sampling_method,
                                sampling_granularity, batching_method)

    return dataset
