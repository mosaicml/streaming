# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""English Wikipedia 2020-01-01 streaming dataset."""

from typing import Any, Optional

import numpy as np

from streaming.base import StreamingDataset

__all__ = ['StreamingEnWiki']


class StreamingEnWiki(StreamingDataset):
    """Implementation of the English Wikipedia 2020-01-01 streaming dataset.

    Args:
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            ``False``.
        choose (int, optional): Number of samples to draw per epoch balanced across all streams.
            If ``None``, takes its value from the total number of underlying samples. Provide this
            field if you are weighting streams relatively to target a larger or smaller epoch size.
            Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards per
            number of workers provided in a dataloader while iterating. Defaults to ``512``.
        cache_limit (int, optional): Maximum size in bytes of this StreamingDataset's shard cache.
            Before downloading a shard, the least recently used resident shard(s) may be evicted
            (deleted from the local cache) in order to stay under the limit. Set to ``None`` to
            disable shard eviction. Defaults to ``None``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. If ``None``, this is interpreted as the number of nodes of the initial run
            multiplied by 64. Defaults to ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
    """

    def __init__(self,
                 *,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 choose: Optional[int] = None,
                 predownload: Optional[int] = 512,
                 cache_limit: Optional[int] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip,
                         choose=choose,
                         predownload=predownload,
                         cache_limit=cache_limit,
                         partition_algo=partition_algo,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed,
                         shuffle_block_size=shuffle_block_size)
        self.field_dtypes = {
            'input_ids': np.int32,
            'input_mask': np.int32,
            'attention_mask': np.int32,
            'segment_ids': np.int32,
            'token_type_ids': np.int32,
            'masked_lm_positions': np.int32,
            'masked_lm_ids': np.int32,
            'masked_lm_weights': np.float32,
            'next_sentence_labels': np.int32,
            'labels': np.int32,
        }

    def get_item(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        obj = super().get_item(idx)

        for key, value in obj.items():
            dtype = self.field_dtypes[key]
            obj[key] = np.frombuffer(value, dtype)

        input_len = len(obj['input_ids'])
        labels = np.full((input_len,), -100)
        labels[obj['masked_lm_positions']] = obj['masked_lm_ids']

        return {
            'input_ids': obj['input_ids'].copy(),
            'token_type_ids': obj['segment_ids'].copy(),
            'attention_mask': obj['input_mask'].copy(),
            'labels': labels,
        }
