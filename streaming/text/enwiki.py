# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""English Wikipedia 2020-01-01 streaming dataset."""

from typing import Any, Optional

import numpy as np

from streaming.base import StreamingDataset

__all__ = ['StreamingEnWiki']


class StreamingEnWiki(StreamingDataset):
    """Implementation of the English Wikipedia 2020-01-01 streaming dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to
            ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        shuffle_world_size (int, optional): Canonical world size for shuffling. Defaults to
            ``None``, which is interpreted as the world size of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 predownload: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 shuffle_seed: Optional[int] = None,
                 shuffle_world_size: Optional[int] = None,
                 batch_size: Optional[int] = None):
        super().__init__(local, remote, split, shuffle, predownload, keep_zip, download_retry,
                         download_timeout, validate_hash, shuffle_seed, shuffle_world_size,
                         batch_size)
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

    def __getitem__(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        obj = super().__getitem__(idx)

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
