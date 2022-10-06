# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""English Wikipedia 2020-01-01 streaming dataset."""

import numpy as np
from typing import Any, Dict, Optional

from transformers.models.auto.tokenization_auto import AutoTokenizer

from streaming.base import Dataset


class EnWiki(Dataset):
    """Implementation of the English Wikipedia 2020-01-01 streaming dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to
            ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to shuffle the samples while iterating. Defaults to ``True``.
        prefetch (int, optional): Target number of samples remaining to prefetch while iterating.
            Defaults to ``None``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        timeout (float): Number of seconds to wait for a shard to download before raising an
            exception. Defaults to ``60``.
        hash (str, optional): Optional hash or checksum algorithm to use to validate shards.
            Defaults to ``None``.
        batch_size (int, optional): Hint the batch_size that will be used on each device's
            DataLoader. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 prefetch: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 retry: int = 2,
                 timeout: float = 60,
                 hash: Optional[str] = None,
                 batch_size: Optional[int] = None) -> None:
        super().__init__(local, remote, split, shuffle, prefetch, keep_zip, retry, timeout, hash,
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
            obj[key] = np.copy(np.frombuffer(value, dtype))

        ret_obj = {}
        ret_obj['input_ids'] = obj['input_ids']
        ret_obj['token_type_ids'] = obj['segment_ids']
        ret_obj['attention_mask'] = obj['input_mask']

        input_len = len(obj['input_ids'])
        ret_obj['labels'] = np.full((input_len,), -100)
        ret_obj['labels'][obj['masked_lm_positions']] = obj['masked_lm_ids']

        return ret_obj
