# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

from transformers.models.auto.tokenization_auto import AutoTokenizer

from streaming.base import Dataset


class C4(Dataset):
    """C4 (Colossal Cleaned Common Crawl) dataset.

    Args:
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Currently only supporting
            ``'truncate'``.
        local (str): Local dataset directory where shards are cached by split.
        remote (Optional[str], default: None): Download shards from this remote path or directory.
            If None, this rank and workers' partition of the dataset must all exist locally.
        split (Optional[str], default: None): Which dataset split to use, if any.
        shuffle (bool, default: True): Whether to shuffle the samples while iterating.
        prefetch (Optional[int], default: 100_000): Target number of samples remaining to prefetch
            while iterating.
        keep_zip (Optional[bool], default: None): Whether to keep or delete the compressed file
            when decompressing downloaded shards. If set to None, keep iff remote is local.
        retry (int, default: 2): Number of download re-attempts before giving up.
        timeout (float, default: 60): Number of seconds to wait for a shard to download before
            raising an exception.
        hash (Optional[str], default: None): Optional hash or checksum algorithm to use to validate
            shards.
        batch_size (Optional[int], default: None): Hint the batch_size that will be used on each
            device's DataLoader.
    """

    def __init__(self,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str,
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
        if group_method not in ['truncate']:
            raise ValueError(f"Only group_method='truncate' is supported at this time.")

        super().__init__(local, remote, split, shuffle, prefetch, keep_zip, retry, timeout, hash,
                         batch_size)
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method

        # Build tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)  # pyright: ignore
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize(self, text_sample: Dict[str, Any]):
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        else:
            truncation = False
            padding = False
            max_length = None
        return self.tokenizer(text_sample['text'],
                              truncation=truncation,
                              padding=padding,
                              max_length=max_length)

    def __getitem__(self, idx: int) -> Any:
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        # Skip any token grouping, currently only supporting group_method='truncate'
        return token_sample
