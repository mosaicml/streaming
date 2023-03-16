# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""C4 (Colossal Cleaned Common Crawl) dataset.

This dataset is a colossal, cleaned version of Common Crawl's web crawl corpus and it is based on
the `Common Crawl <https://commoncrawl.org>`_ dataset.
"""

from typing import Any, Dict, Optional

from transformers.models.auto.tokenization_auto import AutoTokenizer

from streaming.base import StreamingDataset

__all__ = ['StreamingC4']


class StreamingC4(StreamingDataset):
    """Implementation of the C4 (Colossal Cleaned Common Crawl) dataset using StreamingDataset.

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
            downloaded shards. If ``False``, keep if remote is local or no remote. Defaults to
            `False``.
        keep_raw (bool, optional): Whether to keep or delete the decompressed form (or only form)
            of shards after they have been used for the time being this epoch. If ``False``, keep
            if remote is local or no remote and no compression. Defaults to ``None``.
        raw_ttl (float): If ``keep_raw`` is ``False``, the maximum amount of time between
            successive usages of a shard on this node before it is dropped after the last usage, as
            a fraction of the epoch size. Defaults to ``0.25``.
        samples_per_epoch (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Currently only supporting
            ``'truncate'``.
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
                 keep_raw: bool = True,
                 raw_ttl: float = 0.25,
                 samples_per_epoch: Optional[int] = None,
                 predownload: Optional[int] = 100_000,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str) -> None:
        if group_method not in {'truncate'}:
            raise ValueError(f"group_method='{group_method}' must be one of {'truncate'}.")

        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip,
                         keep_raw=keep_raw,
                         raw_ttl=raw_ttl,
                         samples_per_epoch=samples_per_epoch,
                         predownload=predownload,
                         partition_algo=partition_algo,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed)
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method

        # Build tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize(self, text_sample: Dict[str, Any]):
        """Apply the tokenizer to a sample.

        Args:
            text_sample (Dict[str, Any]): Sample to tokenize.
        """
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        else:
            raise ValueError(f'Got unknown group_method={self.group_method}.')
        return self.tokenizer(text_sample['text'],
                              truncation=truncation,
                              padding=padding,
                              max_length=max_length)

    def __getitem__(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        # Skip any token grouping, currently only supporting group_method='truncate'
        return token_sample
