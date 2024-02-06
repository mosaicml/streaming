# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""C4 (Colossal Cleaned Common Crawl) dataset.

This dataset is a colossal, cleaned version of Common Crawl's web crawl corpus and it is based on
the `Common Crawl <https://commoncrawl.org>`_ dataset.
"""

from typing import Any, Dict

from transformers.models.auto.tokenization_auto import AutoTokenizer

from streaming import StreamingDataset

__all__ = ['StreamingC4']


class StreamingC4(StreamingDataset):
    """Implementation of the C4 (Colossal Cleaned Common Crawl) dataset using StreamingDataset.

    Args:
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Currently only supporting
            ``'truncate'``.
        **kwargs (Dict[str, Any]): Keyword arguments.
    """

    def __init__(self, *, tokenizer_name: str, max_seq_len: int, group_method: str,
                 **kwargs: Dict[str, Any]) -> None:
        if group_method not in {'truncate'}:
            raise ValueError(f"group_method='{group_method}' must be one of {'truncate'}.")

        super().__init__(**kwargs)

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

    def get_item(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        text_sample = super().get_item(idx)
        token_sample = self._tokenize(text_sample)
        # Skip any token grouping, currently only supporting group_method='truncate'
        return token_sample