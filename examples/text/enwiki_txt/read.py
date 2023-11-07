# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""English Wikipedia 2020-01-01 streaming dataset."""

from typing import Any, Dict

import numpy as np

from streaming import StreamingDataset

__all__ = ['StreamingEnWiki']


class StreamingEnWiki(StreamingDataset):
    """Implementation of the English Wikipedia 2020-01-01 streaming dataset.

    Args:
        **kwargs (Dict[str, Any]): Keyword arguments.
    """

    def __init__(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

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
