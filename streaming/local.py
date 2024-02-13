# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A non-streaming pytorch map Dataset."""

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from streaming.array import Array
from streaming.format.base.phaser import Phaser
from streaming.spanner import Spanner
from streaming.stream.base import Stream

__all__ = ['LocalDataset']


class LocalDataset(Array, Dataset):
    """A streaming dataset whose shards reside locally as a PyTorch Dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        split (str, optional): Which dataset split sub-path to use, if any. Defaults to ``None``.
        allow_schema_mismatch (bool): If ``True``, continue if sample columns mismatch across
            shards, streams, or the whole dataset. If ``False``, raises if columns mismatch.
            Defaults to ``False``.
        allow_unsafe_types (bool): If ``True``, continue if unsafe type(s) are encountered
            in shard(s). If ``False``, raises if unsafe type(s) encountered. Defaults to ``False``.
        allow_unchecked_resumption (bool): If ``True``, upon resume, accept and use shard
            file phases that we are unable to check the size/hash(es) of. If ``False``, upon
            resume, drop such files, to regenerate on the fly when needed. Defaults to ``True``.
        validate_hash (str | Sequence[str], optional): Ranked list of hashing algorithms to
            apply if expected digest is available. Defaults to ``None``.
        keep_phases (str | Sequence[str] | Dict[str, Optional[bool]] | Phaser): Which phases
            to keep and to drop upon conversion, given either by intended use case or literally.
            Specified as a single use or phase to keep, a sequence of uses or phases to keep, a
            mapping of uses or phases to whether to keep or drop, or a ``Phaser`` (which performs
            the same keeping or dropping). Defaults to ``None``.
        kwargs (Any): Any unsupported (for forward compat) or deprecated args.
    """

    def __init__(
        self,
        local: str,
        split: Optional[str] = None,
        allow_schema_mismatch: bool = False,
        allow_unsafe_types: bool = False,
        allow_unchecked_resumption: bool = True,
        validate_hash: Union[None, str, Sequence[str]] = None,
        keep_phases: Union[None, str, Sequence[str], Dict[str, Optional[bool]], Phaser] = None,
        **kwargs: Any,
    ) -> None:
        self.stream = Stream(
            local=local,
            split=split,
            allow_schema_mismatch=allow_schema_mismatch,
            allow_unsafe_types=allow_unsafe_types,
            allow_unchecked_resumption=allow_unchecked_resumption,
            download_retry=0,
            download_timeout=None,
            download_max_size=None,
            validate_hash=validate_hash,
            keep_phases=keep_phases,
            **kwargs,
        )
        self.stream.apply_defaults()
        self.stream.download_index()
        self.shards = self.stream.load_index()
        shard_sizes = np.array([shard.num_samples for shard in self.shards], np.int64)
        self.spanner = Spanner(shard_sizes)
        self.num_samples = sum(shard_sizes)

    def __len__(self) -> int:
        """Get the length as a PyTorch Dataset.

        Returns:
            int: Dataset length.
        """
        return self.num_samples

    @property
    def size(self) -> int:
        """Get the size of the dataset in samples.

        Returns:
            int: Number of samples.
        """
        return self.num_samples

    def get_item(self, sample_id: int) -> Dict[str, Any]:
        """Get sample by global sample ID.

        Args:
            sample_id (int): Sample ID.

        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard_id, index_in_shard = self.spanner[sample_id]
        shard = self.shards[shard_id]
        return shard[index_in_shard]
