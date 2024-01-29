# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""What shard file phases to keep or delete."""

from enum import Enum
from typing import Union

from streaming.util.auto import Auto


class Locality(Enum):
    """Where a shard file phase is to be found."""

    LOCAL = 1
    REMOTE = 2
    DNE = 3


class Phaser:
    """Keep or drop shard file phases, configured by intended use case and/or explicitly.

    Args:
        storage (bool | Auto): Whether to cache or delete the first phase of the file that its
            format uses. This is the phase used for persistent storage and downloading, i.e, the
            phase required in order to serve as a Stream remote. It is ``zip``  if compression was
            used, or ``raw`` otherwise. If ``Auto``, falls back to ``False``.`Defaults to
            ``Auto()``.
        intermediates (bool | Auto): Whether to cache or delete any phase(s) between the ones used
            for ``storage`` and``checked_resumption``, respectively. Currently, this would only
            select a phase in the case of compressed Streaming Parquet shards with a sized
            canonicalized form, but it is possible that phases may become more complex in the
            future. Useful if you want to keep all phases of a shard file. If ``Auto``, falls
            back to ``False``. Defaults to ``Auto()``.
        checked_resumption (bool | Auto): Whether to cache or delete the last phase of the file
            that its format uses that is able to be checked on resumption, by having associated
            size/hash metadata set. For Parquet shards to be converted on the fly, we will not have
            any expected size or hashes for its last (i.e. canonicalized) phase. Useful if you want
            to not accept any uncheckable shard files you find cached locally upon init. If
            ``Auto()``, falls back to ``True``. Defaults to ``Auto()``.
        access (bool | Auto): Whether to cache or delete the very last phase of the file that its
            format uses, without regard to whether it can be checked or not. This phase is the
            final product of shard preparation. Set to ``True`` if you intend to use the shard in
            any way, and to ``False`` for dry runs of shard preparation, clearing a dataset to just
            the ``storage`` phase of shards, or other such purpose. If ``Auto``, falls back to
            ``True``. Defaults to ``Auto()``.
        zip (None | bool | Auto): Whether to cache or delete the ``zip`` phase of the file, if its
            format uses it. If ``bool``, it overrides what the intended use cases would do. If
            ``None``, it takes its value from from the intended use cases would do. If ``Auto``,
            falls back to ``None``.
        raw (None | bool | Auto): Whether to cache or delete the ``raw`` phase of the file, if its
            format uses it. If ``bool``, it overrides what the intended use cases would do. If
            ``None``, it takes its value from from the intended use cases would do. If ``Auto``,
            falls back to ``None``.
        can (None | bool | Auto): Whether to cache or delete the ``can`` phase of the file, if its
            format uses it. If ``bool``, it overrides what the intended use cases would do. If
            ``None``, it takes its value from from the intended use cases would do. If ``Auto``,
            falls back to ``None``.
    """

    def __init__(
            self,
            storage: Union[bool, Auto] = Auto(),
            intermediates: Union[bool, Auto] = Auto(),
            checked_resumption: Union[bool, Auto] = Auto(),
            access: Union[bool, Auto] = Auto(),
            zip: Union[None, bool, Auto] = Auto(),
            raw: Union[None, bool, Auto] = Auto(),
            can: Union[None, bool, Auto] = Auto(),
    ) -> None:
        self.storage = storage
        self.intermediates = intermediates
        self.checked_resumption = checked_resumption
        self.access = access
        self.zip = zip
        self.raw = raw
        self.can = can
