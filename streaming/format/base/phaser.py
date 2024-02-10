# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""What shard file phases to keep."""

from copy import deepcopy
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self


class Locality(IntEnum):
    """Where a shard file phase is to be found."""

    LOCAL = 1
    REMOTE = 2
    DNE = 3  # Does Not Exist


class Phaser:
    """Keep or drop shard file phases, configured by intended use case and/or explicitly.

    Args:
        persistent (bool): Whether to cache the first phase of the file that its
            format uses. This is the phase used for persistent persistent and downloading, i.e, the
            phase required in order to serve as a Stream remote. It is ``zip``  if compression was
            used, or ``raw`` otherwise. Defaults to ``False``.
        medial (bool): Whether to cache any phase(s) between the ones used
            for ``persistent`` and``checked``, respectively. Currently, this would only
            select a phase in the case of compressed Streaming Parquet shards with a sized
            canonicalized form, but it is possible that phases may become more complex in the
            future. Useful if you want to keep all phases of a shard file. Defaults to ``False``.
        checked (bool): Whether to cache the last phase of the file
            that its format uses that is able to be checked on resumption, by having associated
            size/hash metadata set. For Parquet shards to be converted on the fly, we will not have
            any expected size or hashes for its last (i.e. canonicalized) phase. Useful if you want
            to not accept any uncheckable shard files you find cached locally upon init. Defaults to
            ``True``.
        active (bool): Whether to cache the very last phase of the file that its
            format uses, without regard to whether it can be checked or not. This phase is the
            final product of shard preparation. Set to ``True`` if you intend to use the shard in
            any way, and to ``False`` for dry runs of shard preparation, clearing a dataset to just
            the ``persistent`` phase of shards, or other such purpose. Defaults to ``True``.
        zip (bool): Whether to cache the ``zip`` phase of the file, if its
            format uses it. If ``bool``, it overrides what the intended use cases would do. If
            ``None``, it takes its value from from the intended use cases would do. Defaults to
            ``False``.
        raw (bool): Whether to cache the ``raw`` phase of the file, if its
            format uses it. If ``bool``, it overrides what the intended use cases would do. If
            ``None``, it takes its value from from the intended use cases would do. Defaults to
            ``False``.
        can (bool): Whether to cache the ``can`` phase of the file, if its
            format uses it. If ``bool``, it overrides what the intended use cases would do. If
            ``None``, it takes its value from from the intended use cases would do. Defaults to
            ``False``.
    """

    def __init__(
        self,
        persistent: bool = False,
        medial: bool = False,
        checked: bool = True,
        active: bool = True,
        zip: bool = False,
        raw: bool = False,
        can: bool = False,
    ) -> None:
        self.persistent = persistent
        self.medial = medial
        self.checked = checked
        self.active = active
        self.zip = zip
        self.raw = raw
        self.can = can

    def to_safe(self) -> Self:
        """Get the safe version of this Phaser, which never deletes the persistent phase.

        Returns:
            Self: Safe version of this Phaser.
        """
        ret = deepcopy(self)
        ret.persistent = True
        return ret

    def get_phase_deletions(
        self,
        phase_locs: NDArray[np.int64],
        phase_chks: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Get phases to delete.

        Args:
            phase_locs (NDArray[np.int64]): Phase localities.

        Returns:
            NDArray[np.int64]: Phase deletions.
        """
        # There must be entries for three phases.
        if len(phase_locs) != 3:
            raise ValueError(f'All shard formats use some selection of the three phases.')
        if len(phase_chks) != 3:
            raise ValueError(f'All shard formats use some selection of the three phases.')

        # The `raw` phase must must exist somewhere (either LOCAL or REMOTE).
        if phase_locs[1] == Locality.DNE:
            raise ValueError(f'All shard formats must define at least their `raw` phase, but ' +
                             f'got phase localities: {phase_locs}.')

        # The first phase used is the `persistent` phase. It is either LOCAL or REMOTE.
        for persistent_idx, loc in enumerate(phase_locs):
            if loc != Locality.DNE:
                break
        else:
            raise ValueError(f'All shard formats must define at least their `raw` phase, but ' +
                             f'got phase localities: {phase_locs}.')

        # The last phase of known size is the `checked` phase. -1 if no phase has a known size.
        for checked_idx, is_sized in reversed(list(enumerate(phase_chks))):
            if is_sized:
                break
        else:
            checked_idx = -1

        # Sizes must start known, then transition to unknown (works for -1 too).
        for is_sized in phase_chks[:checked_idx + 1]:
            if not is_sized:
                raise ValueError(f'All sized phases must precede all unsized phases, but got: ' +
                                 f'{phase_chks.tolist()}.')

        # Starting at the desired phase, work backwards to the closest phase present to the actual
        # checked phase. We do this because we need to transition through the phases to get there.
        for checked_idx in reversed(range(checked_idx + 1)):
            if phase_locs[checked_idx] == Locality.LOCAL:
                break
        else:
            checked_idx = -1

        # The last phase used is the `active` phase. The `raw` phase must be used by all formats,
        # so there will always be an active phase. We keep the phase that is closest to it, i.e.
        # the last phase present, because we need to transition through the phases to get there.
        for active_idx, loc in reversed(list(enumerate(phase_locs))):
            if loc == Locality.LOCAL:
                break
        else:
            active_idx = -1

        by_goal = np.zeros(3, np.int64)
        by_goal[persistent_idx] = True
        if checked_idx != -1:
            by_goal[checked_idx] = True
        if active_idx != -1:
            by_goal[active_idx] = True

        by_phase = np.array([self.zip, self.raw, self.can], np.int64)
        if (by_phase < 0).any() or (1 < by_phase.any()):
            raise ValueError(f'Phaser fields zip, raw, and can must be bools, but got: zip ' +
                             f'{self.zip}, raw {self.raw}, can {self.can}.')

        keep = by_goal | by_phase
        dont_keep = 1 - keep
        is_local = phase_locs == Locality.LOCAL
        return dont_keep * is_local
