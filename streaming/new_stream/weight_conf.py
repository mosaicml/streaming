# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Configures how a Stream is weighted."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from streaming.util.shorthand import normalize_count

__all__ = ['StreamWeightConf']


class StreamWeightConf:
    """Configures how a Stream is weighted.

    Args:
        proportion (float, optional): The proportion of this StreamingDataset's samples that are
            sampled from this Stream. As this is a relative measure, use ``epoch_size`` to
            determine the absolute resulting size in samples. Defaults to ``None``.
        repeat (float, optional): Stream size multiplier, aka number of times to see each of this
            Stream's samples per epoch. Defaults to ``None``.
        choose (str | int, optional): Stream size, aka number of samples to draw from this Stream
            per epoch. Defaults to ``None``.
        kwargs (Dict[str, Any]): Any unsupported (for forward compat) or deprecated args.
    """

    def __init__(
        self,
        *,
        proportion: Optional[float] = None,
        repeat: Optional[float] = None,
        choose: Optional[Union[str, int]] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        has_proportion = proportion is not None
        has_repeat = repeat is not None
        has_choose = choose is not None
        if not (0 <= has_proportion + has_repeat + has_choose <= 1):
            raise ValueError('At most one of `proportion`, `repeat`, and `choose` may be ' +
                             'specified; the others are derived')

        if proportion is not None:
            if proportion < 0:
                raise ValueError('`proportion` must be non-negative')
            self.proportion = proportion

        if repeat is not None:
            if repeat < 0:
                raise ValueError('`repeat` must be non-negative')
            self.repeat = repeat

        if choose is not None:
            self.choose = normalize_count(choose)
            if self.choose < 0:
                raise ValueError('`choose` must be non-negative')

    @classmethod
    def validate_weights(
        cls,
        streams: Sequence[Self],
    ) -> Tuple[bool, bool]:
        """Validate stream weights, returning whether relative or absolute weighting was used.

        Args:
            streams (Sequence[Stream]): Every stream comprising the dataset.

        Returns:
            bool: Whether streams are weighted relatively (proportionally).
        """
        # Validate stream weights ("proportion", "repeat", "choose", or none).
        is_proportional = hasattr(streams[0], 'proportion')
        is_unspecified = True
        for stream_id, stream in enumerate(streams):
            has_proportion = hasattr(stream, 'proportion')
            has_repeat = hasattr(stream, 'repeat')
            has_choose = hasattr(stream, 'choose')
            if not (0 <= has_proportion + has_repeat + has_choose <= 1):
                raise ValueError(f'Streams must provide at most one of `proportion`, `repeat`, ' +
                                 f'or `choose` (error in stream {stream_id})')
            if is_proportional != has_proportion:
                raise ValueError(f'Relative (`proportion`) and absolute (`repeat`, `choose`, ' +
                                 f'none) stream weights are incompatible with each other (error ' +
                                 f'in stream {stream_id})')
            if has_proportion or has_repeat or has_choose:
                is_unspecified = False
        return is_proportional, is_unspecified

    @classmethod
    def apply_weights(
        cls,
        streams: Sequence[Self],
        samples_per_stream: NDArray[np.int64],
        choose_per_epoch: Optional[int],
        seed: int,
    ) -> int:
        """Given samples per stream, derive each stream's proportion/repeat/samples.

        Modifies streams to save the derived weights.

        Args:
            streams (Sequence[Stream]): The list of streams which comprise the dataset.
            samples_per_stream (NDArray[np.int64]): Underlying samples of each stream.
            choose_per_epoch (int, optional): Absolute epoch size if weighting relatively.
            seed (int): Random number generator seed used to sample evenly.

        Returns:
            int: Number of samples to draw per epoch.
        """
        # Validate provided weights, determining whether they are relative or absolute.
        are_weights_relative, are_weights_unspecified = cls.validate_weights(streams)

        # Derive weights.
        if are_weights_relative:
            # Relative.
            if not choose_per_epoch:
                choose_per_epoch = sum(samples_per_stream)
            proportion_per_stream = np.array([stream.proportion for stream in streams], np.float64)
            proportion_per_stream /= proportion_per_stream.sum()
            choose_per_stream = (choose_per_epoch * proportion_per_stream).astype(np.int64)
            shortfall = choose_per_epoch - choose_per_stream.sum()
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(streams), shortfall, False)
            choose_per_stream[indices] += 1
            repeat_per_stream = choose_per_stream / samples_per_stream
        elif are_weights_unspecified and choose_per_epoch:
            # weights are unspecified, but epoch size (choose_per_epoch) is provided.
            # sample from each stream in proportion stream's samples
            proportion_per_stream = samples_per_stream.copy().astype(np.float64)
            proportion_per_stream /= proportion_per_stream.sum()
            choose_per_stream = (choose_per_epoch * proportion_per_stream).astype(np.int64)
            shortfall = choose_per_epoch - choose_per_stream.sum()
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(streams), shortfall, False)
            choose_per_stream[indices] += 1
            repeat_per_stream = choose_per_stream / samples_per_stream
        else:
            # Absolute.
            if choose_per_epoch:
                raise ValueError('Only provide `choose` when weighting streams relatively')
            choose_per_stream = np.zeros(len(streams), np.int64)
            for stream_id, stream in enumerate(streams):
                if hasattr(stream, 'repeat'):
                    choose = int(stream.repeat * samples_per_stream[stream_id])
                elif hasattr(stream, 'choose'):
                    choose = stream.choose
                else:
                    choose = samples_per_stream[stream_id]
                choose_per_stream[stream_id] = choose
            repeat_per_stream = choose_per_stream / samples_per_stream
            proportion_per_stream = choose_per_stream / choose_per_stream.sum()
            choose_per_epoch = sum(choose_per_stream)

        # Now that we know the true props/reps/choices, inject those back into the streams.
        for stream, proportion, repeat, choose in zip(streams, proportion_per_stream,
                                                      repeat_per_stream, choose_per_stream):
            stream.proportion = proportion
            stream.repeat = repeat
            stream.choose = choose

        return choose_per_epoch
