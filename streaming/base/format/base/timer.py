# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A recursive timer contextmanager, whose state is serializable, which calculates stats."""

from time import time_ns
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

__all__ = ['Timer']


class Timer:
    """A recursive timer contextmanager, whose state is serializable, which calculates stats.

    Args:
        named_timers (List[Union[str, Tuple[str, Self]]], optional): List of pairs of (name,
            Timer). Defaults to ``None``.
        spans (List[Tuple[int, int]] | NDArray[np.int64], optional): Either a list of pairs of
            (enter, exit) times, or the same in array form. Times are given in nanoseconds since
            the epoch. ``None`` means the empty list. Defaults to ``None``.
    """

    def __init__(self,
                 named_timers: Optional[List[Union[str, Tuple[str, Self]]]] = None,
                 spans: Optional[Union[List[Tuple[int, int]], NDArray[np.int64]]] = None) -> None:
        if spans is None:
            self.spans = []
        elif isinstance(spans, list):
            np.asarray(spans, np.int64)
            self.spans = spans
        else:
            self.spans = spans.tolist()

        self.named_timers = []
        for named_timer in named_timers or []:
            if isinstance(named_timer, str):
                named_timer = named_timer, Timer()
            name, timer = named_timer
            if hasattr(self, name):
                raise ValueError(f'Timer name is already taken: {name}.')
            setattr(self, name, timer)
            self.named_timers.append((name, timer))

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple[Self, int]:
        """Efficiently deserialize our state from bytes.

        The offset is needed because this process is recursive.

        Args:
            data (bytes): Buffer containing its serialized form.
            offset (int): Byte offset into the given buffer. Defaults to ``0``.

        Returns:
            Tuple[Self, int]: Pair of (loaded class, byte offset into the buffer afterward).
        """
        dtype = np.int64()

        if len(data) % dtype:
            raise ValueError(f'`data` size must be divisible by {dtype.nbytes}, but got: ' +
                             f'{len(data)}.')

        if offset % dtype.nbytes:
            raise ValueError(f'`offset` must be divisible by {dtype.nbytes}, but got: {offset}.')

        arr = np.frombuffer(data, np.int64)
        idx = offset // dtype.nbytes

        num_spans = int(arr[idx])
        idx += 1

        spans = arr[idx:idx + num_spans * 2].reshape(num_spans, 2)
        idx += num_spans * 2

        num_named_timers = arr[idx]
        idx += 1

        named_timers = []
        for _ in range(num_named_timers):
            name_size = arr[idx]
            idx += 1

            pad_size = dtype.nbytes - name_size % dtype.nbytes
            name_units = (name_size + pad_size) // 8
            name_bytes = arr[idx:idx + name_units].tobytes()[:-pad_size]
            name = name_bytes.decode('utf-8')
            idx += name_units

            subtimer, offset = cls.from_bytes(data, idx * dtype.nbytes)
            idx = offset // dtype.nbytes

            named_timer = name, subtimer
            named_timers.append(named_timer)

        timer = cls(named_timers, spans)
        return timer, offset

    def to_bytes(self) -> bytes:
        """Efficiently serialize our state to bytes.

        Returns:
            bytes: Serialized state.
        """
        num_spans = np.int64(len(self.spans))
        spans = np.asarray(self.spans, np.int64)
        num_named_timers = np.int64(len(self.named_timers))
        parts = [num_spans.tobytes(), spans.tobytes(), num_named_timers.tobytes()]
        dtype = np.int64()
        for name, timer in self.named_timers:
            name_bytes = name.encode('utf-8')
            name_size = np.int64(len(name_bytes))
            pad_size = dtype.nbytes - name_size % dtype.nbytes
            pad_bytes = '\0' * pad_size
            parts += [name_size.tobytes(), name_bytes, pad_bytes, timer.to_bytes()]
        return b''.join(parts)

    def to_dynamic(self) -> Self:
        """Convert our spans to a dynamic-size list of pairs for fast data collection.

        Returns:
            Self: This object.
        """
        if isinstance(self.spans, np.ndarray):
            self.spans = self.spans.tolist()
        for _, timer in self.named_timers:
            timer.to_dynamic()
        return self

    def to_fixed(self) -> Self:
        """Convert our spans to a fixed-size array for fast data analysis.

        Returns:
            Self: This object.
        """
        if isinstance(self.spans, list):
            self.spans = np.asarray(self.spans, np.int64)
        for _, timer in self.named_timers:
            timer.to_fixed()
        return self

    def _get_duration_stats(self, num_groups: Optional[int] = 100) -> Dict[str, Any]:
        """Calculate duration statistics.

        Args:
            num_groups (int, optional): Number of groups for quantiling. Defaults to ``100``.

        Returns:
            Dict[str, Any]: Duration statistics.
        """
        if isinstance(self.spans, list):
            self.spans = np.asarray(self.spans, np.int64)

        durs = self.spans[:, 1] - self.spans[:, 0]
        durs = durs / 1e9

        obj = {
            'total': float(durs.sum()),
        }

        if 1 < len(durs):
            obj.update({
                'count': len(durs),
                'min': float(min(durs)),
                'max': float(max(durs)),
                'mean': float(durs.mean()),
                'std': float(durs.std()),
            })

            if num_groups:
                fracs = np.linspace(0, 1, num_groups + 1)
                obj['quantiles'] = np.quantile(durs, fracs).tolist()

        return obj

    def get_stats(self, num_groups: Optional[int] = 100) -> Dict[str, Any]:
        """Get statistics.

        Args:
            num_groups (int, optional): Number of groups for quantiling. Defaults to ``100``.

        Returns:
            Dict[str, Any]: Recursive dict of duration statistics.
        """
        obj = {
            'stats': self._get_duration_stats(num_groups),
        }

        if self.named_timers:
            named_timers = []
            for name, timer in self.named_timers:
                named_timers.append([name, timer.get_stats(num_groups)])

            whole = obj['stats']['total']
            sum_of_parts = 0
            for name, timer in named_timers:
                part = timer['stats']['total']
                timer['stats']['frac_of_whole'] = part / whole
                sum_of_parts += part

            for name, timer in named_timers:
                part = timer['stats']['total']
                timer['stats']['frac_of_parts'] = part / sum_of_parts

            obj['named_timers'] = named_timers  # pyright: ignore

        return obj

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns:
            Self: This object.
        """
        if isinstance(self.spans, np.ndarray):
            self.spans = self.spans.tolist()

        span = time_ns(), 0
        self.spans.append(span)
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]] = None,
                 exc: Optional[BaseException] = None,
                 traceback: Optional[TracebackType] = None) -> None:
        """Exit context manager.

        Args:
            exc_type (Type[BaseException], optional): Exception type. Defaults to ``None``.
            exc (BaseException, optional): Exception. Defaults to ``None``.
            traceback (TracebackType, optional): Traceback. Defaults to ``None``.
        """
        if isinstance(self.spans, np.ndarray):
            self.spans = self.spans.tolist()

        span = self.spans[-1]
        self.spans[-1] = span[0], time_ns()
