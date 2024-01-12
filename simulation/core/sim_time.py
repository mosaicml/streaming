# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Time classes ported from MosaicML composer.

Avoids dependency on composer and its many reqs.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Generic, TypeVar, Union, cast


class TimeUnit(Enum):
    """Enum class to represent units of time for the training process.

    Attributes:
        EPOCH (str): Epochs.
        BATCH (str): Batches (i.e. number of optimization steps)
        SAMPLE (str): Samples.
        TOKEN (str): Tokens. Applicable for natural language processing (NLP) models.
        DURATION (str): Fraction of the training process complete, on ``[0.0, 1.0)``
    """
    EPOCH = 'ep'
    BATCH = 'ba'
    SAMPLE = 'sp'
    TOKEN = 'tok'
    DURATION = 'dur'


# regex for parsing time string, matches timeunit and chars prior to unit as value
_TIME_STR_REGEX = re.compile(r'^(.+)(' +
                             r'|'.join([fr'{time_unit.value}' for time_unit in TimeUnit]) + r')$',
                             flags=re.IGNORECASE)

TValue = TypeVar('TValue', int, float)


class Time(Generic[TValue]):
    """Time represents static durations of training time in terms of a `TimeUnit` enum.

    This is identical to the `Time` class in MosaicML Composer. See the Composer docs for more
    details on tracking time during training.

    Args:
        value (int | float): The amount of time.
        unit (str | TimeUnit): The `TimeUnit` for ``value``.
    """

    def __init__(
        self,
        value: TValue,
        unit: Union[str, TimeUnit],
    ):
        unit = TimeUnit(unit)
        if unit == TimeUnit.DURATION:
            value = cast(TValue, float(value))
        else:
            if not isinstance(value, int):
                raise TypeError(
                    f'value {value} is of type {type(value)}. Units {unit} require integer values.'
                )
        self._value, self._unit = value, TimeUnit(unit)

    @classmethod
    def from_epoch(cls, epoch: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.EPOCH`.

        Equivalent to ``Time(epoch, TimeUnit.EPOCH)``.

        Args:
            epoch (int): Number of epochs.

        Returns:
            Time: :class:`Time` instance, in epochs.
        """
        return cls(epoch, TimeUnit.EPOCH)

    @classmethod
    def from_batch(cls, batch: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.BATCH`.

        Equivalent to ``Time(batch, TimeUnit.BATCH)``.

        Args:
            batch (int): Number of batches.

        Returns:
            Time: :class:`Time` instance, in batches.
        """
        return cls(batch, TimeUnit.BATCH)

    @classmethod
    def from_sample(cls, sample: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.SAMPLE`.

        Equivalent to ``Time(sample, TimeUnit.SAMPLE)``.

        Args:
            sample (int): Number of samples.

        Returns:
            Time: :class:`Time` instance, in samples.
        """
        return cls(sample, TimeUnit.SAMPLE)

    @classmethod
    def from_token(cls, token: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.TOKEN`.

        Equivalent to ``Time(sample, TimeUnit.TOKEN)``.

        Args:
            token (int): Number of tokens.

        Returns:
            Time: :class:`Time` instance, in tokens.
        """
        return cls(token, TimeUnit.TOKEN)

    @classmethod
    def from_duration(cls, duration: float) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.DURATION`.

        Equivalent to ``Time(duration, TimeUnit.DURATION)``.

        Args:
            duration (float): Duration of the training process. Should be on ``[0, 1)``
                where ``0`` represents the beginning of the training process and ``1``
                represents a completed training process.

        Returns:
            Time: :class:`Time` instance, in duration.
        """
        return cls(duration, TimeUnit.DURATION)

    @property
    def value(self) -> TValue:
        """The value of the time, as a number."""
        return self._value

    @property
    def unit(self) -> TimeUnit:
        """The unit of the time."""
        return self._unit

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value}, {self.unit})'

    def __str__(self) -> str:
        return f'{self.value}{self.unit.value}'

    def to_timestring(self):
        """Get the time-string representation.

        For example:

        >>> Time(5, TimeUnit.EPOCH).to_timestring()
        '5ep'

        Returns:
            str: The time-string representation.
        """
        return str(self)

    def _parse(self, other: object) -> Time:
        # parse ``other`` into a Time object
        if isinstance(other, Time):
            return other
        if isinstance(other, int):
            return Time(other, self.unit)
        if isinstance(other, str):
            other_parsed = Time.from_timestring(other)
            return other_parsed

        raise TypeError(f'Cannot convert type {other} to {self.__class__.__name__}')

    def _cmp(self, other: Union[int, float, Time, str]) -> int:
        # When doing comparisons, and other is an integer (or float), we can safely infer
        # the unit from self.unit
        # E.g. calls like this should be allowed: if batch < 42: do_something()
        # This eliminates the need to call .value everywhere
        if isinstance(other, (int, float)):
            other = type(self)(other, self.unit)
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(
                f'Cannot compare {self} to {other} since they have different units.')
        if self.value < other.value:
            return -1
        if self.value == other.value:
            return 0
        assert self.value > other.value
        return 1

    def __eq__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) == 0

    def __ne__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) != 0

    def __lt__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) < 0

    def __le__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) <= 0

    def __gt__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) > 0

    def __ge__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) >= 0

    def __add__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f'Cannot add {self} to {other} since they have different units.')
        return Time(self.value + other.value, self.unit)

    def __radd__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        return self + other

    def __sub__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(
                f'Cannot subtract {other} from {self} since they have different units.')
        return Time(self.value - other.value, self.unit)

    def __rsub__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        return (-self) + other

    def __neg__(self) -> Time[TValue]:
        return Time(cast(TValue, -self.value), self.unit)

    def __pos__(self) -> Time[TValue]:
        return Time(self.value, self.unit)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __truediv__(self, other: object) -> Time[float]:
        if isinstance(other, (float, int)):
            return Time(type(self.value)(self.value / other), self.unit)
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f'Cannot divide {self} by {other} since they have different units.')
        return Time(self.value / other.value, TimeUnit.DURATION)

    def __mul__(self, other: object):
        if isinstance(other, (float, int)):
            # Scale by the value.
            return Time(type(self.value)(self.value * other), self.unit)
        other = self._parse(other)
        if other.unit != TimeUnit.DURATION and self.unit != TimeUnit.DURATION:
            raise RuntimeError(f'Multiplication is supported only if one of the units is Duration')
        real_unit = self.unit if other.unit == TimeUnit.DURATION else other.unit
        real_type = float if real_unit == TimeUnit.DURATION else int
        return Time(real_type(self.value * other.value), real_unit)

    def __rmul__(self, other: object):
        return self * other

    def __hash__(self):
        return hash((self.value, self.unit))

    @classmethod
    def from_timestring(cls, timestring: str) -> Time:
        """Parse a time string into a :class:`Time` instance.

        A time string is a numerical value followed by the value of a :class:`TimeUnit` enum. For example:

        >>> Time.from_timestring("5ep")  # describes 5 epochs.
        Time(5, TimeUnit.EPOCH)
        >>> Time.from_timestring("3e4tok")  # describes 30,000 tokens.
        Time(30000, TimeUnit.TOKEN)
        >>> Time.from_timestring("0.5dur")  # describes 50% of the training process.
        Time(0.5, TimeUnit.DURATION)

        Returns:
            Time: An instance of :class:`Time`.
        """
        match = _TIME_STR_REGEX.findall(timestring)
        if len(match) != 1:
            raise ValueError(f'Invalid time string: {timestring}')
        match = match[0]
        match = [x for x in match if x != '']
        if len(match) != 2:
            raise ValueError(f'Each match should have a number followed by the key. Instead, ' +
                             f'got a match, {match}, of length {len(match)}.')
        value = match[0]
        unit = TimeUnit(match[1])
        value = float(value)  # always parsing first as float b/c it could be scientific notation
        if unit != TimeUnit.DURATION:
            if int(value) != value:
                raise TypeError(
                    f'value {value} is not an integer. Units {unit} require integer values.')
            value = int(value)
        return cls(value, unit)


def ensure_time(maybe_time: Union[Time, str, int], int_unit: Union[TimeUnit, str]) -> Time:
    """Ensure ``maybe_time`` is an instance of :class:`.Time`.

    Args:
        maybe_time (Time | str): A time string, integer, or instance of :class:`.Time`.
        int_unit (TimeUnit | str): The unit to use if ``maybe_time`` is an integer

    Returns:
        Time: An instance of :class:`.Time`.
    """
    if isinstance(maybe_time, str):
        return Time.from_timestring(maybe_time)
    if isinstance(maybe_time, int):
        return Time(maybe_time, int_unit)
    return maybe_time
