# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate infinite samples for a 'saying numbers as words' task."""

from typing import Dict, List, Tuple, TypeVar

import numpy as np
from numpy.random import Generator
from tqdm import tqdm


def _generate_int(rng: Generator,
                  pos_prob: float = 0.75,
                  low: int = -1_000_000_000,
                  high: int = 1_000_000_000) -> int:
    """Pick a random integer to say in words.

    This is a synthetic dataset whose random numbers need to be distinct, deterministic given a
    seed, and little else. We choose a distribution that seems the most pleasing to us.

    Properties:
      * About 80% positive and 20% negative.
      * Magnitude of up to a billion on either side of zero.
      * Strongly skewed toward the origin, i.e. chosen uniformly across base-10 digit lengths (at
        least until running out of integers of that length anyway).

    Args:
        rng (Generator): NumPy random number generator.
        pos_prob (float): Probability of output being positive. Defaults to ``0.75``.
        low (int): Minimum of output range. Must be negative. Defaults to ``-1_000_000_000``.
        high (int): Maximum of output range. Must be positive. Defaults to ``1_000_000_000``.
    """
    if not 0 <= pos_prob <= 1:
        raise ValueError(f'Invalid positive probability ``pos_prob``: 0 <= {pos_prob} <= 1.')

    if not low < 0 < high:
        raise ValueError(f'Invalid sampling range ``low`` and/or ``high``: {low} < 0 < {high}.')

    is_pos = rng.uniform() < pos_prob
    max_digits = np.log10(high) if is_pos else np.log10(-low)
    exponent = rng.uniform(0, max_digits)
    magnitude = int(10**exponent)
    sign = is_pos * 2 - 1
    return sign * magnitude


def _generate_ints(count: int,
                   seed: int = 0x1337,
                   pos_prob: float = 0.75,
                   low: int = -1_000_000_000,
                   high: int = 1_000_000_000,
                   show_progress: bool = True) -> List[int]:
    """Sample until we have the given number of distinct integers.

    Args:
        count (int): How many samples to draw.
        seed (int): Seed for the random number generator. Defaults to ``0x1337``.
        pos_prob (float): Probability of output being positive. Defaults to ``0.75``.
        low (int): Minimum of output range. Must be negative. Defaults to ``-1_000_000_000``.
        high (int): Maximum of output range. Must be positive. Defaults to ``1_000_000_000``.
        show_progress (bool): Whether to display a progress bar. Defaults to ``True``.

    Returns:
        List[int]: The integers that were drawn.
    """
    rng = np.random.default_rng(seed)
    nums = set()
    progress_bar = tqdm(total=count, leave=False) if show_progress else None
    while len(nums) < count:
        num = _generate_int(rng)
        if num in nums:
            continue

        nums.add(num)
        if progress_bar:
            progress_bar.update(1)
    if progress_bar:
        progress_bar.close()

    nums = sorted(nums)
    rng.shuffle(nums)
    return nums


_ones = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
         'fifteen sixteen seventeen eighteen nineteen').split()

_tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def _int_to_words(num: int) -> List[str]:
    """Say an integer as a list of words.

    Args:
        num (int): The integer.

    Returns:
        List[str]: The integer as a list of words.
    """
    if num < 0:
        return ['negative'] + _int_to_words(-num)
    elif num <= 19:
        return [_ones[num]]
    elif num < 100:
        tens = [_tens[num // 10 - 2]]
        ones = [_ones[num % 10]] if num % 10 else []
        return tens + ones
    elif num < 1_000:
        hundreds = [_ones[num // 100], 'hundred']
        etc = _int_to_words(num % 100) if num % 100 else []
        return hundreds + etc
    elif num < 1_000_000:
        thousands = _int_to_words(num // 1_000) + ['thousand']
        etc = _int_to_words(num % 1_000) if num % 1_000 else []
        return thousands + etc
    elif num < 1_000_000_000:
        millions = _int_to_words(num // 1_000_000) + ['million']
        etc = _int_to_words(num % 1_000_000) if num % 1_000_000 else []
        return millions + etc
    else:
        raise ValueError('Integer out of range: -1,000,000,000 < {num} < +1,000,000,000.')


def _int_to_text(num: int) -> str:
    """Say an integer as text.

    Args:
        num (int): The integer.

    Returns:
        str: The integer as text.
    """
    words = _int_to_words(num)
    return ' '.join(words)


T = TypeVar('T')


def _split(items: List[T], sizes: List[int]) -> List[List[T]]:
    """Divide the given items across the splits given by their sizes.

    Args:
        items (List[Any]): The items to divide across the spans.
        sizes (List[int]): Number of items per split.

    Returns:
        List[List[Any]]: Each split of items.
    """
    total = sum(sizes)
    if len(items) != total:
        raise ValueError(f'Number of items must match the combined size of the splits: ' +
                         f'{len(items)} items vs splits of size {sizes} = {total}.')

    splits = []
    begin = 0
    for size in sizes:
        split = items[begin:begin + size]
        splits.append(split)
        begin += size

    return splits


def generate(split2size: Dict[str, int],
             seed: int = 0x1337,
             pos_prob: float = 0.75,
             low: int = -1_000_000_000,
             high: int = 1_000_000_000,
             show_progress: bool = True) -> Dict[str, Tuple[List[int], List[str]]]:
    """Generate a dataset, made of splits, to be saved in different forms for comparison.

    Args:
        split2size (Dict[str, int]): Mapping of split name to size in samples.
        seed (int): Seed for the random number generator. Defaults to ``0x1337``.
        pos_prob (float): Probability of output being positive. Defaults to ``0.75``.
        low (int): Minimum of output range. Must be negative. Defaults to ``-1_000_000_000``.
        high (int): Maximum of output range. Must be positive. Defaults to ``1_000_000_000``.
        show_progress (bool): Whether to show a progress bar. Defaults to ``True``.

    Returns:
        Dict[str, Tuple[List[int], List[str]]]: Mapping of split name to nums and texts.
    """
    split_sizes = []
    total = 0
    for split in sorted(split2size):
        size = split2size[split]
        split_sizes.append(size)
        total += size

    nums = _generate_ints(total, seed, low, high, show_progress)
    nums_per_split = _split(nums, split_sizes)

    texts = list(map(_int_to_text, nums))
    texts_per_split = _split(texts, split_sizes)

    dataset = {}
    for index, name in enumerate(sorted(split2size)):
        dataset[name] = nums_per_split[index], texts_per_split[index]

    return dataset
