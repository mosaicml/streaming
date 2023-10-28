# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate infinite samples for a 'saying numbers as words' task."""

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

_ones = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
         'fifteen sixteen seventeen eighteen nineteen').split()

_tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def _say(i: int) -> List[str]:
    """Get the word form of a number.

    Args:
        i (int): The number.

    Returns:
        List[str]: The number in word form.
    """
    if i < 0:
        return ['negative'] + _say(-i)
    elif i <= 19:
        return [_ones[i]]
    elif i < 100:
        return [_tens[i // 10 - 2]] + ([_ones[i % 10]] if i % 10 else [])
    elif i < 1_000:
        return [_ones[i // 100], 'hundred'] + (_say(i % 100) if i % 100 else [])
    elif i < 1_000_000:
        return _say(i // 1_000) + ['thousand'] + (_say(i % 1_000) if i % 1_000 else [])
    elif i < 1_000_000_000:
        return _say(i // 1_000_000) + ['million'] + (_say(i % 1_000_000) if i % 1_000_000 else [])
    else:
        raise ValueError('Integer must be less than a billion, but got: {i}')


def _generate_number() -> int:
    """Generate a random integer to say.

    Returns:
        int: The integer.
    """
    sign = (np.random.uniform() < 0.8) * 2 - 1
    expt = np.random.uniform(0, 9)
    mag = int(10**expt)
    return sign * mag


def _generate_numbers(num_train: int, num_val: int,
                      show_progress: bool) -> Tuple[List[int], List[int]]:
    """Get two non-overlapping splits of integers to say.

    Args:
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.
        show_progress (bool): Whether to display a progress bar.

    Returns:
        Tuple[List[int], List[int]]: The two generated splits.
    """
    total = num_train + num_val
    nums = set()
    pbar = tqdm(total=total, leave=False) if show_progress else None
    while len(nums) < total:
        num = _generate_number()
        if num in nums:
            continue
        nums.add(num)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()
    nums = sorted(nums)
    np.random.shuffle(nums)
    train_nums = nums[:num_train]
    val_nums = nums[num_train:]
    return train_nums, val_nums


_split_type = Tuple[str, List[int], List[str]]


def generate_dataset(num_train: int, num_val: int, show_progress: bool) -> List[_split_type]:
    """Generate the dataset, which will be saved in different forms for comparison.

    Args:
        num_train (int): Number of train samples.
        num_val (int): Number of val samples.
        show_progress (bool): Whether to show a progress bar.

    Returns:
        List[Tuple[str, List[int], List[str]]]: List of dataset splits.
    """
    train_nums, val_nums = _generate_numbers(num_train, num_val, show_progress)

    train_txts = [' '.join(_say(num)) for num in train_nums]
    val_txts = [' '.join(_say(num)) for num in val_nums]

    return [
        ('train', train_nums, train_txts),
        ('val', val_nums, val_txts),
    ]
