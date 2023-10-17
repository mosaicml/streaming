# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a parquet dataset for testing."""

import os
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq
from tqdm import tqdm


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--num_train', type=int, default=1)
    args.add_argument('--num_val', type=int, default=1 << 26)
    args.add_argument('--dataset', type=str, default='data/parquet/')
    args.add_argument('--samples_per_shard', type=int, default=1 << 20)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


ones = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
         'fifteen sixteen seventeen eighteen nineteen').split()

tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def say(i: int) -> List[str]:
    """Get the word form of a number.

    Args:
        i (int): The number.

    Returns:
        List[str]: The number in word form.
    """
    if i < 0:
        return ['negative'] + say(-i)
    elif i <= 19:
        return [ones[i]]
    elif i < 100:
        return [tens[i // 10 - 2]] + ([ones[i % 10]] if i % 10 else [])
    elif i < 1_000:
        return [ones[i // 100], 'hundred'] + (say(i % 100) if i % 100 else [])
    elif i < 1_000_000:
        return say(i // 1_000) + ['thousand'] + (say(i % 1_000) if i % 1_000 else [])
    elif i < 1_000_000_000:
        return say(i // 1_000_000) + ['million'] + (say(i % 1_000_000) if i % 1_000_000 else [])
    else:
        raise ValueError('Integer must be less than a billion, but got: {i}')


def generate_number() -> int:
    """Generate a random integer to say.

    Returns:
        int: The integer.
    """
    sign = (np.random.uniform() < 0.8) * 2 - 1
    expt = np.random.uniform(0, 9)
    mag = int(10**expt)
    return sign * mag


def generate_numbers(num_train: int, num_val: int, use_tqdm: int) -> Tuple[List[int], List[int]]:
    """Get two non-overlapping splits of integers to say.

    Args:
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.
        use_tqdm (int): Whether to display a progress bar.

    Returns:
        Tuple[List[int], List[int]]: The two generated splits.
    """
    total = num_train + num_val
    nums = set()
    pbar = tqdm(total=total) if use_tqdm else None
    while len(nums) < total:
        num = generate_number()
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


def save_parquets(nums: List[int], txts: List[str], dirname: str, samples_per_shard: int) -> None:
    """Save a parquet dataaset given the samples.

    Args:
        nums (List[int]): List of sample integers.
        txts (List[str]): List of sample texts.
        dirname (str): Output dirname.
        samples_per_shard (int): Output shard size in samples.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    num_shards = (len(nums) + samples_per_shard - 1) // samples_per_shard
    for shard_id in range(num_shards):
        begin = shard_id * samples_per_shard
        end = min(begin + samples_per_shard, len(nums))
        shard_nums = nums[begin:end]
        shard_txts = txts[begin:end]
        filename = os.path.join(dirname, f'{shard_id:05}.parquet')
        obj = {
            'num': shard_nums,
            'txt': shard_txts,
        }
        table = pa.Table.from_pydict(obj)
        pq.write_table(table, filename)


def main(args: Namespace) -> None:
    """Generate a parquet dataset for testing.

    Args:
        args (Namespace): Command-line arguments.
    """
    train_nums, val_nums = generate_numbers(args.num_train, args.num_val, args.tqdm)

    train_txts = [' '.join(say(num)) for num in train_nums]
    val_txts = [' '.join(say(num)) for num in val_nums]

    dirname = os.path.join(args.dataset, 'train')
    save_parquets(train_nums, train_txts, dirname, args.samples_per_shard)

    dirname = os.path.join(args.dataset, 'val')
    save_parquets(val_nums, val_txts, dirname, args.samples_per_shard)


if __name__ == '__main__':
    main(parse_args())
