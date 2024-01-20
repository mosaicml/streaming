# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing import Process
from tempfile import TemporaryDirectory
from time import time
from typing import List, Type, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from streaming.base.coord.file.lock import SoftFileLock


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--tick_ms', type=str, default='1,2,4,8,16,32,64')
    args.add_argument('--num_procs', type=str, default='1,2,4,8,16,32,64')
    args.add_argument('--turns_per_proc', type=float, default=1024)
    args.add_argument('--turn_mean_ms', type=float, default=42)
    args.add_argument('--turn_std_ms', type=float, default=7)
    args.add_argument('--progress_bar', type=int, default=1)
    return args.parse_args()


domain2triple = {
    'neg': [True, False, False],
    'nonpos': [True, True, False],
    'negneg': [False, True, True],
    'pos': [False, False, True],
}

Number = Union[int, float]


def get_domain_index(value: Number) -> int:
    """Get the domain index of a number: 0 (neg), 1 (zero), or 2 (pos).

    Args:
        value (Number): Value.

    Returns:
        int: Domain index.
    """
    if value < 0:
        idx = 0
    elif value == 0:
        idx = 1
    else:
        idx = 2
    return idx


def check_number(
    name: str,
    domain: str,
    value: Number,
) -> None:
    """Check a number.

    Args:
        name (str): Argument command-line name.
        domain (str): Output argument value valid range: one of {neg, nonpos, nonneg, pos}.'
        value (Number): Parsed argument value.
    """
    is_ok = domain2triple[domain]
    idx = get_domain_index(value)
    if not is_ok[idx]:
        raise ValueError(f'Argument --{name} must be {domain}, but got: {value}.')


T = TypeVar('T', bound=Number)


def parse_number(
    name: str,
    kind: Type[T],
    domain: str,
    text: str,
) -> T:
    """Parse a number passed in a str command-line argument.

    Args:
        name (str): Argument command-line name.
        kind (Type[T]): Type of output argument values.
        domain (str): Output argument value valid range: one of {neg, nonpos, nonneg, pos}.'
        text (str): Argument value as a str.

    Returns:
        T: Output argument values.
    """
    value = kind(text)
    check_number(name, domain, value)
    return value


def parse_numbers(
    name: str,
    kind: Type[T],
    domain: str,
    text: str,
) -> List[T]:
    """Parse a list of numbers passed in a str command-line argument.

    Args:
        name (str): Argument command-line name.
        kind (Type[T]): Type of output argument values.
        domain (str): Output argument value valid range: one of {neg, nonpos, nonneg, pos}.'
        text (str): Argument value as a str.

    Returns:
        List[T]: Output argument values.
    """
    parse = partial(parse_number, name, kind, domain)
    texts = text.split(',') if text else []
    return list(map(parse, texts))


def get_lock_filename(dirname: str) -> str:
    return os.path.join(dirname, f'lock')


def get_times_filename(dirname: str, proc_id: int) -> str:
    return os.path.join(dirname, f'{proc_id}_times.npy')


def bench_process(
    dirname: str,
    tick: float,
    turns_per_proc: int,
    proc_id: int,
) -> None:
    lock_filename = get_lock_filename(dirname)
    lock = SoftFileLock(lock_filename, 60, tick)
    times = np.zeros(turns_per_proc + 1, np.float64)
    for i in range(turns_per_proc):
        times[i] = time()
        with lock:
            pass
    times[-1] = time()
    times_filename = get_times_filename(dirname, proc_id)
    times.tofile(times_filename)


def bench(
    dirname: str,
    tick: float,
    num_procs: int,
    turns_per_proc: int,
) -> NDArray[np.float64]:
    procs = []
    for proc_id in range(num_procs):
        proc = Process(target=bench_process, args=(dirname, tick, turns_per_proc, proc_id))
        procs.append(proc)

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    times = np.zeros((num_procs, turns_per_proc + 1), np.float64)
    for proc_id in range(num_procs):
        times_filename = get_times_filename(dirname, proc_id)
        times[proc_id] = np.fromfile(times_filename, np.float64)
    return times


def main(args: Namespace) -> None:
    """Main.

    Args:
        args (Namespace): Command-line arguments.
    """
    tick_mss = parse_numbers('tick_ms', float, 'pos', args.tick_ms)
    ticks = list(map(lambda tick_ms: tick_ms / 1000, tick_mss))
    proc_counts = parse_numbers('num_procs', int, 'pos', args.num_procs)
    turns_per_proc = parse_number('turns_per_proc', int, 'pos', args.turns_per_proc)

    durs = []
    with TemporaryDirectory() as dirname:
        assert os.path.isdir(dirname)
        total = len(proc_counts) * len(ticks)
        progress_bar = tqdm(total=total, leave=False) if args.progress_bar else None
        for tick in ticks:
            for num_procs in proc_counts:
                arr = bench(dirname, tick, num_procs, turns_per_proc)
                dur = arr.max() - arr.min()
                durs.append(dur)
                if progress_bar is not None:
                    progress_bar.update(1)
        assert os.path.isdir(dirname)
    durs = np.array(durs, np.float64)
    durs = durs.reshape(len(ticks), len(proc_counts))
    durs *= 1000
    print(durs)


if __name__ == '__main__':
    main(parse_args())
