# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Collect data on the dataset padding needed by partitioning."""

from argparse import ArgumentParser, Namespace
from time import time
from typing import List

import numpy as np
from tqdm import tqdm

from streaming.base.partition import get_partitions


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('-a1', '--algo1', type=str, default='orig')
    args.add_argument('-a2', '--algo2', type=str, required=True)
    args.add_argument('-c', '--canonical_nodes', type=str, default='1-64')
    args.add_argument('-p', '--physical_nodes', type=str, default='1-64')
    args.add_argument('-r', '--ranks_per_node', type=str, default='1-8')
    args.add_argument('-w', '--workers_per_rank', type=str, default='1-8')
    args.add_argument('-b', '--batch_size', type=str, default='1-8')
    args.add_argument('-n', '--dataset_size', type=str, default='512-576')
    return args.parse_args()


def parse(text: str) -> list[int]:
    """Parse an int range.

    Args:
        text (str): String defining the int range.

    Returns:
        List[int]: List of ints.
    """
    ss = text.split(',')
    rr = []
    for s in ss:
        if '-' in s:
            a, b = map(int, s.split('-'))
            r = list(range(a, b + 1))
        else:
            r = [int(s)]
        rr += r
    return rr


def main(args: Namespace) -> None:
    """Collect data on the dataset padding needed by partitioning.

    Args:
        args (Namespace): Command-line arguments.
    """
    canonical_node_counts = parse(args.canonical_nodes)
    physical_node_counts = parse(args.physical_nodes)
    ranks_per_nodes = parse(args.ranks_per_node)
    workers_per_ranks = parse(args.workers_per_rank)
    batch_sizes = parse(args.batch_size)
    dataset_sizes = parse(args.dataset_size)

    shape = (len(canonical_node_counts), len(physical_node_counts), len(ranks_per_nodes),
             len(workers_per_ranks), len(batch_sizes), len(dataset_sizes))

    total = 0
    for c in canonical_node_counts:
        for p in physical_node_counts:
            if c < p:
                if p % c:
                    continue
            elif p < c:
                if c % p:
                    continue
            total += 1
    total *= np.prod(shape[2:])

    tt = []
    tt2 = []
    with tqdm(total=total, leave=False) as pbar:
        for c in canonical_node_counts:
            for p in physical_node_counts:
                if c < p:
                    if p % c:
                        continue
                elif p < c:
                    if c % p:
                        continue
                for r in ranks_per_nodes:
                    for w in workers_per_ranks:
                        for b in batch_sizes:
                            for n in dataset_sizes:
                                t0 = time()
                                ids = get_partitions(args.algo1, n, c, p, r, w, b, 0)
                                t = time() - t0
                                tt.append(t)
                                t0 = time()
                                ids2 = get_partitions(args.algo2, n, c, p, r, w, b, 0)
                                t2 = time() - t0
                                tt2.append(t2)
                                try:
                                    assert (ids == ids2).all()
                                except Exception as e:
                                    print(
                                        f'Error: c{c} p{p} r{r} w{w} b{b} n{n} {" ".join(str(e).split())}'
                                    )
                                    raise
                                pbar.update()
    t = sum(tt) / len(tt)
    print('%10.6f' % t)
    t2 = sum(tt2) / len(tt2)
    print('%10.6f' % t2)


if __name__ == '__main__':
    main(parse_args())
