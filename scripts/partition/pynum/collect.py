# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Collect data on the dataset padding needed by partitioning."""

from argparse import ArgumentParser, Namespace
from typing import List

import numpy as np
from tqdm import tqdm

from streaming.base.partition.pynum import get_dataset_padding_brute


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('-o', '--out', type=str, required=True)
    args.add_argument('-c', '--canonical_nodes', type=str, default='1-16')
    args.add_argument('-p', '--physical_nodes', type=str, default='1-16')
    args.add_argument('-r', '--ranks_per_node', type=str, default='1-8')
    args.add_argument('-w', '--workers_per_rank', type=str, default='1-8')
    args.add_argument('-b', '--batch_size', type=str, default='1-8')
    args.add_argument('-n', '--dataset_size', type=str, default='1024-1535')
    return args.parse_args()


def parse(text: str) -> List[int]:
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

    x = np.zeros(shape, np.int8) - 1
    with tqdm(total=total, leave=False) as pbar:
        for ci, c in enumerate(canonical_node_counts):
            for pi, p in enumerate(physical_node_counts):
                if c < p:
                    if p % c:
                        continue
                elif p < c:
                    if c % p:
                        continue
                for ri, r in enumerate(ranks_per_nodes):
                    for wi, w in enumerate(workers_per_ranks):
                        for bi, b in enumerate(batch_sizes):
                            for ni, n in enumerate(dataset_sizes):
                                dp = get_dataset_padding_brute(n, c, p, r, w, b)
                                assert 0 <= dp <= 127
                                x[ci, pi, ri, wi, bi, ni] = dp
                                pbar.update()

    x.dump(args.out)


if __name__ == '__main__':
    main(parse_args())
