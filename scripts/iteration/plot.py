# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Plot dataset iteration time."""

import json
from argparse import ArgumentParser, Namespace
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--stats', type=str, required=True)
    args.add_argument('--plot', type=str, required=True)
    return args.parse_args()


def get_color(key: str, pq_mds_colors: Dict[str, str], lance_colors: Dict[int, str]) -> str:
    """Get a plot color for a given statistic key.

    Args:
        key (str): The statistic key.
        pq_mds_colors (Dict[str, str]): Mapping of PQ/MDS type to color.
        lance_colors (Dict[int, str]): Mapping of Lance take count to color.

    Returns:
        str: Color.
    """
    parts = key.split('_')
    first = parts[0]
    if first in {'pq', 'mds'}:
        kind = '_'.join(parts[:-1])
        color = pq_mds_colors[kind]
    elif first == 'lance':
        take_count = int(parts[-1])
        color = lance_colors[take_count]
    else:
        raise ValueError(f'Unknown type of key: {key}.')
    return color


def main(args: Namespace) -> None:
    """Randomly iterate over a Parquet dataset with Streaming.

    Args:
        args (Namespace): Command-line arguments.
    """
    pq_mds_colors = {'pq': 'green', 'pq_mds': 'blue', 'mds': 'red'}

    lance_take_counts = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    lance_colors = '#730', '#840', '#950', '#a60', '#b70', '#c80', '#d90', '#ea0', '#fb1', \
        '#fc4', '#fd7'
    lance_colors = dict(zip(lance_take_counts, lance_colors))

    stats = json.load(open(args.stats))

    plt.rc('legend', fontsize=6)
    plt.title('Time to iterate')
    plt.xlabel('Seconds')
    plt.ylabel('Samples')
    line_width = 0.75

    for key in sorted(stats):
        stat = stats[key]
        times = np.array(stat['times']) / 1e9
        color = get_color(key, pq_mds_colors, lance_colors)
        line_style = '-' if 'seq' in key else ':'
        label = stat['label']
        plt.plot(times, np.arange(len(times)), c=color, ls=line_style, lw=line_width, label=label)

    plt.legend()
    plt.grid(which='major', ls='--', c='#ddd')
    plt.savefig(args.plot, dpi=500)


if __name__ == '__main__':
    main(parse_args())
