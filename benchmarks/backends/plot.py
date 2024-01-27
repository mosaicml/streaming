# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Plot dataset iteration time."""

import json
from argparse import ArgumentParser, Namespace

import numpy as np
from matplotlib import pyplot as plt


def _parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--stats', type=str, default='data/backends/stats.json')
    args.add_argument('--plot', type=str, default='data/backends/plot.png')
    return args.parse_args()


def main(args: Namespace) -> None:
    """Randomly iterate over a Parquet dataset with Streaming.

    Args:
        args (Namespace): Command-line arguments.
    """
    streaming_colors = {
        'csv': '#c00',
        'jsonl': '#a00',
        'mds': '#800',
    }

    parquet_colors = {
        'native': 'green',
        'cold': 'blue',
        'warm': 'red',
    }

    lance_take_counts = 2**np.arange(11)
    lance_colors = '#730', '#840', '#950', '#a60', '#b70', '#c80', '#d90', '#ea0', '#fb1', \
        '#fc4', '#fd7'
    lance_colors = dict(zip(map(str, lance_take_counts), lance_colors))

    colors = {
        'streaming': streaming_colors,
        'parquet': parquet_colors,
        'lance': lance_colors,
    }

    stats = json.load(open(args.stats))

    plt.rc('legend', fontsize=5)
    plt.title('Throughput')
    plt.xlabel('Seconds')
    plt.ylabel('Samples')
    line_width = 0.75

    for backend in sorted(colors):
        keys = sorted(colors[backend])
        if backend == 'lance':
            keys = sorted(map(int, keys))
            keys = list(map(str, keys))
        for key in keys:
            for ordering in ['seq', 'rand']:
                color = colors[backend][key]
                try:
                    obj = stats[backend][key][ordering]
                except:
                    continue
                times = np.array(obj['times']) / 1e9
                line_style = '-' if ordering == 'seq' else ':'
                label = obj['label']
                plt.plot(times,
                         np.arange(len(times)),
                         c=color,
                         ls=line_style,
                         lw=line_width,
                         label=label)

    plt.legend()
    plt.grid(which='major', ls='--', c='#ddd')
    plt.savefig(args.plot, dpi=600)


if __name__ == '__main__':
    main(_parse_args())
