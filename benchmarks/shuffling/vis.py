# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize different shuffling algorithms on fake shards."""

import colorsys
from argparse import ArgumentParser, Namespace

import numpy as np

from streaming.shuffle import algos, get_shuffle


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--num_shards', type=int, default=100)
    args.add_argument('--base_samples_per_shard', type=int, default=20)
    args.add_argument('--num_canonical_nodes', type=int, default=16)
    args.add_argument('--seed', type=int, default=0xD06E)
    args.add_argument('--epoch', type=int, default=0)
    args.add_argument('--block_size', type=int, default=256)
    return args.parse_args()


def main(args: Namespace) -> None:
    """Visualize different shuffling algorithms on fake shards.

    Args:
        args (Namespace): Command-line arguments.
    """
    rng = np.random.default_rng(args.seed)
    shard_sizes = (args.base_samples_per_shard * 2**rng.uniform(0, 4, args.num_shards)).astype(
        np.int64)
    num_samples = sum(shard_sizes)
    starts = num_samples * np.arange(
        args.num_canonical_nodes * 8) // (args.num_canonical_nodes * 8)
    starts = num_samples * np.arange(args.num_canonical_nodes) // (args.num_canonical_nodes)

    colors = []
    for hue in np.linspace(0, 1, num_samples):
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        colors.append(rgb)
    colors = np.array(colors)
    colors = (255 * colors).astype(np.int64)

    print('''
<!doctype html>
<html>
<head>
<style type="text/css">
body {
    background: black;
}
table {
    background: black;
    padding: 5px;
    margin: 5px;
}
td {
    height: 10px
}
</style>
</head>
<body>
''')

    print('<center>')
    print('<div style="background: white; padding: 5px">')

    print('<div style="font-family: monospace; font-size: 200%; padding: 5px">')
    print('unshuffled')
    print('</div>')
    print('<table>')
    print('<tr>')
    for idx, sample_id in enumerate(np.arange(num_samples)):
        if idx in starts:
            print('</tr>')
            print('<tr>')
        red, green, blue = colors[sample_id]
        print(f'<td style="background: rgb({red}, {green}, {blue})"></td>')
    print('</tr>')
    print('</table>')

    for algo in sorted(algos):
        sample_ids = get_shuffle(algo, shard_sizes, args.num_canonical_nodes, args.seed,
                                 args.epoch, args.block_size)
        print('<div style="font-family: monospace; font-size: 200%; padding: 5px">')
        print(algo)
        print('</div>')
        print('<table>')
        print('<tr>')
        for idx, sample_id in enumerate(sample_ids):
            if idx in starts:
                print('</tr>')
                print('<tr>')
            red, green, blue = colors[sample_id]
            print(f'<td style="background: rgb({red}, {green}, {blue})"></td>')
        print('</tr>')
        print('</table>')
    print('</body>')
    print('</table>')

    print('</div>')
    print('</center>')

    print('</body>')
    print('</html>')


if __name__ == '__main__':
    main(parse_args())
