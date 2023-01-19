# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions to plot compression information."""

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt

algo2color = {
    'br': None,
    'br:0': '#fc0',
    'br:1': '#fb0',
    'br:2': '#fa1',
    'br:3': '#f92',
    'br:4': '#e73',
    'br:5': '#d54',
    'br:6': '#c35',
    'br:7': '#b26',
    'br:8': '#a17',
    'br:9': '#908',
    'br:10': '#809',
    'br:11': '#70a',
    'bz2': None,
    'bz2:1': '#08f',
    'bz2:2': '#07e',
    'bz2:3': '#06d',
    'bz2:4': '#05c',
    'bz2:5': '#04b',
    'bz2:6': '#03a',
    'bz2:7': '#029',
    'bz2:8': '#018',
    'bz2:9': '#017',
    'gz': None,
    'gz:0': None,  # Level 0 = no compression.
    'gz:1': '#0e1',
    'gz:2': '#0d2',
    'gz:3': '#0c3',
    'gz:4': '#0b4',
    'gz:5': '#0a5',
    'gz:6': '#096',
    'gz:7': '#087',
    'gz:8': '#078',
    'gz:9': '#069',
    'snappy': 'cyan',
    'zstd': None,
    'zstd:1': '#ffffff',
    'zstd:2': '#f0f0f0',
    'zstd:3': '#e0e0e0',
    'zstd:4': '#d0d0d0',
    'zstd:5': '#c0c0c0',
    'zstd:6': '#b0b0b0',
    'zstd:7': '#a0a0a0',
    'zstd:8': '#989898',
    'zstd:9': '#909090',
    'zstd:10': '#888888',
    'zstd:11': '#808080',
    'zstd:12': '#787878',
    'zstd:13': '#707070',
    'zstd:14': '#686868',
    'zstd:15': '#606060',
    'zstd:16': '#585858',
    'zstd:17': '#505050',
    'zstd:18': '#484848',
    'zstd:19': '#404040',
    'zstd:20': '#383838',
    'zstd:21': '#303030',
    'zstd:22': '#282828',
}


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Args:
        Namespace: command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--data', type=str, required=True)
    args.add_argument('--min_dec_size', type=int, default=8192)
    args.add_argument('--dpi', type=int, default=300)
    args.add_argument('--font_size', type=int, default=5)
    args.add_argument('--line_width', type=float, default=0.5)
    args.add_argument('--compression_rates', type=str, default='compression_rates.png')
    args.add_argument('--compressed_sizes', type=str, default='compressed_sizes.png')
    args.add_argument('--decompression_rates', type=str, default='decompression_rates.png')
    return args.parse_args()


@dataclass
class Datum(object):
    """A data point."""
    algo: str
    dec_size: int
    enc_size: int
    enc_time: float
    dec_time: float


def load(f: str, min_dec_size: int) -> List[Datum]:
    """Load data CSV.

    Args:
        f (str): Filename.
        min_dec_size (int): Minimum uncompressed data size to plot (smallest amounts of data show
            the highest spurious variability).

    Returns:
        List[Datum]: List of data points.
    """
    fp = open(f)
    next(fp)
    rr = []
    for s in fp:
        ss = s.strip().split(',')
        algo = ss[0]
        dec_size = int(ss[1])
        enc_size = int(ss[2])
        enc_time = float(ss[3])
        dec_time = float(ss[4])
        r = Datum(algo, dec_size, enc_size, enc_time, dec_time)
        if min_dec_size <= r.dec_size:
            rr.append(r)
    return rr


def plot_compression_rates(data: List[Datum], algo2color: Dict[str, Optional[str]], dpi: int,
                           font_size: float, line_width: float, filename: str) -> None:
    """Plot compression rate by size.

    Args:
        data (List[Datum]): List of data points.
        algo2color (Dict[str, Optional[str]): Color of algo in plots, or None to omit.
        dpi (int): DPI of plots.
        font_size (float): Font size of plots.
        line_width (float): Line width of plots.
        filename (str): Plot filename.
    """
    plt.style.use('dark_background')
    plt.rc('font', size=font_size)

    algo2dec_sizes = defaultdict(list)
    algo2enc_times = defaultdict(list)
    for datum in data:
        algo2dec_sizes[datum.algo].append(datum.dec_size)
        algo2enc_times[datum.algo].append(datum.enc_time)

    for algo in sorted(algo2dec_sizes):
        dec_sizes = algo2dec_sizes[algo]
        enc_times = algo2enc_times[algo]
        ratios = np.array(dec_sizes) / np.array(enc_times)
        color = algo2color[algo]
        if color:
            plt.plot(dec_sizes, ratios, c=color, label=algo, lw=line_width)

    plt.xscale('log')
    plt.yscale('log')

    plt.title('compression rate by size')
    plt.xlabel('size (bytes)')
    plt.ylabel('compression rate (bytes / sec)')
    plt.legend()

    plt.savefig(filename, dpi=dpi)
    plt.clf()


def plot_compressed_sizes(data: List[Datum], algo2color: Dict[str, Optional[str]], dpi: int,
                          font_size: float, line_width: float, filename: str) -> None:
    """Plot compressed size by size.

    Args:
        data (List[Datum]): List of data points.
        algo2color (Dict[str, Optional[str]): Color of algo in plots, or None to omit.
        dpi (int): DPI of plots.
        font_size (float): Font size of plots.
        line_width (float): Line width of plots.
        filename (str): Plot filename.
    """
    plt.style.use('dark_background')
    plt.rc('font', size=font_size)

    algo2dec_sizes = defaultdict(list)
    algo2enc_sizes = defaultdict(list)
    for datum in data:
        algo2dec_sizes[datum.algo].append(datum.dec_size)
        algo2enc_sizes[datum.algo].append(datum.enc_size)

    for algo in sorted(algo2dec_sizes):
        dec_sizes = algo2dec_sizes[algo]
        enc_sizes = algo2enc_sizes[algo]
        ratios = 100 * np.array(enc_sizes) / np.array(dec_sizes)
        color = algo2color[algo]
        if color:
            plt.plot(dec_sizes, ratios, c=color, label=algo, lw=line_width)

    plt.xscale('log')

    plt.title('compression rate by size')
    plt.xlabel('size (bytes)')
    plt.ylabel('compression rate (compressed / uncompressed)')
    plt.legend()

    plt.savefig(filename, dpi=dpi)
    plt.clf()


def plot_decompression_rates(data: List[Datum], algo2color: Dict[str, Optional[str]], dpi: int,
                             font_size: float, line_width: float, filename: str) -> None:
    """Plot decompression rate by size.

    Args:
        data (List[Datum]): List of data points.
        algo2color (Dict[str, Optional[str]): Color of algo in plots, or None to omit.
        dpi (int): DPI of plots.
        font_size (float): Font size of plots.
        line_width (float): Line width of plots.
        filename (str): Plot filename.
    """
    plt.style.use('dark_background')
    plt.rc('font', size=font_size)

    algo2dec_sizes = defaultdict(list)
    algo2dec_times = defaultdict(list)
    for datum in data:
        algo2dec_sizes[datum.algo].append(datum.dec_size)
        algo2dec_times[datum.algo].append(datum.dec_time)

    for algo in sorted(algo2dec_sizes):
        dec_sizes = algo2dec_sizes[algo]
        dec_times = algo2dec_times[algo]
        ratios = np.array(dec_sizes) / np.array(dec_times)
        color = algo2color[algo]
        if color:
            plt.plot(dec_sizes, ratios, c=color, label=algo, lw=line_width)

    plt.xscale('log')
    plt.yscale('log')

    plt.title('decompression rate by size')
    plt.xlabel('size (bytes)')
    plt.ylabel('decompression rate (bytes / sec)')
    plt.legend()

    plt.savefig(filename, dpi=dpi)
    plt.clf()


def main(args: Namespace) -> None:
    """Plot info about compression.

    Args:
        args (Namespace): command-line arguments.
    """
    data = load(args.data, args.min_dec_size)
    plot_compression_rates(data, algo2color, args.dpi, args.font_size, args.line_width,
                           args.compression_rates)
    plot_compressed_sizes(data, algo2color, args.dpi, args.font_size, args.line_width,
                          args.compressed_sizes)
    plot_decompression_rates(data, algo2color, args.dpi, args.font_size, args.line_width,
                             args.decompression_rates)


if __name__ == '__main__':
    main(parse_args())
