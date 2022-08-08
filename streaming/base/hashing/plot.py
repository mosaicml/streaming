from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

algo2color: dict[str, Optional[str]] = {
    'blake2b': 'purple',
    'blake2s': 'purple',
    'md5': 'green',
    'sha1': 'yellow',
    'sha224': 'orange',
    'sha256': 'orange',
    'sha384': 'orange',
    'sha512': 'orange',
    'sha3_224': 'red',
    'sha3_256': 'red',
    'sha3_384': 'red',
    'sha3_512': 'red',
    'xxh32': 'cyan',
    'xxh64': 'cyan',
    'xxh128': 'cyan',
    'xxh3_64': 'blue',
    'xxh3_128': 'blue'
}


def parse_args() -> Namespace:
    """Parse commandline arguments.

    Args:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--data', type=str, required=True)
    args.add_argument('--dpi', type=int, default=300)
    args.add_argument('--font_size', type=int, default=5)
    args.add_argument('--line_width', type=float, default=0.5)
    args.add_argument('--hash_rates', type=str, default='hash_rates.png')
    return args.parse_args()


def load(f: str) -> list[tuple[str, int, float]]:
    """Load data CSV.

    Args:
        f (str): Filename.

    Returns:
        list[tuple[str, int, float]]: Tuples of (algo, size, time).
    """
    fp = open(f)
    next(fp)
    rr = []
    for s in fp:
        ss = s.strip().split(',')
        algo = ss[0]
        size = int(ss[1])
        time = float(ss[3])
        time = max(time, 1e-9)
        r = algo, size, time
        rr.append(r)
    return rr


def plot_hash_rates(data: list[tuple[str, int, float]], algo2color: dict[str, Optional[str]],
                    dpi: int, font_size: int, line_width: float, filename: str) -> None:
    """Plot hash rate by size.

    Args:
        data (list[tuple[str, int, float]]): Tuples of (algo, size, time).
        algo2color (dict[str, Optional[str]): Color of algo in plots, or None to omit.
        dpi (int): DPI of plots.
        font_size (int): Font size of plots.
        line_width (float): Line width.
        filename (str): Plot filename.
    """
    plt.style.use('dark_background')
    plt.rc('font', size=font_size)

    algo2sizes = defaultdict(list)
    algo2times = defaultdict(list)
    for algo, size, time in data:
        algo2sizes[algo].append(size)
        algo2times[algo].append(time)

    for algo in sorted(algo2sizes):
        sizes = algo2sizes[algo]
        times = algo2times[algo]
        ratios = np.array(sizes) / np.array(times)
        color = algo2color[algo]
        if color:
            plt.plot(sizes, ratios, c=color, label=algo, lw=line_width)

    plt.xscale('log')
    plt.yscale('log')

    plt.title('hash rate by size')
    plt.xlabel('size (bytes)')
    plt.ylabel('hash rate (bytes / sec)')
    plt.legend()

    plt.savefig(filename, dpi=dpi)
    plt.clf()


def main(args) -> None:
    """Plot info about hashing.

    Args:
        args (Namespace): Commandline arguments.
    """
    data = load(args.data)
    plot_hash_rates(data, algo2color, args.dpi, args.font_size, args.line_width, args.hash_rates)


if __name__ == '__main__':
    main(parse_args())
