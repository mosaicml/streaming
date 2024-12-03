# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Plot results of comparing streaming dataset shuffling algorithms."""

from argparse import ArgumentParser, Namespace

from matplotlib import pyplot as plt

mark_sizes = [
    ('CIFAR-10 (50K)', 50_000, '#dfd'),
    ('ImageNet (1.2M)', 1_281_167, '#cfc'),
    ('C4 (365M)', 364_868_892, '#8c8'),
    ('LAION-5B (5.85B)', 5_850_000_000, '#484'),
]

mark_times = [
    ('Instant (1s)', 1, '#bdf'),
    ('Fast (10s)', 10, '#9ce'),
    ('Slow (60s)', 60, '#08f'),
]

algo_colors = {
    'naive': 'black',
    'py1e': 'green',
    'py1br': 'orange',
    'py2s': 'purple',
    'py1s': 'red',
}


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in', type=str, required=True, help='Shuffle benchmarking results')
    args.add_argument('--out', type=str, required=True, help='Shuffle benchmarking plot')
    return args.parse_args()


def load(filename: str) -> tuple[list[int], dict[str, list[float]]]:
    """Load dataset sizes and shuffle times generated by benchmarking.

    Args:
        filename (str): Path to data file.

    Returns:
        Tuple[List[int], Dict[str, List[float]]]: Sizes and algorithm time curves.
    """
    fp = open(filename)
    line = next(fp)
    keys = line.split()[2:]
    sizes = []
    tuples = []
    for line in fp:
        ss = line.split()
        size = int(ss[1].replace(',', ''))
        sizes.append(size)
        times = list(map(float, ss[2:]))
        tuples.append(times)
    curves = tuple(zip(*tuples))
    key2times = dict(zip(keys, curves))
    return sizes, key2times  # pyright: ignore


def main(args: Namespace) -> None:
    """Plot results of comparing streaming dataset shuffling algorithms.

    Args:
        args (Namespace): Command-line arguments.
    """
    sizes, key2times = load(getattr(args, 'in'))

    min_size = min(sizes)
    max_size = max(sizes)
    min_time = min(map(min, key2times.values()))
    max_time = max(map(max, key2times.values()))

    for label, size, color in mark_sizes:
        plt.plot([size, size], [min_time, max_time], c=color, label=label)

    for label, time, color in mark_times:
        plt.plot([min_size, max_size], [time, time], c=color, label=label)

    for key in sorted(key2times):
        times = key2times[key]
        times = list(filter(lambda t: 0 < t, times))
        sub_sizes = sizes[:len(times)]
        color = algo_colors[key]
        plt.plot(sub_sizes, times, c=color, label=key)

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Shuffle Time vs Dataset Size')
    plt.xlabel('Dataset size (samples)')
    plt.ylabel('Shuffle time (seconds)')
    plt.legend()
    plt.grid(which='major', ls='--', c='#ddd')
    plt.grid(which='minor', ls=':', c='#eee')
    plt.savefig(args.out, dpi=500)


if __name__ == '__main__':
    main(parse_args())
