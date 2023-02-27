# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Find the patterns in required dataset padding."""

from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in', type=str, required=True)
    args.add_argument('--over_w', type=int, default=False)
    args.add_argument('--over_b', type=int, default=False)
    return args.parse_args()


def is_pattern(text: str, pattern: str) -> bool:
    """Tell whether the given text can be explained by the pattern."""
    num_repeats = len(text) // len(pattern) + 1
    text2 = ''.join([pattern] * num_repeats)[:len(text)]
    return text == text2


def get_pattern(text: str) -> Optional[str]:
    """Find the shortest pattern in the given text.

    Args:
        text (str): Full text.

    Returns:
        Optional[str]: Pattern, if one is found.
    """
    for pattern_len in range(1, len(text) // 2):
        pattern = text[:pattern_len]
        if is_pattern(text, pattern):
            return pattern


def normalize_pattern(pattern: str) -> str:
    """Rotate the pattern to the canonical position.

    Args:
        pattern (str): Input pattern.

    Returns:
        str: Pattern in canonical form.
    """
    patterns = []
    for i in range(len(pattern)):
        pattern2 = pattern[i:] + pattern[:i]
        patterns.append(pattern2)
    patterns.sort()
    return patterns[-1]


def analyze(seq: NDArray[np.int64]) -> None:
    """Search for a pattern to explain the given sequence.

    Args:
        seq (NDArray[np.int64]): Sequence to analyze.
    """
    # if not any(seq):
    #     return

    text = ''.join(map(chr, seq))
    pattern = get_pattern(text) or ''
    if pattern:
        pattern = normalize_pattern(pattern)

    def chr_to_human(c: str) -> str:
        n = ord(c)
        if not n:
            return '.'
        elif n < 10:
            return chr(ord('0') + n)
        elif n < 36:
            return chr(ord('A') + n - 10)
        else:
            raise ValueError(f'Integer is too big for alphanumeric code: {n}.')

    def str_to_human(s: str) -> str:
        return ''.join(map(chr_to_human, s))

    human_text = str_to_human(text)
    human_pattern = str_to_human(pattern)
    print(f'        {human_text[:40]} -> {human_pattern}')


def main(args: Namespace) -> None:
    """Find the patterns in required dataset padding.

    Args:
        args (Namespace): Command-line arguments.
    """
    x = np.load(getattr(args, 'in'), allow_pickle=True)
    num_c, num_p, num_r, num_w, num_b, _ = x.shape

    for ci in range(num_c):
        c = 1 + ci
        print(f'c {c}')
        for pi in range(num_p):
            p = 1 + pi
            if c < p:
                if p % c:
                    continue
            elif p < c:
                if c % p:
                    continue
            print(f'    c {c}, p {p}')
            for ri in range(num_r):
                for wi in range(num_w) if args.over_w else [0]:
                    for bi in range(num_b) if args.over_b else [0]:
                        analyze(x[ci, pi, ri, wi, bi])
            print()

    print()

    xc = x.reshape(num_c, -1).max(1)
    print('Max over canonical nodes:', xc)

    xp = x.transpose(1, 0, 2, 3, 4, 5).reshape(num_p, -1).max(1)
    print('Max over physical nodes:', xp)

    xr = x.transpose(2, 0, 1, 3, 4, 5).reshape(num_r, -1).max(1)
    print('Max over ranks per node:', xr)

    xw = x.transpose(3, 0, 1, 2, 4, 5).reshape(num_w, -1).max(1)
    print('Max over workers per node:', xw)

    xb = x.transpose(4, 0, 1, 2, 3, 5).reshape(num_b, -1).max(1)
    print('Max over batch size per rank:', xb)

    print()

    x2 = x.reshape(num_c, num_p, -1).max(2)
    print('Max over (canonical nodes, physical nodes):')

    def dump(a: int) -> str:
        if a == -1:
            return ' .'
        else:
            return f'{a:2}'

    print('  ', ' ', ' '.join(map(dump, range(1, num_p + 1))))
    print('  ', ' ', ' '.join(['--' for _ in range(num_p)]))
    for i, aa in enumerate(x2):
        print(f'{i + 1:2}', '|', ' '.join(map(dump, aa)))

    print()

    x2 = x.reshape(num_c, num_p, num_r, -1).max(3)
    print('Max over (canonical nodes, physical nodes, ranks per node):')
    print()

    for ri in range(num_r):
        r = 1 + ri
        print(f'r {r}, c x p:')
        print('   ', '  ', ' ', ' '.join(map(dump, range(1, num_p + 1))))
        print('   ', '  ', ' ', ' '.join(['--' for _ in range(num_p)]))
        for i, aa in enumerate(x2[:, :, ri]):
            print('   ', f'{i + 1:2}', '|', ' '.join(map(dump, aa)))
        print()

    print()

    a = np.bincount(1 + x.flatten())
    a = 100 * a / a.sum()
    print('Distribution (-1 means this combination of c x p is invalid):')
    print('   ', ' '.join(map(lambda b: f'{b:3}     ', range(-1, len(a) - 1))))
    print('   ', ' '.join(map(lambda b: f'{b:7.3f}%', a)))
    print()

    x2 = x.reshape(num_c, num_p, -1).mean(2)
    print('Mean (canonical nodes, physical nodes):')

    def dump(a: float) -> str:
        if a == -1:
            return '  .   '
        else:
            return f'{a:6.3f}'

    print('  ', ' ', ' '.join(map(dump, range(1, num_p + 1))))
    print('  ', ' ', ' '.join(['------' for _ in range(num_p)]))
    for i, aa in enumerate(x2):
        print(f'{i + 1:2}', '|', ' '.join(map(dump, aa)))

    print()


if __name__ == '__main__':
    main(parse_args())
