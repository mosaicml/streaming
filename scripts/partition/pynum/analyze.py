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
    return args.parse_args()


def is_pattern(s: str, t: str) -> bool:
    """Tell whether the given text can be explained by the pattern."""
    n = len(s) // len(t) + 1
    s2 = ''.join([t] * n)[:len(s)]
    return s == s2


def get_pattern(s: str) -> Optional[str]:
    """Find the shortest pattern in the given text."""
    for i in range(1, len(s) // 2):
        t = s[:i]
        if is_pattern(s, t):
            return t


def normalize_pattern(t: str) -> str:
    """Rotate the pattern to the canonical position."""
    tt = []
    for i in range(len(t)):
        t2 = t[i:] + t[:i]
        tt.append(t2)
    tt.sort()
    return tt[-1]


def analyze(x: NDArray[np.int64]) -> None:
    """Search for a pattern to explain the given sequence."""
    # if not any(x):
    #     return

    s = ''.join(map(chr, x))
    t = get_pattern(s)
    if t:
        t = normalize_pattern(t)
        t = ''.join(map(lambda c: chr(ord(c) + ord('a')), t))
    xs = ''.join(map(lambda n: chr(n + ord('a')), x))
    print(f'        {xs[:40]} -> {t}')


def main(args: Namespace) -> None:
    """Find the patterns in required dataset padding.

    Args:
        args (Namespace): Command-line arguments.
    """
    data = np.load(getattr(args, 'in'), allow_pickle=True)
    num_c, num_p, num_r, num_w, num_b, _ = data.shape
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
            print(f'    p {p}')
            for ri in range(num_r):
                for wi in [0]:  # range(num_w):
                    for bi in [0]:  # range(num_b):
                        analyze(data[ci, pi, ri, wi, bi])
            print()


if __name__ == '__main__':
    main(parse_args())
