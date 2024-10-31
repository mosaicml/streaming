# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark iterating an MP4-inside MDS dataset."""

from argparse import ArgumentParser, Namespace
from time import time

import numpy as np

from joshua.multimodal.webvid import StreamingInsideWebVid


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--local', type=str, required=True, help='Streaming dataset local')
    args.add_argument('--remote', type=str, required=True, help='Streaming dataset remote')
    args.add_argument('--log', type=str, required=True, help='Output log file')
    return args.parse_args()


def main(args: Namespace):
    """Benchmark iterating an MP4-inside MDS dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    dataset = StreamingInsideWebVid(local=args.local, remote=args.remote)
    tt = []
    t0 = time()
    for _ in dataset:
        t = time() - t0
        tt.append(t)
    tt = np.array(tt, np.float32)
    tt.tofile(args.log)


if __name__ == '__main__':
    main(parse_args())
