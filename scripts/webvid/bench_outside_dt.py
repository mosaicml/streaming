# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark iterating an MP4-outside MDS dataset."""

from argparse import ArgumentParser, Namespace
from time import time

import numpy as np

from streaming.multimodal.webvid import StreamingOutsideDTWebVid


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--local', type=str, required=True, help='Local')
    args.add_argument('--extra_local', type=str, required=True, help='Extra local')
    args.add_argument('--remote', type=str, required=True, help='Remote')
    args.add_argument('--extra_remote', type=str, required=True, help='Extra remote')
    args.add_argument('--log', type=str, required=True, help='Log')
    return args.parse_args()


def main(args: Namespace):
    """Benchmark iterating an MP4-outside MDS dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    dataset = StreamingOutsideDTWebVid(local=args.local,
                                       remote=args.remote,
                                       extra_local=args.extra_local,
                                       extra_remote=args.extra_remote)
    with open(args.log, 'wb') as out:
        t0 = time()
        for _ in dataset:
            t = time() - t0
            t = np.float32(t)
            out.write(t.tobytes())


if __name__ == '__main__':
    main(parse_args())
