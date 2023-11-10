# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark iterating an MP4-outside MDS dataset."""

from argparse import ArgumentParser, Namespace
from time import time

import numpy as np

from examples.multimodal.webvid.read import StreamingOutsideDTWebVid


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--local', type=str, required=True, help='Streaming dataset local',)
    args.add_argument('--extra_local',
                      type=str,
                      required=True,
                      help='Streaming dataset extra local')
    args.add_argument('--remote', type=str, required=True, help='Streaming dataset remote',)
    args.add_argument('--extra_remote',
                      type=str,
                      required=True,
                      help='Streaming dataset extra remote')
    args.add_argument('--log', type=str, required=True, help='Output log file')
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
    tt = []
    t0 = time()
    for _ in dataset:
        t = time() - t0
        tt.append(t)
    tt = np.array(tt, np.float32)
    tt.tofile(args.log)


if __name__ == '__main__':
    main(parse_args())
