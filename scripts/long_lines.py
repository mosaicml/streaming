# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Note long lines."""

import os
import re
import sys
from argparse import ArgumentParser, Namespace
from re import Pattern
from typing import IO, Iterator, Optional


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--root',
        type=str,
        default='.',
        help='Start with all the files under this directory.',
    )
    args.add_argument(
        '--include',
        type=str,
        default=r'.*\.py$',
        help='Drop all files whose paths fail to match this pattern, if given.',
    )
    args.add_argument(
        '--exclude',
        type=str,
        default='',
        help='Among the remaining files, drop any whose paths match this pattern, if given.',
    )
    args.add_argument(
        '--max_len',
        type=int,
        default=100,
        help='Maximum line length, excluding any trailing newline.',
    )
    args.add_argument(
        '--non_text',
        type=str,
        default='error',
        help='What to do if we encounter a binary (non-text) file while processing the files. ' +
        'Generally, this would suggest ``include`` was set too loose, or data corruption. ' +
        'Options: ``error``, ``warn``, ``ignore``.',
    )
    args.add_argument(
        '--color',
        type=str,
        default='light',
        help='Whether to output in color. Supported options: none, light.',
    )
    return args.parse_args()


non_text_behaviors = {'error', 'warn', 'ignore'}


def each_path(root: str,
              include: Optional[Pattern] = None,
              exclude: Optional[Pattern] = None) -> Iterator[str]:
    """Get each file path under root, in order, possibly included and excluded.

    Args:
        root (str): Evaluate for inclusion every file under the given root dir.
        include (Pattern, optional): First, check if the include pattern matches against ecah file
            path. If no include pattern was provided, we match all files. Defaults to ``None``.
        exclude (Pattern, optional): Second, for each of the included file paths, check if the
            exclude pattern matches it. If no exclude pattern, we do nothing. Defaults to ``None``.

    Returns:
       Iterator[str]: Each file path, in order.
    """
    for parent, _, file_basenames in os.walk(root):
        for basename in file_basenames:
            path = os.path.join(parent, basename)

            if include:
                if not include.match(path):
                    continue

            if exclude:
                if exclude.match(path):
                    continue

            yield path


def handle_non_text(behavior: str, path: str) -> None:
    """Handle having received a binary file instead of a text file.

    Args:
        behavior (str): Which non-text behavior to employ.
        path (str): Path to file.
    """
    if behavior == 'error':
        raise ValueError(f'Encountered non-text file: {path}.')
    elif behavior == 'warn':
        print(f'{path}:binary')
    elif behavior == 'ignore':
        pass
    else:
        txt = ', '.join(sorted(non_text_behaviors))
        raise ValueError(f'Unknown non-text behavior (must be one of: {txt}): {behavior}.')


def open_text(path: str, non_text_behavior: str = 'warn') -> Optional[IO[str]]:
    """Open the file as text (for reading line by line), with handling for binary files.

    Args:
        path (str): Path to text file.
        non_text_behavior (str): What to do when we got a binary file instead.

    Returns:
        IO[str], optional: On success, IO in mode 'r'.
    """
    try:
        return open(path)
    except:
        handle_non_text(non_text_behavior, path)


def drop_newline(line: str) -> str:
    """Remove the line's optional trailing newline.

    Args:
        line (str): Original line.

    Returns:
        str: Normalized line.
    """
    if line.endswith('\n'):
        return line[:-1]
    elif line.endswith('\r\n'):
        return line[:-2]
    else:
        return line


def main(args: Namespace) -> int:
    """Note long lines.

    Args:
        args (Namespace): Command-line arguments.
    """
    colors = ['none', 'light']
    if args.color not in colors:
        raise ValueError('Color option must be one of {colors}, but got: {args.color}.')

    include = re.compile(args.include) if args.include else None
    exclude = re.compile(args.exclude) if args.exclude else None

    if args.max_len < 0:
        raise ValueError(f'max_len must be non-negative, but got: {args.max_len}')

    if args.non_text not in non_text_behaviors:
        txt = ', '.join(sorted(non_text_behaviors))
        raise ValueError(f'Unknown non-text behavior (must be one of: {txt}): {args.non_text}.')

    count = 0
    for path in sorted(each_path(args.root, include, exclude)):
        if not (file := open_text(path, args.non_text)):
            continue

        lines = map(drop_newline, file)
        for line_no, line in enumerate(lines):
            if args.max_len < len(line):
                good_line = line[:args.max_len]
                bad_line = line[args.max_len:]
                if args.color == 'light':
                    path = f'\033[0;97m{path}\033[0;0m'
                    line_no = f'\033[0;92m{line_no}\033[0;0m'
                    good_line = f'\033[0;94m{good_line}\033[0;0m'
                    bad_line = f'\033[0;91m{bad_line}\033[0;0m'
                print(f'{path}:{line_no}:{good_line}{bad_line}')
                count += 1

    return 1 if count else 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
