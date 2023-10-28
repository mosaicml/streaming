# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Survey the space of practical fixed-size decimal encodings."""

from argparse import ArgumentParser, Namespace

import numpy as np

from streaming.base.util.pretty import parse_strs


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--byte_widths',
        type=str,
        default='4,8,16',
        help='Comma-delited list of decimal type sizes in bytes',
    )
    args.add_argument(
        '--min_exp_range',
        type=int,
        default=32,
        help='Maximum number of possible decimal positions before we consider the exponent to be '
        + 'excessively high and skip it',
    )
    args.add_argument(
        '--max_exp_range',
        type=int,
        default=512,
        help='Maximum number of possible decimal positions before we consider the exponent to be '
        + 'excessively high and skip it',
    )
    return args.parse_args()


def survey(min_exp_range: int, max_exp_range: int, byte_width: int, is_signed: bool) -> None:
    """Survey possible fixed decimal types of a given byte width and signedness.

    We assume our decimal type will be serialized as:
    - Its digits as an integer, followed by
    - Its exponent (i.e., how many places to shift the decimal point).

    Specifically, for N-bit decimals which use K bits for the digits part:
    - Unsigned:
      - Int (K bits)
      - Exponent sign (1 bit)
      - Exponent magnitude (N - K - 1 bits)
    - Signed:
      - Int sign (1 bit)
      - Int (K bits)
      - Exponent sign (1vbit)
      - Exponent magnitude (N - 1 - K - 1 bits)

    Args:
        min_exp_range (int): Minimum number of possible decimal positions before we consider the
            exponent to be excessively low and skip it.
        min_exp_range (int): Maximum number of possible decimal positions before we consider the
            exponent to be excessively high and skip it.
        byte_width (int): Width of the serialized form in bits.
        is_signed (bool): Whether the number is signed.
    """
    header = 'int bits', 'pow bits', 'precision', 'dec range'
    table = [tuple(map(str, header))]
    bit_width = 8 * byte_width
    int_sign_bits = int(is_signed)
    prev_int_digits = None
    for int_bits in range(1, bit_width - int_sign_bits - 1 - 1):
        exp_bits = bit_width - int_sign_bits - int_bits - 1
        assert exp_bits
        max_int = 2**int_bits
        int_digits = int(np.log10(float(max_int)))
        exp_range = 2 * 2**exp_bits
        if not (min_exp_range <= exp_range <= max_exp_range):
            continue
        if int_digits == prev_int_digits:
            continue
        prev_int_digits = int_digits
        row = int_bits, exp_bits, int_digits, exp_range
        row = tuple(map(str, row))
        table.append(row)

    is_signed_str = 'signed' if is_signed else 'unsigned'
    print(f'{byte_width}-byte {is_signed_str} decimal:')
    cols = zip(*table)
    col_lens = [max(map(len, col)) for col in cols]
    for row in table:
        padded_row = [cell.rjust(col_len) for (cell, col_len) in zip(row, col_lens)]
        print('    ' + ' | '.join(padded_row))
    print()


def main(args: Namespace) -> None:
    """Survey the space of practical fixed-size decimal encodings.

    Args:
        Command-line arguments.
    """
    print('Columns:')
    print('- int bits: bits used to hold the digits as a scalar.')
    print('- pow bits: bits used to hold the exponent (location of decimal place).')
    print('- precision: Digits of precision.')
    print('- dec range: Range of decimal places (half left, half right).')
    print()

    byte_widths = list(map(int, parse_strs(args.byte_widths)))
    for byte_width in byte_widths:
        for is_signed in [False, True]:
            survey(args.min_exp_range, args.max_exp_range, byte_width, is_signed)


if __name__ == '__main__':
    main(parse_args())
