# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Line by line text table printer."""

from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import Self

__all__ = ['Tabulator']


class Tabulator:
    """Line by line text table printer.

    Example:
        conf = '''
            < format 8
            > sec 6
            > samples 12
            > usec/sp 8
            > bytes 14
            > files 6
            > bytes/file 12
            > max bytes/file 14
        '''
        left = 4 * ' '
        tab = Tabulator.from_conf(conf, left)

    Args:
        cols (List[Tuple[str, str, int]]: Each column config (i.e., just, name, width).
        left (str, optional): Print this before each line (e.g., indenting). Defaults to ``None``.
    """

    def __init__(self, cols: List[Tuple[str, str, int]], left: Optional[str] = None) -> None:
        self.cols = cols
        self.col_justs = []
        self.col_names = []
        self.col_widths = []
        for just, name, width in cols:
            if just not in {'<', '>'}:
                raise ValueError(f'Invalid justify (must be one of "<" or ">"): {just}.')

            if not name:
                raise ValueError('Name must be non-empty.')
            elif width < len(name):
                raise ValueError(f'Name is too wide for its column width: {width} vs {name}.')

            if width <= 0:
                raise ValueError(f'Width must be positive, but got: {width}.')

            self.col_justs.append(just)
            self.col_names.append(name)
            self.col_widths.append(width)

        self.left = left

        self.box_chr_horiz = chr(0x2500)
        self.box_chr_vert = chr(0x2502)

    @classmethod
    def from_conf(cls, conf: str, left: Optional[str] = None) -> Self:
        """Initialize a Tabulator from a text table defining its columns.

        Args:
            conf (str): The table config.
            left (str, optional): Optional string that is printed before each line (e.g., indents).
        """
        cols = []
        for line in conf.strip().split('\n'):
            words = line.split()

            if len(words) < 3:
                raise ValueError(f'Invalid col config (must be "just name width"): {line}.')

            just = words[0]
            name = ' '.join(words[1:-1])
            width = int(words[-1])
            cols.append((just, name, width))
        return cls(cols, left)

    def draw_row(self, row: Dict[str, Any]) -> str:
        """Draw a row, given a mapping of column name to field value.

        Args:
            row (Dict[str, Any]): Mapping of column name to field value.

        Returns:
            str: Text line.
        """
        fields = []
        for just, name, width in self.cols:
            val = row[name]

            txt = val if isinstance(val, str) else str(val)
            if width < len(txt):
                raise ValueError(f'Field is too wide for its column: column (just: {just}, ' +
                                 f'name: {name}, width: {width}) vs field {txt}.')

            txt = txt.ljust(width) if just == '<' else txt.rjust(width)
            fields.append(txt)

        left_txt = self.left or ''
        fields_txt = f' {self.box_chr_vert} '.join(fields)
        return f'{left_txt}{self.box_chr_vert} {fields_txt} {self.box_chr_vert}'

    def draw_header(self) -> str:
        """Draw a header row.

        Returns:
            str: Text line.
        """
        row = dict(zip(self.col_names, self.col_names))
        return self.draw_row(row)

    def draw_line(self) -> str:
        """Draw a divider row.

        Returns:
            str: Text line.
        """
        seps = (self.box_chr_horiz * width for width in self.col_widths)
        row = dict(zip(self.col_names, seps))
        line = self.draw_row(row)
        return line.replace(self.box_chr_vert, self.box_chr_horiz)
