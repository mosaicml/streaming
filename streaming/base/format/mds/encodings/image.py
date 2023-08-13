# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Encode and decode images in MDS."""

from io import BytesIO

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile

from streaming.base.format.mds.encodings.base import Encoding

__all__ = ['PIL', 'JPEG', 'PNG']


class PIL(Encoding):
    """Store PIL image as raw CHW tensor.

    Format: [width: 4] [height: 4] [mode size: 4] [mode] [raw image].
    """

    def encode(self, obj: Image.Image) -> bytes:
        self._validate(obj, Image.Image)
        mode = obj.mode.encode('utf-8')
        width, height = obj.size
        raw = obj.tobytes()
        ints = np.array([width, height, len(mode)], np.uint32)
        return ints.tobytes() + mode + raw

    def decode(self, data: bytes) -> Image.Image:
        idx = 3 * 4
        width, height, mode_size = np.frombuffer(data[:idx], np.uint32)
        idx2 = idx + mode_size
        mode = data[idx:idx2].decode('utf-8')
        size = width, height
        raw = data[idx2:]
        return Image.frombytes(mode, size, raw)  # pyright: ignore


class JPEG(Encoding):
    """Store PIL image as JPEG."""

    def encode(self, obj: Image.Image) -> bytes:
        self._validate(obj, Image.Image)
        if isinstance(obj, JpegImageFile) and hasattr(obj, 'filename'):
            # Read the source file to prevent lossy re-encoding.
            with open(obj.filename, 'rb') as f:
                return f.read()
        else:
            out = BytesIO()
            obj.save(out, format='JPEG')
            return out.getvalue()

    def decode(self, data: bytes) -> Image.Image:
        inp = BytesIO(data)
        return Image.open(inp)


class PNG(Encoding):
    """Store PIL image as PNG."""

    def encode(self, obj: Image.Image) -> bytes:
        self._validate(obj, Image.Image)
        out = BytesIO()
        obj.save(out, format='PNG')
        return out.getvalue()

    def decode(self, data: bytes) -> Image.Image:
        inp = BytesIO(data)
        return Image.open(inp)
