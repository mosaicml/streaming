# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Bytes DBS type."""

from io import BytesIO
from typing import Tuple as TyTuple

import numpy as np
from PIL import Image as image_module
from PIL.Image import Image as TyImage

from streaming.base.format.dbs.type.base import SimpleVarLeaf, decode_int, prepend_int


class Image(SimpleVarLeaf):
    """Image DBS type abstract base class."""

    py_type = TyImage

    def encode(self, obj: TyImage) -> bytes:
        raise NotImplementedError

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TyImage, int]:
        raise NotImplementedError


class RawImage(Image):
    """Raw image DBS type.

    Notes:
    - A custom serialization format that stores an uncompressed dump of the image data.
    - It is very, very inefficient for images of any size.
    - This is the "best" option when your images are less than about 40 pixels square.
    """

    def encode(self, obj: TyImage) -> bytes:
        mode = obj.mode.encode('utf-8')
        width, height = obj.size
        raw = obj.tobytes()
        ints = np.array([width, height, len(mode)], np.uint32)
        data = ints.tobytes() + mode + raw
        return prepend_int(np.uint32, len(data), data)

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TyImage, int]:
        size, offset = decode_int(data, offset, np.uint32)
        ints_size = 3 * np.uint32().nbytes
        width, height, mode_size = np.frombuffer(data[offset:offset + ints_size], np.uint32)
        offset += ints_size
        mode = data[offset:offset + mode_size]
        offset += mode_size
        mode = mode.decode('utf-8')
        raw = data[offset:offset + size]
        image = image_module.frombytes(mode, (width, height), raw)
        return image, offset + size


class FmtImage(Image):
    """Standard image format DBS type abstract base class."""

    format: str = ''

    def encode(self, obj: TyImage) -> bytes:
        if hasattr(obj, 'filename'):
            filename = getattr(obj, 'filename')
            with open(filename, 'rb') as fp:
                data = fp.read()
        else:
            out = BytesIO()
            obj.save(out, format=self.format)
            data = out.getvalue()
        return prepend_int(np.uint32, len(data), data)

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TyImage, int]:
        size, offset = decode_int(data, offset, np.uint32)
        buf = BytesIO(data[offset:offset + size])
        img = image_module.open(buf)
        return img, offset + size


class JPG(FmtImage):
    """JPG DBS type."""

    format = 'JPEG'


class PNG(FmtImage):
    """PNG DBS type."""

    format = 'PNG'
