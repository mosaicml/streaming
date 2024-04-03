import numpy as np
from typing import Any

from streaming.base.format.mds.encodings import Encoding, _encodings

class Complex128(Encoding):

    def encode(self, obj: Any) -> bytes:
        return np.complex128(obj).tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.complex128)[0]

_encodings['complex128'] = Complex128