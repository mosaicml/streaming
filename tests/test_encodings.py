# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from decimal import Decimal
from typing import Any, Tuple, Union

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

import streaming.base.format.json.encodings as jsonEnc
import streaming.base.format.mds.encodings as mdsEnc
import streaming.base.format.xsv.encodings as xsvEnc


class TestMDSEncodings:

    @pytest.mark.parametrize('data', [b'5', b'\x00\x00'])
    def test_byte_encode_decode(self, data: bytes):
        byte_enc = mdsEnc.Bytes()
        assert byte_enc.size is None
        output = byte_enc.encode(data)
        assert output == data
        output = byte_enc.decode(data)
        assert output == data

    @pytest.mark.parametrize('data', ['9', 25])
    def test_byte_encode_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            byte_enc = mdsEnc.Bytes()
            _ = byte_enc.encode(data)

    @pytest.mark.parametrize(('data', 'encode_data'),
                             [('99', b'99'), ('streaming dataset', b'streaming dataset')])
    def test_str_encode_decode(self, data: str, encode_data: bytes):
        str_enc = mdsEnc.Str()
        assert str_enc.size is None

        # Test encode
        enc_data = str_enc.encode(data)
        assert isinstance(enc_data, bytes)
        assert enc_data == encode_data

        # Test decode
        dec_data = str_enc.decode(encode_data)
        assert isinstance(dec_data, str)
        assert dec_data == data

    @pytest.mark.parametrize('data', [b'9', 25])
    def test_str_encode_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            str_enc = mdsEnc.Str()
            _ = str_enc.encode(data)

    @pytest.mark.parametrize(('data', 'encode_data'), [(99, b'c\x00\x00\x00\x00\x00\x00\x00'),
                                                       (987654321, b'\xb1h\xde:\x00\x00\x00\x00')])
    def test_int_encode_decode(self, data: int, encode_data: bytes):
        int_enc = mdsEnc.Int()
        assert int_enc.size == 8

        # Test encode
        enc_data = int_enc.encode(data)
        assert isinstance(enc_data, bytes)
        assert enc_data == encode_data

        # Test decode
        dec_data = int_enc.decode(encode_data)
        assert isinstance(dec_data, int)
        assert dec_data == data

    @pytest.mark.parametrize('data', [b'9', 25.9])
    def test_int_encode_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            int_enc = mdsEnc.Int()
            _ = int_enc.encode(data)

    @pytest.mark.parametrize('dtype_str', [
        'uint8',
        'uint16',
        'uint32',
        'uint64',
        'int8',
        'int16',
        'int32',
        'int64',
        'float16',
        'float32',
        'float64',
    ])
    @pytest.mark.parametrize('shape', [
        (1,),
        (1, 1),
        (2, 3, 4, 5),
        (1, 3, 1, 3, 1),
        (300,),
        (1, 256, 1),
        (65536, 7, 1),
    ])
    def test_ndarray_encode_decode(self, dtype_str: str, shape: Tuple[int]):
        dtype = getattr(np, dtype_str)
        a = np.random.randint(0, 1000, shape).astype(dtype)

        encoding = 'ndarray'
        assert mdsEnc.is_mds_encoding(encoding)
        assert mdsEnc.get_mds_encoded_size(encoding) is None
        b = mdsEnc.mds_encode(encoding, a)
        c = mdsEnc.mds_decode(encoding, b)
        assert (a == c).all()
        b1_len = len(b)

        encoding = f'ndarray:{dtype.__name__}'
        assert mdsEnc.is_mds_encoding(encoding)
        assert mdsEnc.get_mds_encoded_size(encoding) is None
        b = mdsEnc.mds_encode(encoding, a)
        c = mdsEnc.mds_decode(encoding, b)
        assert (a == c).all()
        b2_len = len(b)

        shape_str = ','.join(map(str, shape))
        encoding = f'ndarray:{dtype.__name__}:{shape_str}'
        assert mdsEnc.is_mds_encoding(encoding)
        b_size = mdsEnc.get_mds_encoded_size(encoding)
        assert b_size is not None
        b = mdsEnc.mds_encode(encoding, a)
        c = mdsEnc.mds_decode(encoding, b)
        assert (a == c).all()
        assert len(b) == b_size
        b3_len = len(b)

        assert b3_len < b2_len < b1_len
        assert b3_len == np.prod(shape) * dtype().nbytes

    def test_error_no_elements_ndarray(self):
        encoding = 'ndarray'
        with pytest.raises(ValueError,
                           match='Attempting to encode a numpy array with 0 elements.*'):
            _ = mdsEnc.mds_encode(encoding, np.array([]))

    @pytest.mark.parametrize('array', [np.array(0.5), np.empty(()), np.array(1)])
    def test_error_scalar_ndarray(self, array: NDArray):
        encoding = 'ndarray'
        with pytest.raises(ValueError,
                           match='Attempting to encode a scalar with NDArray encoding.*'):
            _ = mdsEnc.mds_encode(encoding, array)

    @pytest.mark.parametrize('mode', ['I', 'L', 'RGB'])
    def test_pil_encode_decode(self, mode: str):
        pil_enc = mdsEnc.PIL()
        assert pil_enc.size is None

        # Creating the (32 x 32) NumPy Array with random values
        np_data = np.random.randint(255, size=(32, 32), dtype=np.uint32)
        # Default image mode of PIL Image is 'I'
        img = Image.fromarray(np_data).convert(mode)

        # Test encode
        enc_data = pil_enc.encode(img)
        assert isinstance(enc_data, bytes)

        # Test decode
        dec_data = pil_enc.decode(enc_data)
        dec_data = dec_data.convert('I')
        np_dec_data = np.asarray(dec_data, dtype=np.uint32)
        assert isinstance(dec_data, Image.Image)

        # Validate data content
        assert np.array_equal(np_data, np_dec_data)

    @pytest.mark.parametrize('data', [b'9', 25.9])
    def test_pil_encode_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            pil_enc = mdsEnc.PIL()
            _ = pil_enc.encode(data)

    @pytest.mark.parametrize('mode', ['L', 'RGB'])
    def test_jpeg_encode_decode(self, mode: str):
        jpeg_enc = mdsEnc.JPEG()
        assert jpeg_enc.size is None

        # Creating the (32 x 32) NumPy Array with random values
        np_data = np.random.randint(255, size=(32, 32), dtype=np.uint32)
        # Default image mode of PIL Image is 'I'
        img = Image.fromarray(np_data).convert(mode)

        # Test encode
        enc_data = jpeg_enc.encode(img)
        assert isinstance(enc_data, bytes)

        # Test decode
        dec_data = jpeg_enc.decode(enc_data)
        dec_data = dec_data.convert('I')
        assert isinstance(dec_data, Image.Image)

    @pytest.mark.parametrize('mode', ['L', 'RGB'])
    def test_jpegfile_encode_decode(self, mode: str):
        jpeg_enc = mdsEnc.JPEG()
        assert jpeg_enc.size is None

        # Creating the (32 x 32) NumPy Array with random values
        size = {'RGB': (224, 224, 3), 'L': (28, 28)}[mode]
        np_data = np.array(np.random.randint(255, size=size, dtype=np.uint8))
        # Default image mode of PIL Image is 'I'
        img = Image.fromarray(np_data).convert(mode)

        with tempfile.NamedTemporaryFile('wb') as f:
            img.save(f, format='jpeg')
            img = Image.open(f.name)

            # Test encode
            enc_data = jpeg_enc.encode(img)
            assert isinstance(enc_data, bytes)

            # Test decode
            dec_data = jpeg_enc.decode(enc_data)
            assert isinstance(dec_data, Image.Image)

            assert np.array_equal(np.array(img), np.array(dec_data))

    @pytest.mark.parametrize('data', [b'99', 12.5])
    def test_jpeg_encode_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            jpeg_enc = mdsEnc.JPEG()
            _ = jpeg_enc.encode(data)

    @pytest.mark.parametrize('mode', ['I', 'L', 'RGB'])
    def test_png_encode_decode(self, mode: str):
        png_enc = mdsEnc.PNG()
        assert png_enc.size is None

        # Creating the (32 x 32) NumPy Array with random values
        np_data = np.random.randint(255, size=(32, 32), dtype=np.uint32)
        # Default image mode of PIL Image is 'I'
        img = Image.fromarray(np_data).convert(mode)

        # Test encode
        enc_data = png_enc.encode(img)
        assert isinstance(enc_data, bytes)

        # Test decode
        dec_data = png_enc.decode(enc_data)
        dec_data = dec_data.convert('I')
        np_dec_data = np.asarray(dec_data, dtype=np.uint32)
        assert isinstance(dec_data, Image.Image)

        # Validate data content
        assert np.array_equal(np_data, np_dec_data)

    @pytest.mark.parametrize('data', [b'123', 77.7])
    def test_png_encode_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            png_enc = mdsEnc.PNG()
            _ = png_enc.encode(data)

    @pytest.mark.parametrize('data', [25, 'streaming', np.array(7)])
    def test_pickle_encode_decode(self, data: Any):
        pkl_enc = mdsEnc.Pickle()
        assert pkl_enc.size is None

        # Test encode
        enc_data = pkl_enc.encode(data)
        assert isinstance(enc_data, bytes)

        # Test decode
        dec_data = pkl_enc.decode(enc_data)
        assert isinstance(dec_data, type(data))

        # Validate data content
        assert dec_data == data

    @pytest.mark.parametrize('data', [25, 'streaming', {'alpha': 1, 'beta': 2}])
    def test_json_encode_decode(self, data: Any):
        json_enc = mdsEnc.JSON()
        assert json_enc.size is None

        # Test encode
        enc_data = json_enc.encode(data)
        assert isinstance(enc_data, bytes)

        # Test decode
        dec_data = json_enc.decode(enc_data)
        assert isinstance(dec_data, type(data))

        # Validate data content
        assert dec_data == data

    def test_json_invalid_data(self):
        wrong_json_with_single_quotes = "{'name': 'streaming'}"
        with pytest.raises(json.JSONDecodeError):
            json_enc = mdsEnc.JSON()
            json_enc._is_valid(wrong_json_with_single_quotes, wrong_json_with_single_quotes)

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'*')])
    def test_mds_uint8(self, decoded: int, encoded: bytes):
        coder = mdsEnc.UInt8()
        assert coder.size == 1

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.integer)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'*\0')])
    def test_mds_uint16(self, decoded: int, encoded: bytes):
        coder = mdsEnc.UInt16()
        assert coder.size == 2

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.integer)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'*\0\0\0')])
    def test_mds_uint32(self, decoded: int, encoded: bytes):
        coder = mdsEnc.UInt32()
        assert coder.size == 4

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.integer)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'*\0\0\0\0\0\0\0')])
    def test_mds_uint64(self, decoded: int, encoded: bytes):
        coder = mdsEnc.UInt64()
        assert coder.size == 8

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.integer)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'*')])
    def test_mds_int8(self, decoded: int, encoded: bytes):
        coder = mdsEnc.Int8()
        assert coder.size == 1

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.integer)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'*\0')])
    def test_mds_int16(self, decoded: int, encoded: bytes):
        coder = mdsEnc.Int16()
        assert coder.size == 2

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.integer)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'*\0\0\0')])
    def test_mds_int32(self, decoded: int, encoded: bytes):
        coder = mdsEnc.Int32()
        assert coder.size == 4

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.integer)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'*\0\0\0\0\0\0\0')])
    def test_mds_int64(self, decoded: int, encoded: bytes):
        coder = mdsEnc.Int64()
        assert coder.size == 8

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.integer)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42.0, b'@Q')])
    def test_mds_float16(self, decoded: float, encoded: bytes):
        coder = mdsEnc.Float16()
        assert coder.size == 2

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.floating)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42.0, b'\0\0(B')])
    def test_mds_float32(self, decoded: float, encoded: bytes):
        coder = mdsEnc.Float32()
        assert coder.size == 4

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.floating)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42.0, b'\0\0\0\0\0\0E@')])
    def test_mds_float64(self, decoded: float, encoded: bytes):
        coder = mdsEnc.Float64()
        assert coder.size == 8

        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, np.floating)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42, b'42'), (-42, b'-42')])
    def test_mds_StrInt(self, decoded: int, encoded: bytes):
        coder = mdsEnc.StrInt()
        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, int)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(42.0, b'42.0'), (-42.0, b'-42.0')])
    def test_mds_StrFloat(self, decoded: float, encoded: bytes):
        coder = mdsEnc.StrFloat()
        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, float)
        assert dec == decoded

    @pytest.mark.parametrize(('decoded', 'encoded'), [(Decimal('4E15'), b'4E+15'),
                                                      (Decimal('-4E15'), b'-4E+15')])
    def test_mds_StrDecimal(self, decoded: Decimal, encoded: bytes):
        coder = mdsEnc.StrDecimal()
        enc = coder.encode(decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded

        dec = coder.decode(encoded)
        assert isinstance(dec, Decimal)
        assert dec == decoded

    def test_get_mds_encodings(self):
        uints = {'uint8', 'uint16', 'uint32', 'uint64'}
        ints = {'int8', 'int16', 'int32', 'int64', 'str_int'}
        floats = {'float16', 'float32', 'float64', 'str_float', 'str_decimal'}
        scalars = uints | ints | floats
        expected_encodings = {
            'int', 'bytes', 'json', 'ndarray', 'png', 'jpeg', 'str', 'pil', 'pkl'
        } | scalars
        enc = mdsEnc.get_mds_encodings()
        assert len(enc) == len(expected_encodings)
        assert enc == expected_encodings

    @pytest.mark.parametrize(('enc_name', 'expected_output'), [('jpeg', True), ('', False),
                                                               ('pngg', False)])
    def test_is_mds_encoding(self, enc_name: str, expected_output: bool):
        is_supported = mdsEnc.is_mds_encoding(enc_name)
        assert is_supported is expected_output

    @pytest.mark.parametrize(('encoding', 'decoded', 'encoded'),
                             [('uint8', 42, b'*'), ('uint16', 42, b'*\0'),
                              ('uint32', 42, b'*\0\0\0'), ('uint64', 42, b'*\0\0\0\0\0\0\0'),
                              ('int8', 42, b'*'), ('int16', 42, b'*\0'), ('int32', 42, b'*\0\0\0'),
                              ('int64', 42, b'*\0\0\0\0\0\0\0'), ('float16', 42.0, b'@Q'),
                              ('float32', 42.0, b'\0\0(B'), ('float64', 42.0, b'\0\0\0\0\0\0E@')])
    def test_mds_scalar(self, encoding: str, decoded: Union[int, float], encoded: bytes):
        enc = mdsEnc.mds_encode(encoding, decoded)
        assert isinstance(enc, bytes)
        assert enc == encoded
        dec = mdsEnc.mds_decode(encoding, enc)
        assert dec == decoded
        dec = mdsEnc.mds_decode(encoding, encoded)
        assert dec == decoded

    @pytest.mark.parametrize(('enc_name', 'data'), [('bytes', b'9'), ('int', 27),
                                                    ('str', 'mosaicml')])
    def test_mds_encode(self, enc_name: str, data: Any):
        output = mdsEnc.mds_encode(enc_name, data)
        assert isinstance(output, bytes)

    @pytest.mark.parametrize(('enc_name', 'data'), [('bytes', 9), ('int', '27'), ('str', 12.5)])
    def test_mds_encode_invalid_data(self, enc_name: str, data: Any):
        with pytest.raises(AttributeError):
            _ = mdsEnc.mds_encode(enc_name, data)

    @pytest.mark.parametrize(('enc_name', 'data', 'expected_data_type'),
                             [('bytes', b'c\x00\x00\x00\x00\x00\x00\x00', bytes),
                              ('str', b'mosaicml', str)])
    def test_mds_decode(self, enc_name: str, data: Any, expected_data_type: Any):
        output = mdsEnc.mds_decode(enc_name, data)
        assert isinstance(output, expected_data_type)

    @pytest.mark.parametrize(('enc_name', 'expected_size'), [('bytes', None), ('int', 8)])
    def test_get_mds_encoded_size(self, enc_name: str, expected_size: Any):
        output = mdsEnc.get_mds_encoded_size(enc_name)
        assert output is expected_size


class TestXSVEncodings:

    @pytest.mark.parametrize(('data', 'encode_data'), [('99', '99'),
                                                       ('streaming dataset', 'streaming dataset')])
    def test_str_encode_decode(self, data: str, encode_data: str):
        str_enc = xsvEnc.Str()

        # Test encode
        enc_data = str_enc.encode(data)
        assert isinstance(enc_data, str)
        assert enc_data == encode_data

        # Test decode
        dec_data = str_enc.decode(encode_data)
        assert isinstance(dec_data, str)
        assert dec_data == data

    @pytest.mark.parametrize('data', [99, b'streaming dataset', 123.45])
    def test_str_encode_invalid_data(self, data: Any):
        with pytest.raises(Exception):
            str_enc = xsvEnc.Str()
            _ = str_enc.encode(data)

    @pytest.mark.parametrize(('data', 'encode_data'), [(99, '99'), (987675432, '987675432')])
    def test_int_encode_decode(self, data: int, encode_data: str):
        int_enc = xsvEnc.Int()

        # Test encode
        enc_data = int_enc.encode(data)
        assert isinstance(enc_data, str)
        assert enc_data == encode_data

        # Test decode
        dec_data = int_enc.decode(encode_data)
        assert isinstance(dec_data, int)
        assert dec_data == data

    @pytest.mark.parametrize('data', ['99', b'streaming dataset', 123.45])
    def test_int_encode_invalid_data(self, data: Any):
        with pytest.raises(Exception):
            int_enc = xsvEnc.Int()
            _ = int_enc.encode(data)

    @pytest.mark.parametrize(('data', 'encode_data'), [(1.24, '1.24'), (9.0, '9.0')])
    def test_float_encode_decode(self, data: int, encode_data: str):
        float_enc = xsvEnc.Float()

        # Test encode
        enc_data = float_enc.encode(data)
        assert isinstance(enc_data, str)
        assert enc_data == encode_data

        # Test decode
        dec_data = float_enc.decode(encode_data)
        assert isinstance(dec_data, float)
        assert dec_data == data

    @pytest.mark.parametrize('data', ['99', b'streaming dataset', 12])
    def test_float_encode_invalid_data(self, data: Any):
        with pytest.raises(Exception):
            float_enc = xsvEnc.Float()
            _ = float_enc.encode(data)

    @pytest.mark.parametrize(('enc_name', 'expected_output'), [
        ('str', True),
        ('int', True),
        ('float', True),
        ('', False),
    ])
    def test_is_xsv_encoding(self, enc_name: str, expected_output: bool):
        is_supported = xsvEnc.is_xsv_encoding(enc_name)
        assert is_supported is expected_output

    @pytest.mark.parametrize(('enc_name', 'data', 'expected_data'),
                             [('str', 'mosaicml', 'mosaicml'), ('int', 27, '27'),
                              ('float', 1.25, '1.25')])
    def test_xsv_encode(self, enc_name: str, data: Any, expected_data: str):
        output = xsvEnc.xsv_encode(enc_name, data)
        assert isinstance(output, str)
        assert output == expected_data

    @pytest.mark.parametrize(('enc_name', 'data', 'expected_data'),
                             [('str', 'mosaicml', 'mosaicml'), ('int', '27', 27),
                              ('float', '1.25', 1.25)])
    def test_xsv_decode(self, enc_name: str, data: str, expected_data: Any):
        output = xsvEnc.xsv_decode(enc_name, data)
        assert isinstance(output, type(expected_data))
        assert output == expected_data


class TestJSONEncodings:

    @pytest.mark.parametrize('data', ['99', 'mosaicml'])
    def test_str_is_encoded(self, data: str):
        json_enc = jsonEnc.Str()

        # Test encode
        enc_data = json_enc.is_encoded(data)
        assert isinstance(enc_data, bool)

    @pytest.mark.parametrize('data', [99, b'mosaicml'])
    def test_str_is_encoded_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            json_enc = jsonEnc.Str()
            _ = json_enc.is_encoded(data)

    @pytest.mark.parametrize('data', [99, 987675432])
    def test_int_is_encoded(self, data: int):
        int_enc = jsonEnc.Int()

        # Test encode
        enc_data = int_enc.is_encoded(data)
        assert isinstance(enc_data, bool)

    @pytest.mark.parametrize('data', ['99', b'mosaicml', 1.25])
    def test_int_is_encoded_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            int_enc = jsonEnc.Int()
            _ = int_enc.is_encoded(data)

    @pytest.mark.parametrize('data', [1.25])
    def test_float_is_encoded(self, data: int):
        float_enc = jsonEnc.Float()

        # Test encode
        enc_data = float_enc.is_encoded(data)
        assert isinstance(enc_data, bool)

    @pytest.mark.parametrize('data', ['99', b'mosaicml', 25])
    def test_float_is_encoded_invalid_data(self, data: Any):
        with pytest.raises(AttributeError):
            float_enc = jsonEnc.Float()
            _ = float_enc.is_encoded(data)

    @pytest.mark.parametrize(('enc_name', 'expected_output'), [
        ('str', True),
        ('int', True),
        ('float', True),
        ('', False),
    ])
    def test_is_json_encoding(self, enc_name: str, expected_output: bool):
        is_supported = jsonEnc.is_json_encoding(enc_name)
        assert is_supported is expected_output

    @pytest.mark.parametrize(('enc_name', 'data', 'expected_output'), [('str', 'hello', True),
                                                                       ('int', 10, True),
                                                                       ('float', 9.9, True)])
    def test_is_json_encoded(self, enc_name: str, data: Any, expected_output: bool):
        is_supported = jsonEnc.is_json_encoded(enc_name, data)
        assert is_supported is expected_output
