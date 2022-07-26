from io import TextIOWrapper
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Self


from ..compression import compress, is_compression
from ..hashing import get_hash, is_hash
from .encodings import decode, encode, get_encoded_size, is_encoding


def get_index_basename() -> str:
    """Get the canonical index file basename.

    Returns:
        str: Index file basename.
    """
    return 'index.json'


class FileInfo(object):
    """Metadata to locate and validate a file.

    Args:
        basename (str): File basename.
        bytes (int): Size in bytes.
        hashes (Dict[str, str]): Hashes of the file data.
    """

    def __init__(self, basename: str, bytes: int, hashes: Dict[str, str]) -> None:
        self.basename = basename
        self.bytes = bytes
        self.hashes = hashes

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> Self:
        """Load from JSON.

        Args:
            obj (Dict[str, Any]): JSON object.

        Returns:
            cls: Loaded cls.
        """
        basename = obj['basename']
        bytes = obj['bytes']
        hashes = obj['hashes']
        return cls(basename, bytes, hashes)

    def json(self) -> Dict[str, Any]:
        """Dump as JSON.

        Returns:
            Dict[str, Any]: JSON object.
        """
        return {
            'basename': self.basename,
            'bytes': self.bytes,
            'hashes': self.hashes,
        }


class MDSShard(object):
    """MDS-formatted streaming dataset shard.

    Args:
        samples (int): Number of samples in this shard.
        raw (FileInfo): Decompressed file information.
        zip (FileInfo): Compressed file information.
    """

    def __init__(self, samples: int, raw: FileInfo, zip: Optional[FileInfo] = None) -> None:
        self.samples = samples
        self.raw = raw
        self.zip = zip

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> Self:
        """Load from JSON.

        Args:
            obj (Dict[str, Any]): JSON object.

        Returns:
            cls: Loaded cls.
        """
        samples = obj['samples']
        raw = FileInfo.from_json(obj['raw'])
        zip = FileInfo.from_json(obj['zip']) if obj['zip'] else None
        return cls(samples, raw, zip)

    def json(self) -> Dict[str, Any]:
        """Dump as JSON.

        Returns:
            Dict[str, Any]: JSON object.
        """
        return {
            'samples': self.samples,
            'raw': self.raw.json() if self.raw else None,
            'zip': self.zip.json() if self.zip else None,
        }


class MDSIndex(object):
    """Creates a MDS-formatted streaming dataset.

    Args:
        fields (Dict[str, str]): Mapping of field names to their types (encodings).
        compression (Optional[str], default: None): Optional compression.
        hashes (Optional[List[str]], default: None): Optional list of hash algorithms.
        limit (int, default: 1 << 26): Uncompressed shard size limit, at which point it flushes 
            the shard and starts a new one.
    """

    def __init__(
        self,
        fields: Dict[str, str],
        compression: Optional[str] = None,
        hashes: Optional[List[str]] = None,
        limit: int = 1 << 26,
        shards: Optional[List[MDSShard]] = None
    ) -> None:
        for encoding in fields.values():
            assert is_encoding(encoding)

        compression = compression or None
        if isinstance(compression, str):
            assert is_compression(compression)

        hashes = hashes or []
        assert list(hashes) == list(sorted(hashes))
        for algo in hashes:
            assert is_hash(algo)

        shards = shards or []

        self.fields = fields
        self.compression = compression
        self.hashes = hashes
        self.limit = limit
        self.shards = shards

        self.field_names = sorted(fields)
        self.field_sizes = [get_encoded_size(self.fields[key]) for key in self.field_names]

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> Self:
        """Load from JSON.

        Args:
            obj (Dict[str, Any]): JSON object.

        Returns:
            cls: Loaded cls.
        """
        assert obj['version'] == 2
        assert obj['format'] == 'mds'
        fields = obj['fields']
        compression = obj['compression']
        hashes = obj['hashes']
        limit = obj['limit']
        shards = [MDSShard.from_json(sub) for sub in obj['shards']]
        return cls(fields, compression, hashes, limit, shards)

    @classmethod
    def loads(cls, text: str) -> Self:
        """Load from JSON string.

        Args:
            text (str): JSON string.

        Returns:
            cls: Loaded cls.
        """
        obj = json.loads(text)
        return cls.from_json(obj)

    @classmethod
    def load(cls, fp: TextIOWrapper) -> Self:
        """Load from JSON file.

        Args:
            fp (TextIOWrapper): JSON file.

        Returns:
            cls: Loaded cls.
        """
        obj = json.load(fp)
        return cls.from_json(obj)

    def json(self) -> Dict[str, Any]:
        """Dump to JSON.

        Returns:
            Dict[str, Any]: JSON object.
        """
        return {
            'version': 2,
            'format': 'mds',
            'fields': self.fields,
            'compression': self.compression,
            'hashes': self.hashes,
            'limit': self.limit,
            'shards': [shard.json() for shard in self.shards],
        }

    def dumps(self) -> str:
        """Dump to JSON string.

        Returns:
            str: Dumped JSON string.
        """
        return json.dumps(self.json(), sort_keys=True)

    def dump(self, fp: TextIOWrapper) -> None:
        """Dump to JSON file.

        Args:
            fp (TextIOWrapper): JSON file.
        """
        json.dump(self.json(), fp)

    def encode_sample(self, sample: Dict[str, Any]) -> bytes:
        """Encode a sample dict to bytes.

        Args:
            sample (Dict[str, Any]): Smaple dict.

        Returns:
            bytes: Sample encoded as bytes.
        """
        sizes = []
        datas = []
        for key, size in zip(self.field_names, self.field_sizes):
            encoding = self.fields[key]
            value = sample[key]
            data = encode(encoding, value)
            if size is None:
                size = len(data)
                sizes.append(size)
            else:
                assert size == len(data)
            datas.append(data)
        head = np.array(sizes, np.uint32).tobytes()
        body = b''.join(datas)
        return head + body

    def decode_sample(self, data: bytes) -> Dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        sizes = []
        idx = 0
        for key, size in zip(self.field_names, self.field_sizes):
            if size:
                sizes.append(size)
            else:
                size, = np.frombuffer(data[idx:idx + 4], np.uint32)
                sizes.append(size)
                idx += 4
        sample = {}
        for key, size in zip(self.field_names, sizes):
            encoding = self.fields[key]
            value = data[idx:idx + size]
            sample[key] = decode(encoding, value)
            idx += size
        return sample

    def _name_next_shard(self) -> Tuple[str, Optional[str]]:
        """Get the filenames of the next shard to be created.

        Returns:
            Tuple[str, str]: Pair of (decompressed, compressed) filenames.
        """
        shard = len(self.shards)
        parts = ['shard', f'{shard:05}', 'mds']
        raw_basename = '.'.join(parts)
        if self.compression:
            idx = self.compression.find(':')
            if idx == -1:
                ext = self.compression
            else:
                ext = self.compression[:idx]
            parts.append(ext)
            zip_basename = '.'.join(parts)
        else:
            zip_basename = None
        return raw_basename, zip_basename

    def _hash(self, data: bytes, basename: str) -> FileInfo:
        """Generate file metadata.

        Args:
            data (bytes): The file data.
            basename (str): The file's basename.

        Returns:
            FileInfo: File metadata.
        """
        size = len(data)
        hashes = {}
        for algo in self.hashes:
            hashes[algo] = get_hash(algo, data)
        return FileInfo(basename, size, hashes)

    def add_shard(self, data: bytes, num_samples: int) -> Tuple[bytes, str]:
        """Add a new shard.

        Args:
            data (bytes): The raw shard data.
            num_samples (int): The samples to add.

        Returns:
            Tuple[bytes, str]: Optionally compressed data, and shard basename.
        """
        raw_basename, zip_basename = self._name_next_shard()

        raw_info = self._hash(data, raw_basename)
        if self.compression:
            data = compress(self.compression, data)
            zip_info = self._hash(data, zip_basename)  # pyright: ignore
        else:
            zip_info = None

        info = MDSShard(num_samples, raw_info, zip_info)
        self.shards.append(info)

        basename = zip_basename if zip_basename else raw_basename
        return data, basename
