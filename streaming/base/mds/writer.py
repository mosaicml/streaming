import json
import numpy as np
import os
from types import TracebackType
from typing import Dict, List, Optional, Type
from typing_extensions import Self

from .index import get_index_basename, MDSIndex


class MDSWriter(object):
    """Writes a MDS-formatted streaming dataset.

    Args:
        dirname (str): Local dataset directory.
        index (MDSIndex): Index of contents.
    """

    def __init__(self, dirname: str, index: MDSIndex) -> None:
        self.dirname = dirname
        self.index = index
        self.field_data = json.dumps(self.index.fields, sort_keys=True).encode('utf-8')
        self._reset_cache()

    def _reset_cache(self) -> None:
        """Reset our internal shard-building cache (called on init or after writing a shard)."""
        self.new_samples = []
        self.new_shard_size = 4 + 4 + len(self.field_data)

    def _encode_shard(self) -> bytes:
        """Encode cached sample data into a shard file.

        Format: (num samples) (sample offsets) (field data) (sample data).

        This format allows random access on samples via two seeks (begin/end offsets, then sample
        data). It also contains all the information needed to decode the sample dicts within the
        shard itself.

        Returns:
            bytes: The encoded data.
        """
        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        offsets += len(num_samples.tobytes()) + len(offsets.tobytes()) + len(self.field_data)
        sample_data = b''.join(self.new_samples)
        return num_samples.tobytes() + offsets.tobytes() + self.field_data + sample_data

    def _flush_shard(self) -> None:
        """Flush cached samples to storage, creating a new shard."""
        data = self._encode_shard()
        num_samples = len(self.new_samples)
        data, basename = self.index.add_shard(data, num_samples)
        filename = os.path.join(self.dirname, basename)
        with open(filename, 'wb') as out:
            out.write(data)
        self._reset_cache()

    def write(self, sample: Dict[str, bytes]) -> None:
        """Write a sample. Either caches or flushes an entire new shard.

        Args:
            sample (Dict[str, bytes]): The raw sample data.
        """
        data = self.index.encode_sample(sample)
        self.new_samples.append(data)
        new_sample_size = 4 + len(data)
        self.new_shard_size += new_sample_size
        if self.index.limit <= self.new_shard_size + new_sample_size:
            self._flush_shard()

    def _write_index(self) -> None:
        """Write the index, having written all the shards."""
        assert not self.new_samples
        filename = os.path.join(self.dirname, get_index_basename())
        with open(filename, 'w') as out:
            self.index.dump(out)

    def finish(self) -> None:
        """Finish writing samples."""
        if self.new_samples:
            self._flush_shard()
        self._write_index()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """Exist context manager.

        Args:
            exc_type (Optional[Type[BaseException]]): Exc type.
            exc (Optional[BaseException]): Exc.
            traceback (Optional[TracebackType]): Traceback.
        """
        self.finish()


class MDSCreator(MDSWriter):
    """Creates a MDS-formatted streaming dataset.

    Args:
        dirname (str): Local dataset directory.
        fields (Dict[str, str]): Mapping of field names to their types (encodings).
        compression (Optional[str], default: None): Optional compression.
        hashes (Optional[List[str]], default: None): Optional list of hash algorithms.
        limit (int, default: 1 << 26): Uncompressed shard size limit, at which point it flushes 
            the shard and starts a new one.
    """

    def __init__(
        self,
        dirname: str,
        fields: Dict[str, str],
        compression: Optional[str] = None,
        hashes: Optional[List[str]] = None,
        limit: int = 1 << 26
    ) -> None:
        index = MDSIndex(fields, compression, hashes, limit)
        super().__init__(dirname, index)
        os.makedirs(dirname)


class MDSAppender(MDSWriter):
    """Appends to a MDS-formatted streaming dataset.

    Args:
        dirname (str): Local dataset directory.
    """

    def __init__(self, dirname: str):
        filename = os.path.join(dirname, get_index_basename())
        index = MDSIndex.load(open(filename))
        super().__init__(dirname, index)
