# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""An MDS shard."""

import json
import os
from dataclasses import dataclass
from typing import IO, Any, Dict, List, Optional

import numpy as np
from typing_extensions import Self

from streaming.format.base.file import ShardFile
from streaming.format.base.phase import ShardFilePhase
from streaming.format.base.shard.mono_row import MonoRowShard
from streaming.format.mds.encodings import is_mds_encoding_safe, mds_decode
from streaming.stream.dir_conf import StreamDirConf

__all__ = ['MDSShard']


@dataclass
class MDSColumn:
    """An MDS column."""

    name: str
    encoding: str
    size: Optional[int]


class MDSShard(MonoRowShard):
    """An MDS shard.

    Args:
        conf (Any, optional): JSON shard config. Defaults to ``None``.
        stream (StreamDirConf): Link back up to the Stream that owns this shard, from which we
            get arguments which are shared across all shards like remote/local paths. Avoids an
            import cycle by Stream subclassing StreamDirConf.
        num_samples (int): Number of samples in this shard.
        file (ShardFile): The file containing shard data and metadata.
        columns (List[MDSColumn]): Column metadata.
    """

    def __init__(
        self,
        *,
        conf: Any,
        stream: StreamDirConf,
        num_samples: int,
        file: ShardFile,
        columns: List[MDSColumn],
    ) -> None:
        super().__init__(
            conf=conf,
            stream=stream,
            num_samples=num_samples,
            file=file,
        )
        self.columns = columns

    @classmethod
    def from_json(cls, stream: StreamDirConf, obj: Dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Args:
            stream (StreamDirConf): Reference to the owning Stream.
            obj (Dict[str, Any]): MDS shard JSON metadata.

        Returns:
            Self: The loaded MDS shard object.
        """
        num_samples = obj['samples']
        zip_obj = obj.get('zip_data')
        zip_phase = ShardFilePhase.from_json(stream, zip_obj) if zip_obj else None
        zip_algo = obj.get('compression')
        raw_phase = ShardFilePhase.from_json(stream, obj['raw_data'])
        file = ShardFile(
            stream=stream,
            zip_phase=zip_phase,
            zip_algo=zip_algo,
            raw_phase=raw_phase,
        )
        names = obj['column_names']
        encodings = obj['column_encodings']
        sizes = obj['column_sizes']
        columns = [MDSColumn(*args) for args in zip(names, encodings, sizes)]
        return cls(
            conf=obj,
            stream=stream,
            num_samples=num_samples,
            file=file,
            columns=columns,
        )

    def validate(self):
        """Check whether this shard is acceptable to be part of some Stream."""
        super().validate()
        if not self.stream.allow_unsafe_types:
            for column in self.columns:
                if not is_mds_encoding_safe(column.encoding):
                    raise ValueError(f'Column `{column.name}` contains an unsafe type: ' +
                                     f'`{column.encoding}`. To proceed anyway, set ' +
                                     f'`allow_unsafe_types` to `True`.')

    def _analyze(self, fp: IO[bytes]) -> Dict[str, Any]:
        obj = {
            'index_adv_samples': self.num_samples,
            'index_adv_bytes': self.file.raw_phase.size,
        }

        obj['file_got_bytes'] = fp.seek(0, os.SEEK_END)

        fp.seek(0)
        data = fp.read(4)
        if len(data) < 4:
            return obj
        file_adv_samples, = np.frombuffer(data, np.uint32)
        obj['file_adv_samples'] = file_adv_samples

        fp.seek(4 + file_adv_samples * 4)
        data = fp.read(4)
        if len(data) < 4:
            return obj
        file_adv_bytes, = np.frombuffer(data, np.uint32)
        obj['file_adv_bytes'] = file_adv_bytes

        obj['samples_match'] = obj['index_adv_samples'] == obj['file_adv_samples']
        obj['bytes_match'] = obj['index_adv_bytes'] == obj['file_adv_bytes'] == \
            obj['file_got_bytes']

        obj['match'] = obj['samples_match'] and obj['bytes_match']

        return obj

    def get_sample_data(self, sample_id: int) -> bytes:
        """Get the raw sample data at the index.

        Args:
            sample_id (int): Sample index.

        Returns:
            bytes: Sample data.
        """
        filename = self.file.raw_phase.get_local_filename()

        def get_err(fp: IO[bytes], msg_txt: str) -> ValueError:
            chk = self._analyze(fp)
            chk_txt = json.dumps(chk, sort_keys=True)
            return ValueError(
                f'{msg_txt}: filename {filename}, sample ID {sample_id}, predicted num samples ' +
                f'{self.num_samples}, cross-check {chk_txt}.')

        with open(filename, 'rb', 0) as fp:
            if not (0 <= sample_id < self.num_samples):
                raise get_err(fp, 'Attempted to read and out-of-range sample ID')

            begin = (1 + sample_id) * 4
            fp.seek(begin)
            data = fp.read(8)

            if len(data) < 8:
                raise get_err(fp, 'Dataset corruption detected')

            begin, end = np.frombuffer(data, np.uint32)
            fp.seek(begin)
            data = fp.read(end - begin)

            if len(data) < end - begin:
                raise get_err(fp, 'Dataset corruption detected')

        return data

    def decode_sample(self, data: bytes) -> Dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        sizes = []
        idx = 0
        for column in self.columns:
            if column.size is not None:
                sizes.append(column.size)
            else:
                size, = np.frombuffer(data[idx:idx + 4], np.uint32)
                sizes.append(size)
                idx += 4

        sample = {}
        for column, size in zip(self.columns, sizes):
            value = data[idx:idx + size]
            sample[column.name] = mds_decode(column.encoding, value)
            idx += size

        return sample
