# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""An XSV shard."""

from typing import Any, Dict, List, Optional

import numpy as np
from typing_extensions import Self

from streaming.format.base.file import ShardFile
from streaming.format.base.phase import ShardFilePhase
from streaming.format.base.shard.dual_row import DualRowShard
from streaming.format.xsv.encodings import xsv_decode
from streaming.stream.dir_conf import StreamDirConf

__all__ = ['XSVShard']


class XSVShard(DualRowShard):
    """An XSV shard.

    Args:
        conf (Any, optional): XSV shard config. Defaults to ``None``.
        stream (StreamDirConf): Link back up to the Stream that owns this shard, from which we
            get arguments which are shared across all shards like remote/local paths. Avoids an
            import cycle by Stream subclassing StreamDirConf.
        num_samples (int): Number of samples in this shard.
        data_file (ShardFile): The file containing shard data.
        meta_file (ShardFile): The file containing shard metadata.
        column_names (List[str]): Column names.
        column_encodings (List[str]): Column encodings.
        newline (str): Newline separator.
        separator (str): Field separator.
    """

    def __init__(
        self,
        *,
        conf: Optional[Any] = None,
        stream: StreamDirConf,
        num_samples: int,
        data_file: ShardFile,
        meta_file: ShardFile,
        column_names: List[str],
        column_encodings: List[str],
        newline: str,
        separator: str,
    ) -> None:
        super().__init__(
            conf=conf,
            stream=stream,
            num_samples=num_samples,
            data_file=data_file,
            meta_file=meta_file,
        )
        self.column_names = column_names
        self.column_encodings = column_encodings
        self.newline = newline
        self.separator = separator

    @classmethod
    def from_json(cls, stream: StreamDirConf, obj: Dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Args:
            stream (StreamDirConf): Reference to the owning Stream.
            obj (Dict[str, Any]): XSV shard JSON metadata.

        Returns:
            Self: The loaded XSV shard object.
        """
        zip_algo = obj.get('compression')
        key_pairs = [
            ('raw_data', 'zip_data'),
            ('raw_meta', 'zip_meta'),
        ]
        files = []
        for raw_key, zip_key in key_pairs:
            zip_obj = obj.get(zip_key)
            zip_phase = ShardFilePhase.from_json(stream, zip_obj) if zip_obj else None
            raw_phase = ShardFilePhase.from_json(stream, obj[raw_key])
            file = ShardFile(
                stream=stream,
                zip_phase=zip_phase,
                zip_algo=zip_algo,
                raw_phase=raw_phase,
            )
            files.append(file)
        data_file, meta_file = files
        column_names = obj['column_names']
        column_encodings = obj['column_encodings']
        return cls(
            conf=obj,
            stream=stream,
            num_samples=obj['samples'],
            data_file=data_file,
            meta_file=meta_file,
            column_names=column_names,
            column_encodings=column_encodings,
            newline=obj['newline'],
            separator=obj['separator'],
        )

    def get_sample_data(self, idx: int) -> bytes:
        """Get the raw sample data at the index.

        Args:
            idx (int): Sample index.

        Returns:
            bytes: Sample data.
        """
        meta_filename = self.meta_file.raw_phase.get_local_filename()
        offset = (1 + idx) * 4
        with open(meta_filename, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
        data_filename = self.data_file.raw_phase.get_local_filename()
        with open(data_filename, 'rb', 0) as fp:
            fp.seek(begin)
            data = fp.read(end - begin)
        return data

    def decode_sample(self, data: bytes) -> Dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        text = data.decode('utf-8')
        text = text[:-len(self.newline)]
        parts = text.split(self.separator)
        sample = {}
        for col_name, col_encoding, part in zip(self.column_names, self.column_encodings, parts):
            sample[col_name] = xsv_decode(col_encoding, part)
        return sample


class CSVShard(XSVShard):
    """A CSV shard.

    Args:
        kwargs (Any): Keyword arguments.
    """

    separator = ','

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        kwargs['separator'] = self.separator
        super().__init__(**kwargs)


class TSVShard(XSVShard):
    """A TSV shard.

    Args:
        kwargs (Any): Keyword arguments.
    """

    separator = '\t'

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        kwargs['separator'] = self.separator
        super().__init__(**kwargs)
