# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

""":class:`MDSWriter` writes samples to ``.mds`` files that can be read by :class:`MDSReader`."""

import json
import os
from itertools import chain
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from tqdm import tqdm

from streaming.format.index import get_index_basename
from streaming.format.mds.encodings import (get_mds_encoded_size, get_mds_encodings,
                                            is_mds_encoding, mds_encode)
from streaming.format.writer import JointWriter

__all__ = ['MDSWriter']


class MDSWriter(JointWriter):
    """Writes a streaming MDS dataset.

    Args:
        columns (Dict[str, str]): Sample columns.
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in the
               `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (Union[int, str], optional): Optional shard size limit, after which point to
            start a new shard. If ``None``, puts everything in one shard. Can specify bytes
            human-readable format as well, for example ``"100kb"`` for 100 kilobyte
            (100*1024) and so on. Defaults to ``1 << 26``.
        **kwargs (Any): Additional settings for the Writer.

            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            max_workers (int): Maximum number of threads used to upload output dataset files in
                parallel to a remote location. One thread is responsible for uploading one shard
                file to a remote location. Default to ``min(32, (os.cpu_count() or 1) + 4)``.
    """

    format = 'mds'
    extra_bytes_per_sample = 4

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[Union[int, str]] = 1 << 26,
                 **kwargs: Any) -> None:
        super().__init__(out=out,
                         keep_local=keep_local,
                         compression=compression,
                         hashes=hashes,
                         size_limit=size_limit,
                         extra_bytes_per_sample=self.extra_bytes_per_sample,
                         **kwargs)
        self.columns = columns
        self.column_names = []
        self.column_encodings = []
        self.column_sizes = []
        for name in sorted(columns):
            encoding = columns[name]
            if not is_mds_encoding(encoding):
                raise TypeError(f'MDSWriter passed column `{name}` with encoding `{encoding}` ' +
                                f'is unsupported. Supported encodings are {get_mds_encodings()}')
            size = get_mds_encoded_size(encoding)
            self.column_names.append(name)
            self.column_encodings.append(encoding)
            self.column_sizes.append(size)

        obj = self.get_config()
        text = json.dumps(obj, sort_keys=True)
        self.config_data = text.encode('utf-8')
        self.extra_bytes_per_shard = 4 + 4 + len(self.config_data)
        self._reset_cache()

    def encode_sample(self, sample: Dict[str, Any]) -> bytes:
        """Encode a sample dict to bytes.

        Args:
            sample (Dict[str, Any]): Sample dict.

        Returns:
            bytes: Sample encoded as bytes.
        """
        sizes = []
        data = []
        for key, encoding, size in zip(self.column_names, self.column_encodings,
                                       self.column_sizes):
            value = sample[key]
            datum = mds_encode(encoding, value)
            if size is None:
                size = len(datum)
                sizes.append(size)
            else:
                if size != len(datum):
                    raise KeyError(f'Unexpected data size; was this data typed with the correct ' +
                                   f'encoding ({encoding})?')
            data.append(datum)
        head = np.array(sizes, np.uint32).tobytes()
        body = b''.join(data)
        return head + body

    def get_config(self) -> Dict[str, Any]:
        """Get object describing shard-writing configuration.

        Returns:
            Dict[str, Any]: JSON object.
        """
        obj = super().get_config()
        obj.update({
            'column_names': self.column_names,
            'column_encodings': self.column_encodings,
            'column_sizes': self.column_sizes
        })
        return obj

    def encode_joint_shard(self) -> bytes:
        """Encode a joint shard out of the cached samples (single file).

        Returns:
            bytes: File data.
        """
        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        offsets += len(num_samples.tobytes()) + len(offsets.tobytes()) + len(self.config_data)
        sample_data = b''.join(self.new_samples)
        return num_samples.tobytes() + offsets.tobytes() + self.config_data + sample_data


_type2enc = {
    int: 'int',
    str: 'str',
}


def infer_column(field: Any) -> str:
    """Infer the best MDS encoding for a column, given an example field.

    Args:
        field (Any): The example.

    Returns:
        MDS encoding signature.
    """
    ty = type(field)
    return _type2enc[ty]


def infer_columns(sample: Dict[str, Any]) -> Dict[str, str]:
    """Infer dataset columns given a sample.

    Args:
        sample (Dict[str, Any]): Mapping of field name to value.

    Returns:
        Dict[str, str]: Mapping of field name to type.
    """
    ret = {}
    for key in sorted(sample):
        val = sample[key]
        ret[key] = infer_column(val)
    return ret


def write_dataset(samples: Iterable[Dict[str, Any]],
                  out: Union[str, Tuple[str, str]],
                  *,
                  num_samples: Optional[int] = None,
                  keep_local: bool = False,
                  columns: Optional[Dict[str, str]] = None,
                  compression: Optional[str] = None,
                  hashes: Optional[List[str]] = None,
                  max_file_bytes: Optional[Union[int, str]] = '32mib',
                  num_upload_threads: Optional[int] = None,
                  upload_retry: int = 2,
                  show_write_progress: bool = True,
                  show_upload_progress: bool = True) -> None:
    """Write the samples as an MDS dataset.

    Args:
        samples (Iterable[Dict[str, Any]]): Iterable of sample dicts.
        out (str | Tuple[str, str]): Dataaset save directory, or pair of (local, remote).
        num_samples ((int, optional): If ``samples`` is a generator, specify ``num_samples``to
            still get a useful progress bar. Defaults to ``None``.
        keep_local (bool): Whether to keep local files after upload. Defaults to ``False``.
        columns (Dict[str, str], optional): Inferred column overrides. Defaults to ``None``.
        compression (str, optional): What compression scheme to use, if any. Defaults to ``None``.
        hashes (List[str], optional): List of hashes to apply to dataset files.
        max_file_bytes (int | str, optional): Optional maximum shard size, in bytes. If no limit,
            we will write exactly one (potentially very large) shard. Defaults to ``32mib``.
        num_upload_threads (int, optional): Number of threads used to upload shards. Defaults to
            ``None``, which means to take the default, which is scaled for CPU count, etc.
        upload_retry (int): Number of upload reattempts before bailing. Defaults to ``2``.
        show_write_progress (bool): Show a progress bar for write progress. Defaults to ``True``.
        show_upload_progress (bool): Show a progress bar for upload progress. Defaults to ``True``.
    """
    # TODO: Use the part.00000/ subdir trick to make datasets easily appendable to.

    # First, count the number of samples to write from the input Iterable, falling back to the
    # user-provided hint if it has no size.
    total = len(samples) if hasattr(samples, '__len__') else num_samples  # pyright: ignore

    # If user did not tell us the schema, pop a sample off the front of the iterator, infer
    # columns, then put it back lol.
    it = iter(samples)
    if not columns:
        sample = next(it)  # If samples is empty, user goofed.
        columns = infer_columns(sample)
        it = chain([sample], it)

    # Now that we have an iteator for reals, wrap it with the "write" progress bar.
    if show_write_progress:
        it = tqdm(it, total=total, leave=False)

    # Finally walk/write the samples.
    with MDSWriter(columns=columns,
                   out=out,
                   keep_local=keep_local,
                   compression=compression,
                   hashes=hashes,
                   size_limit=max_file_bytes,
                   progress_bar=show_upload_progress,
                   max_workers=num_upload_threads,
                   retry=upload_retry) as writer:
        for sample in it:
            writer.write(sample)


def write_shard(*args: Any,
                tmp_dir: Optional[str] = None,
                shard_basename: str = 'shard.00000.mds',
                **kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Write the samples as a single MDS shard.

    Args:
        *args (Any): Positional arguments for ``write_dataset()``.
        tmp_dir (str, optional): Write the MDS dataset to this specific directory instead of
            lettting python tempfile pick one for us. Empties and removes the diretory when done.
            This argument is useful if your shard is very large and your system's standard temp
            root is across a filesystem boundary from the local cache dir you are using.
        shard_basename (str): Path to shard, relative to dataset. Defaults to ``shard.00000.mds``.
        **kwargs (Dict[str, Any]): Keyword arguments for ``write_dataset()``.

    Returns:
        Dict[str, Any]: JSON dict of the shard metadata.
    """
    # We happen to only have a need for this restricted use case.
    shard_dest = kwargs.get('out')
    if not isinstance(shard_dest, str) or urlparse(shard_dest).scheme:
        raise ValueError(f'Streaming is restricted to only writing MDS datasets of one unlimited' +
                         f'shard when the output is just a local abs/rel path with no file:// ' +
                         f'prefix, but got: {shard_dest}.')

    # Verify our actions are aligned with our goals, which is one shard of technically unlimited
    # size because of specific weird reasons (i.e., mirroring Parquets to MDS).
    if kwargs.get('max_file_bytes'):
        raise ValueError('We question your values.')
    kwargs.__dict__['max_file_bytes'] = None

    # Fall back to using python tempfile.
    if not tmp_dir:
        tmp_dir = mkdtemp()

    # Verify scratch dir not present.
    if os.path.exists(tmp_dir):
        raise ValueError(f'Scratch path already exists: {tmp_dir}.')

    # Serialize a uni-shard dataset to the temp directory.
    kwargs.__dict__['out'] = tmp_dir
    write_dataset(*args, **kwargs)

    # Move the shard from its dataset to the desired location.
    shard_source = os.path.join(tmp_dir, shard_basename)
    os.rename(shard_source, shard_dest)

    # Get the shard metadata from the index (could also get it from the MDS shard itself).
    index_path = os.path.join(tmp_dir, get_index_basename())
    obj = json.load(open(index_path))
    info, = obj['shards']

    # Cleanup.
    rmtree(tmp_dir)

    # Return shard metadata.
    return info
