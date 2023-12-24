# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A dataset, or sub-dataset if mixing, from which we stream/cache samples."""

import hashlib
import os
import tempfile
from typing import Optional

from streaming.distributed import barrier, get_local_rank


class StreamCore:
    """A dataset, or sub-dataset if mixing, from which we stream/cache samples.

    We initialize a StreamingDataset with one or more Streams. Streams may be resampled to achieve
    different mixtures of samples.

    Stream init takes three kinds of arguments:

    * At least one of ``remote`` and ``local`` must exist. If no ``remote``, the data must be
      local. If no ``local``, we cache to a temp directory.

      * ``remote``
      * ``local``

    * The remaining arguments are optional knobs for controlling downloading behavior and default
      to ``None``. If ``None``, they take a default value provided to or by the StreamingDataset
      init.

      * ``split``
      * ``download_retry``
      * ``download_timeout``
      * ``validate_hash``
      * ``keep_zip``

    Args:
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set. Defaults to
            ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int, optional): Number of download re-attempts before giving up. Defaults
            to ``None``.
        download_timeout (float, optional): Number of seconds to wait for a shard to download
            before raising an exception. Defaults to ``None``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool, optional): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep if and only if remote is local or no remote.
            Defaults to ``None``.
    """

    def __init__(self,
                 *,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: Optional[int] = None,
                 download_timeout: Optional[float] = None,
                 validate_hash: Optional[str] = None,
                 keep_zip: Optional[bool] = None) -> None:
        self.remote = remote
        self._local = local
        self.split = split or ''

        self._download_retry = download_retry
        if download_retry is not None:
            if download_retry < 0:
                raise ValueError('`download_retry` must be non-negative')
            self.download_retry = download_retry

        self._download_timeout = download_timeout
        if download_timeout is not None:
            if download_timeout <= 0:
                raise ValueError('`download_timeout` must be positive')
            self.download_timeout = download_timeout

        self.validate_hash = validate_hash

        if local is None:
            self.local = self._get_temporary_directory()
            if get_local_rank() == 0:
                if os.path.exists(self.local):
                    raise ValueError(
                        f'Could not create a temporary local directory {self.local} . Either ' +
                        f'delete the directory or specify a unique local directory with the ' +
                        f'`local` value.')
                os.makedirs(self.local)
            barrier()
        else:
            self.local = local

        self._keep_zip = keep_zip
        if keep_zip is not None:
            self.keep_zip = keep_zip
            self.safe_keep_zip = self.keep_zip or self.remote in {None, self.local}

    def _get_temporary_directory(self) -> str:
        """Construct a path to a temporary directory based on remote and split."""
        root = tempfile.gettempdir()
        hash = ''
        if self.remote is not None:
            hash = hashlib.blake2s(self.remote.encode('utf-8'), digest_size=16).hexdigest()
        return os.path.join(root, hash, self.split)

    def apply_default(self, default: dict) -> None:
        """Apply defaults, setting any unset fields.

        We use pairs of (name, _name) in order to make type checking happy.

        Args:
            default (Self): Stream containing default values for all optional fields.
        """
        if not (self.remote or self._local):
            raise ValueError('`remote` and/or `local` path must be provided')

        if not self.split:
            self.split = default['split'] or ''
        if self._download_retry is None:
            self.download_retry = default['download_retry']
        if self._download_timeout is None:
            self.download_timeout = default['download_timeout']
        if self.validate_hash is None:
            self.validate_hash = default['validate_hash'] or None
        if self._keep_zip is None:
            self.keep_zip = default['keep_zip']
            self.safe_keep_zip = default['keep_zip'] or self.remote in {None, self.local}
