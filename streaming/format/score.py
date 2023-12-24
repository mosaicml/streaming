# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A dataset, or sub-dataset if mixing, from which we stream/cache samples."""

import os
from hashlib import blake2s
from tempfile import gettempdir
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
        if local is not None:
            self.local = local
        self.split = split or ''

        if download_retry is not None:
            if download_retry < 0:
                raise ValueError('`download_retry` must be non-negative')
            self.download_retry = download_retry

        if download_timeout is not None:
            if download_timeout <= 0:
                raise ValueError('`download_timeout` must be positive')
            self.download_timeout = download_timeout

        self.validate_hash = validate_hash

        if keep_zip is not None:
            self.keep_zip = keep_zip
            self.safe_keep_zip = self.keep_zip or self.remote in {None, self.local}

    def _generate_local(self, remote: str, split: Optional[str]) -> str:
        """Derive a local dirname deterministically from remote and optional split.

        Args:
            remote (str): Remote path. Must exist.
            split (str, optional): Optional split.

        Returns:
            str: Local path.
        """
        data = remote.encode('utf-8')
        hex_digest = blake2s(data, digest_size=16).hexdigest()
        return os.path.join(gettempdir(), hex_digest, self.split)

    def apply_default(self, default: dict) -> None:
        """Apply defaults, setting any unset fields.

        We use pairs of (name, _name) in order to make type checking happy.

        Args:
            default (Self): Stream containing default values for all optional fields.
        """
        if not self.split:
            self.split = default['split'] or ''

        if not hasattr(self, 'local'):
            if self.remote is None:
                raise ValueError('`remote` and/or `local` path must be provided')
            self.local = self._generate_local(self.remote, self.split)

            if not get_local_rank():
                if os.path.exists(self.local):
                    raise ValueError(
                        f'Could not create a temporary local directory {self.local} . Either ' +
                        f'delete the directory or specify a unique local directory with the ' +
                        f'`local` value.')
                os.makedirs(self.local)
            barrier()

        if not hasattr(self, 'download_retry'):
            self.download_retry = default['download_retry']

        if not hasattr(self, 'download_timeout'):
            self.download_timeout = default['download_timeout']

        if self.validate_hash is None:
            self.validate_hash = default['validate_hash'] or None

        if not hasattr(self, 'keep_zip'):
            self.keep_zip = default['keep_zip']
            self.safe_keep_zip = default['keep_zip'] or self.remote in {None, self.local}
