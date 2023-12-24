# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A dataset, or sub-dataset if mixing, from which we stream/cache samples."""

import os
from hashlib import blake2s
from tempfile import gettempdir
from typing import List, Optional, Sequence, Union
from warnings import warn

from streaming.distributed import barrier, get_local_rank
from streaming.hashing import is_hash
from streaming.phasing import get_phasings, get_safe_phasing, is_phasing
from streaming.util.auto import Auto, auto
from streaming.util.shorthand import normalize_duration


def _normalize_download_retry(download_retry: int) -> int:
    """Normalize ``download_retry``.

    Args:
        download_retry (int): Input download retry.

    Returns:
        int: Normalized download retry.
    """
    if download_retry < 0:
        raise ValueError(f'Download retry must be non-negative, but got: {download_retry}.')
    return download_retry


def _normalize_download_timeout(download_timeout: Union[str, float]) -> float:
    """Normalize ``download_timeout``.

    Args:
        download_timeout (str | float): Input download timeout.

    Returns:
        float: Normalized download timeout.
    """
    norm_download_timeout = normalize_duration(download_timeout)
    if norm_download_timeout <= 0:
        raise ValueError(f'Download timeout must be positive, but got: {download_timeout}.')
    return norm_download_timeout


def _normalize_hash_algos(hash_algos: Optional[Union[str, Sequence[str], Auto]],
                          validate_hash: Optional[str]) -> List[str]:
    """Normalize ``hash_algos`` and ``validate_hash`` (deprecated argument).

    Args:
        hash_algos (str | Sequence[str] | Auto, optional): Input hash algos.
        validate_hash (str, optional): Input validate hash.

    Returns:
        List[str]: Normalized hash algos.
    """
    # Normalize `hash_algos`.
    if not hash_algos:
        norm_hash_algos = None
    elif isinstance(hash_algos, str):
        norm_hash_algos = [hash_algos]
    elif isinstance(hash_algos, Sequence):
        norm_hash_algos = list(hash_algos)
    else:
        norm_hash_algos = None

    # Normalize `validate_hash`.
    if validate_hash:
        warn(f'`validate_hash` is deprecated. Please use `hash_algos` instead, which also ' +
             f'accepts a ranked list specifying the hashing algorithms to attempt to apply.')
        norm_validate_hash = [validate_hash]
    else:
        norm_validate_hash = None

    # Compare and combine normalized `hash_algos` and normalized `validate_hash`.
    if not norm_hash_algos:
        if not norm_validate_hash:
            algos = []
        else:
            algos = norm_validate_hash
    else:
        if not norm_validate_hash:
            algos = norm_hash_algos
        else:
            if norm_hash_algos != norm_validate_hash:
                raise ValueError(f'You have specified hashes to check in both the old way and ' +
                                 f'the new way, and also differently: `hash_algos` = ' +
                                 f'{hash_algos}, `validate_hash` = {validate_hash}.')
            algos = norm_hash_algos

    # Check each hash algo.
    for algo in algos:
        if not is_hash(algo):
            raise ValueError('Unknown hash algorithm: {algo}.')

    return algos


def _normalize_keep_zip(keep_zip: bool) -> str:
    """Normalize ``keep_zip`` (deprecated argument).

    Args:
        keep_zip (bool): Input keep zip.

    Returns:
        str: Normalized phasing.
    """
    warn(f'`keep_zip` is deprecated. Please use `keep_old_phases="src"` instead. You stream ' +
         f'the earliest form of a file (say, zipped), and access samples from its latest ' +
         f'form (say, after unzipping). The intent of the argument is: do we keep that ' +
         f'earliest form, so we will be able to stream with this dir as a remote? Options ' +
         f'for `keep_old_phases` are {sorted(get_phasings())}.')
    return 'src' if keep_zip else 'nil'


def _normalize_keep_old_phases(keep_old_phases: Optional[str], keep_zip: Optional[bool]) -> str:
    """Normalize ``keep_old_phases`` and ``keep_zip`` (deprecated argument).

    Args:
        keep_old_phases (str, optional): Input keep old phases.
        keep_zip (bool, optional): Input keep zip.

    Returns:
        Normalized phasing.
    """
    if keep_old_phases is None:
        if keep_zip is None:
            phasing = 'nil'
        else:
            phasing = _normalize_keep_zip(keep_zip)
    else:
        if keep_zip is None:
            phasing = keep_old_phases
        else:
            norm_keep_zip = _normalize_keep_zip(keep_zip)
            if keep_old_phases != norm_keep_zip:
                raise ValueError(f'You have specified old phases to keep in both the old way ' +
                                 f'and the new way, and also differently: `keep_old_phases` = ' +
                                 f'{keep_old_phases}, `keep_zip` = {keep_zip}.')
            phasing = keep_old_phases

    if not is_phasing(phasing):
        raise ValueError('Unknown phasing (i.e., `keep_old_phases` or `keep_zip`): {phasing}.')

    return phasing


def _generate_local(remote: str, split: Optional[str]) -> str:
    """Derive a local dirname deterministically from remote and optional split.

    Args:
        remote (str): Remote path. Must exist.
        split (str, optional): Optional split.

    Returns:
        str: Local path.
    """
    data = remote.encode('utf-8')
    hex_digest = blake2s(data, digest_size=16).hexdigest()
    return os.path.join(gettempdir(), hex_digest, split or '')


class StreamCore:
    """The core configuration of a Streaming dataset directory (Stream).

    A StreamingDataset is composed of one/multiple/many Streams.

    Notes:
      * Paths: You must provide ``remote`` and/or ``local``. If no ``remote``, the dataset must be
        cached. If no ``local``, it deterministically picks ``local`` based on the other paths.
      * Splits: This is implemented as sub-path which is appended to ``remote`` and/or ``local`` in
        order to derive the root of this Streaming dataset directory (Stream), which all other
        dataset paths descend from. E.g., ``/path/to/dataset/index.json`` if ``split=None``, vs
        ``/path/to/dataset/train/index.json`` if ``split='train'``.
      * Hashing: Trying a hash algorithm means if the Streaming index records the expected hex
        digest for this hash of this file, we apply the hash, compare the result to the expected,
        and then we are done: either exit success on match, or raise an error on mismatch. If we
        are given hash algorithms to apply but the index notes none of them for a file, we raise an
        error. Typically, because of the somewhat severe performance impact, hashes are not used
        in training.
      * Phasing: Streaming downloads shards as their first phase and accesses samples from their
        last phase, to which they are converted on the fly. Do we keep the old phases (until the
        shard is evicted)? Options are ``nil``, ``src``, and ``all``. ``safe_keep_old_phases`` is
        derived from ``keep_old_phases`` -- it is the same, unless there is no separate remote, in
        which case ``nil`` is converted to ``src`` (i.e., keep first phase) in order to prevent
        making the streaming dataset directory un-streamable by using it.

    Args:
        remote (str, optional): Remote path to stream the dataset from. If ``None``, dataset must
            be complete locally. Defaults to ``None``.
        local (str, optional): Local working directory to stream the dataset to. Uses a temp
            directory if not set. Defaults to ``None``.
        split (str | Auto, optional): Which dataset split to use, if any. Set to ``auto`` to
            inherit from StreamingDataset. Defaults to ``auto``.
        download_retry (int | Auto): Number of download re-attempts before raising an error. Set to
            ``auto`` to inherit from StreamingDataset. Defaults to ``auto``.
        download_timeout (str | float | Auto, optional): Time in seconds to wait for a file
            download to complete before raising an error. Streaming duration shorthand (e.g.,
            ``1m23s``) is also accepted. Set to ``auto`` to inherit from StreamingDataset. Defaults
            to ``auto``.
        hash_algos (str | Sequence[str] | Auto, optional): Ranked list of hashing algorithms to
            try. Set to ``auto`` to inherit from StreamingDataset. Defaults to ``auto``.
        validate_hash (str, optional): Deprecated. See ``hash_algos``. Defaults to ``None``.
        keep_old_phases (str | Auto): Which old phases of shard files to cache (until shard
            eviction). Must be one of ``nil``, ``src``, or ``all``. Set to ``auto`` to inherit from
            StreamingDataset. If ``None``, uses ``keep_zip``, falling back to ``nil``. Defaults to
            ``None``.
        keep_zip (bool, optional): Deprecated. See ``keep_old_phases``. Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[Union[str, Auto]] = auto,
        download_retry: Union[int, Auto] = auto,
        download_timeout: Union[str, float, Auto] = auto,
        hash_algos: Optional[Union[str, Sequence[str], Auto]] = auto,
        validate_hash: Optional[str] = None,
        keep_old_phases: Optional[Union[str, Auto]] = auto,
        keep_zip: Optional[bool] = None,
    ) -> None:
        self.remote = remote

        if local is not None:
            self.local = local

        if remote is None and local is None:
            raise ValueError('Remote and/or local paths must be provided.')

        if not isinstance(split, Auto):
            self.split = split

        if not isinstance(download_retry, Auto):
            self.download_retry = _normalize_download_retry(download_retry)

        if not isinstance(download_timeout, Auto):
            self.download_timeout = _normalize_download_timeout(download_timeout)

        if not isinstance(hash_algos, Auto) or validate_hash:
            self.hash_algos = _normalize_hash_algos(hash_algos, validate_hash)

        if not isinstance(keep_old_phases, Auto):
            self.keep_old_phases = _normalize_keep_old_phases(keep_old_phases, keep_zip)
        elif keep_zip is not None:
            self.keep_old_phases = _normalize_keep_zip(keep_zip)

        if hasattr(self, 'keep_old_phases') and hasattr(self, 'local'):
            self.safe_keep_old_phases = get_safe_phasing(self.keep_old_phases, self.remote,
                                                         self.local)

    def apply_defaults(
        self,
        *,
        split: Optional[str],
        download_retry: int,
        download_timeout: Union[str, float],
        hash_algos: Optional[Union[str, Sequence[str]]],
        validate_hash: Optional[str],
        keep_old_phases: Optional[str],
        keep_zip: Optional[bool],
    ) -> None:
        """Apply defaults, setting any unknown fields.

        Args:
            split (str, optional): Which dataset split to use, if any.
            download_retry (int): Number of download re-attempts before raising an error.
            download_timeout (str | float, optional): Time in seconds to wait for a file download
                to complete before raising an error. Streaming duration shorthand (e.g., ``1m23s``)
                is also accepted.
            hash_algos (str | Sequence[str], optional): Ranked list of hashing algorithms to try.
            validate_hash (str, optional): Deprecated. See ``hash_algos``.
            keep_old_phases (str, optional): Which old phases of shard files to cache (until shard
                eviction). If set, must be one of ``nil``, ``src``, or ``all``. If ``None``, uses
                ``keep_zip``, falling back to ``nil``.
            keep_zip (bool, optional): Deprecated. See ``keep_old_phases``.
        """
        if not hasattr(self, 'split'):
            self.split = split

        if not hasattr(self, 'local'):
            if self.remote is None:
                raise ValueError('`remote` and/or `local` path must be provided.')
            self.local = _generate_local(self.remote, self.split)

            # TODO: why does this code exist?
            if not get_local_rank():
                if os.path.exists(self.local):
                    raise ValueError(
                        f'Could not create a temporary local directory {self.local} . Either ' +
                        f'delete the directory or specify a unique local directory with the ' +
                        f'`local` value.')
                os.makedirs(self.local)
            barrier()

        if not hasattr(self, 'download_retry'):
            self.download_retry = _normalize_download_retry(download_retry)

        if not hasattr(self, 'download_timeout'):
            self.download_timeout = _normalize_download_timeout(download_timeout)

        if not hasattr(self, 'hash_algos'):
            self.hash_algos = _normalize_hash_algos(hash_algos, validate_hash)

        if not hasattr(self, 'keep_old_phases'):
            self.keep_old_phases = _normalize_keep_old_phases(keep_old_phases, keep_zip)

        if not hasattr(self, 'safe_keep_old_phases'):
            self.safe_keep_old_phases = get_safe_phasing(self.keep_old_phases, self.remote,
                                                         self.local)

    @property
    def safe_keep_zip(self) -> bool:
        """Derive ``safe_keep_zip`` for existing code.

        Returns:
            bool: Whether to keep the zip phase of files.
        """
        return self.safe_keep_old_phases != 'nil'
