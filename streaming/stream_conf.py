# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Configures a StreamingDataset directory."""

import os
from collections.abc import Sequence
from hashlib import blake2s
from tempfile import gettempdir
from typing import Any, Dict, List, Optional, Union

from streaming.format.base.phaser import Phaser
from streaming.util.auto import Auto
from streaming.util.shorthand import normalize_bytes, normalize_count, normalize_duration


def _derive_local(remote: str, split: Optional[str]) -> str:
    """Derive a local dirname deterministically from remote and optional split.

    Args:
        remote (str): Remote path. Must exist.
        split (str, optional): Optional split.

    Returns:
        str: Local path.
    """
    data = remote.encode('utf-8')
    hex_digest = blake2s(data, digest_size=16).hexdigest()
    return os.path.join(gettempdir(), 'streaming', 'auto_local', hex_digest, split or '')


def _get_bool(arg: bool) -> bool:
    """Normalize a bool arg.

    Args:
        arg (bool): Input arg.

    Returns:
        bool: Normalized arg.
    """
    return arg


def _get_bytes(arg: Union[None, str, int]) -> Optional[float]:
    """Normalize a bytes arg.

    Args:
        arg (None | str | int): Input arg.

    Returns:
        float: Normalized arg.
    """
    return normalize_bytes(arg) if arg is not None else None


def _get_count(arg: Union[None, str, int]) -> Optional[int]:
    """Normalize a count arg.

    Args:
        arg (str | int): Input arg.

    Returns:
        int: Normalized arg.
    """
    return normalize_count(arg) if arg is not None else None


def _get_duration(arg: Union[None, str, float]) -> Optional[float]:
    """Normalize a duration arg.

    Args:
        arg (None | str | float): Input arg.

    Returns:
        float: Normalized arg.
    """
    return normalize_duration(arg) if arg is not None else None


def _get_hash_algos(arg: Union[None, str, Sequence[str]]) -> List[str]:
    """Normalize a hash algo(s) arg.

    Args:
        arg (None | str | Sequence[str]): Input arg.

    Returns:
        List[str]: Normalized arg.
    """
    if arg is None:
        return []
    elif isinstance(arg, str):
        return [arg]
    else:
        return list(arg)


def _get_phaser(arg: Union[None, str, Sequence[str], Dict[str, Optional[bool]], Phaser]) -> Phaser:
    """Normalize a keep phases arg.

    Args:
        arg (None | str | Sequence[str] | Dict[str, Optional[bool]] | Phaser): Input arg.

    Returns:
        Phaser: Normalized arg.
    """
    if arg is None:
        return Phaser()
    elif isinstance(arg, str):
        return Phaser(**{arg: True})
    elif isinstance(arg, Sequence):
        return Phaser(**dict(zip(arg, [True] * len(arg))))
    elif isinstance(arg, dict):
        return Phaser(**arg)
    else:
        return arg


def _get_keep_zip(arg: bool) -> Phaser:
    """Normalize a keep zip arg.

    Args:
        arg (bool): Whether to keep zip.

    Returns:
        Phaser: Normalized arg.
    """
    return Phaser(storage=arg)


class StreamConf:
    """Configures a StreamingDataset directory.

    Args:
        remote (str, optional): Remote path to stream the dataset from. If ``None``, dataset must
            be complete locally. Defaults to ``None``.
        local (str, optional): Local working directory to stream the dataset to. Uses a
            deterministically-calculated temp directory if not set. Defaults to ``None``.
        split (str | Auto, optional): Which dataset split sub-path to use, if any. Set to ``Auto``
            to inherit from StreamingDataset. Defaults to ``Auto()``.
        allow_schema_mismatch (bool | Auto): If ``True``, continue if schemas mismatch across
            shards, streams, or the whole dataset. If ``False``, raises if schemas mismatch. Set to
            ``Auto`` to inherit from StreamingDataset. Defaults to ``Auto()``.
        allow_unsafe_types (bool | Auto): If ``True``, continue if unsafe type(s) are encountered
            in shard(s). If ``False``, raises if unsafe type(s) encountered. Set to ``Auto`` to
            inherit from StreamingDataset. Defaults to ``Auto()``.
        allow_unchecked_resumption (bool | Auto): If ``True``, upon resume, accept and use shard
            file phases that we are unable to check the size/hash(es) of. If ``False``, upon
            resume, drop such files, to regenerate on the fly when needed. Set to ``Auto`` to
            inherit from StreamingDataset. Defaults to ``Auto()``.
        download_retry (str | int | Auto): Number of download re-attempts before raising an error.
            Set to ``Auto`` to inherit from StreamingDataset. Defaults to ``Auto()``.
        download_timeout (None | str | float | Auto): Time to wait for a file download to complete
            before raising an error, if any. Set to ``None`` for no limit. Streaming duration
            shorthand (e.g., ``1m23s``) is also accepted. Numeric values are in seconds. Set to
            ``Auto`` to inherit from StreamingDataset. Defaults to ``Auto()``.
        download_max_size (None, str | int | Auto): Maximum size of an individual download, if any.
            This is used to prevent over-large shard files from cripping Streaming performance. Set
            to ``None`` for no size limit. Set to ``str`` to specify the limit using Streaming
            bytes shorthand. Set to ``int`` to specify bytes. Set to ``Auto`` to inherit from
            StreamingDataset. Defaults to ``Auto()``.
        validate_hash (None | str | Sequence[str] | Auto): Ranked list of hashing algorithms to
            apply if expected digest is available. Set to ``Auto`` to inherit from
            StreamingDataset. Defaults to ``Auto()``.
        keep_phases (str | Sequence[str] | Dict[str, Optional[bool]] | Phaser | Auto): Which phases
            to keep and to drop upon conversion, given either by intended use case or literally.
            Specified as a single use or phase to keep, a sequence of uses or phases to keep, a
            mapping of uses or phases to whether to keep or drop, a ``Phaser`` (which performs the
            same keeping or dropping), or ``Auto`` to inherit from StreamingDataset. Defaults to
            ``Auto()``.
        kwargs (Dict[str, Any]): Deprecated arguments, if any: ``keep_zip``.
    """

    def __init__(
        self,
        *,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Union[None, str, Auto] = Auto(),
        allow_schema_mismatch: Union[bool, Auto] = Auto(),
        allow_unsafe_types: Union[bool, Auto] = Auto(),
        allow_unchecked_resumption: Union[bool, Auto] = Auto(),
        download_retry: Union[str, int, Auto] = Auto(),
        download_timeout: Union[None, str, float, Auto] = Auto(),
        download_max_size: Union[None, str, int, Auto] = Auto(),
        validate_hash: Union[None, str, Sequence[str], Auto] = Auto(),
        keep_phases: Union[None, str, Sequence[str], Dict[str, Optional[bool]], Phaser, Auto] = \
            Auto(),
        **kwargs: Dict[str, Any],
    ) -> None:
        # 1. Maybe set remote, local, and split.
        self.remote = os.path.abspath(remote) if remote is not None else None
        if local is not None:
            self.local = os.path.abspath(local)
        if not split:
            split = None
        if not isinstance(split, Auto):
            self.split = split

        # 2. Maybe derive local from provided remote and split.
        if local is None:
            if remote is not None:
                if split is None:
                    self.local = _derive_local(remote, split)
            else:
                raise ValueError(f'You must provide `remote` and/or `local`.')

        # 3.A. Maybe override init args.
        if not isinstance(allow_schema_mismatch, Auto):
            self.allow_schema_mismatch = _get_bool(allow_schema_mismatch)
        if not isinstance(allow_unsafe_types, Auto):
            self.allow_unsafe_types = _get_bool(allow_unsafe_types)
        if not isinstance(allow_unchecked_resumption, Auto):
            self.allow_unchecked_resumption = _get_bool(allow_unchecked_resumption)

        # 3.B. Maybe override download args.
        if not isinstance(download_retry, Auto):
            self.download_retry = _get_count(download_retry)
        if not isinstance(download_timeout, Auto):
            self.download_timeout = _get_duration(download_timeout)
        if not isinstance(download_max_size, Auto):
            self.download_max_size = _get_bytes(download_max_size)
        if not isinstance(validate_hash, Auto):
            self.check_hashes = _get_hash_algos(validate_hash)

        # 3.C. Maybe override phaser (phase keeper), then cache self.keep_zip in case needed.
        if not isinstance(keep_phases, Auto):
            self.keep_phases = _get_phaser(keep_phases)
            self.safe_keep_phases = self.keep_phases.to_safe()
        self.keep_zip = kwargs.get('keep_zip')

    def apply_defaults(
        self,
        *,
        split: Union[None, str, Auto] = Auto(),
        allow_schema_mismatch: Union[bool, Auto] = Auto(),
        allow_unsafe_types: Union[bool, Auto] = Auto(),
        allow_unchecked_resumption: Union[bool, Auto] = Auto(),
        download_retry: Union[str, int, Auto] = Auto(),
        download_timeout: Union[str, float, Auto] = Auto(),
        download_max_size: Union[str, int, Auto] = Auto(),
        validate_hash: Union[None, str, Sequence[str], Auto] = Auto(),
        keep_phases: Union[None, str, Sequence[str], Dict[str, Optional[bool]], Phaser, Auto] = \
            Auto(),
        **kwargs: Dict[str, Any],
    ) -> None:
        """Inherit from our owning StreamingDataset the values of any arguments left as auto.

        Args:
            remote (str, optional): Remote path to stream the dataset from. If ``None``, dataset
                must be complete locally. Defaults to ``None``.
            local (str, optional): Local working directory to stream the dataset to. Uses a
                deterministically-calculated temp directory if not set. Defaults to ``None``.
            split (str | Auto, optional): Which dataset split sub-path to use, if any. Set to
                ``Auto`` to not inherit from StreamingDataset. Defaults to ``Auto()``.
            allow_schema_mismatch (bool | Auto): If ``True``, continue if schemas mismatch across
                shards, streams, or the whole dataset. If ``False``, raises if schemas mismatch.
                Set to ``Auto`` to not inherit from StreamingDataset. Defaults to ``Auto()``.
            allow_unsafe_types (bool | Auto): If ``True``, continue if unsafe type(s) are
                encountered in shard(s). If ``False``, raises if unsafe type(s) encountered. Set to
                ``Auto`` to not inherit from StreamingDataset. Defaults to ``Auto()``.
            allow_unchecked_resumption (bool | Auto): If ``True``, upon resume, accept and use
                shard file phases that we are unable to check the size/hash(es) of. If ``False``,
                upon resume, drop such files, to regenerate on the fly when needed. Set to ``Auto``
                to not inherit from StreamingDataset. Defaults to ``Auto()``.
            download_retry (stgr | int | Auto): Number of download re-attempts before raising an
                error. Set to ``Auto`` to not inherit from StreamingDataset. Defaults to
                ``Auto()``.
            download_timeout (None | str | float | Auto): Time to wait for a file download to
                complete before raising an error, if any. Set to ``None`` for no limit. Streaming
                duration shorthand (e.g., ``1m23s``) is also accepted. Numeric values are in
                seconds. Set to ``Auto`` to not inherit from StreamingDataset. Defaults to
                ``Auto()``.
            download_max_size (None, str | int | Auto): Maximum size of an individual download, if
                any. This is used to prevent over-large shard files from cripping Streaming
                performance. Set to ``None`` for no size limit. Set to ``str`` to specify the limit
                using Streaming bytes shorthand. Set to ``int`` to specify bytes. Set to ``Auto``
                to not inherit from StreamingDataset. Defaults to ``Auto()``.
            validate_hash (None | str | Sequence[str] | Auto): Ranked list of hashing algorithms to
                apply if expected digest is available. Set to ``Auto`` to not inherit from
                StreamingDataset. Defaults to ``Auto()``.
            keep_phases (str | Sequence[str] | Dict[str, Optional[bool]] | Phaser | Auto): Which
                phases to keep and to drop upon conversion, given either by intended use case or
                literally. Specified as a single use or phase to keep, a sequence of uses or phases
                to keep, a mapping of uses or phases to whether to keep or drop, a ``Phaser``
                (which performs the same keeping or dropping), or ``Auto`` to not inherit from
                StreamingDataset. Defaults to ``Auto()``.
            kwargs (Dict[str, Any]): Deprecated arguments, if any: ``keep_zip``.
        """
        err_pattern = 'You must set {txt} in `Stream.__init__()` and/or `Stream.apply_defaults()`.'
        err_unset = lambda txt: ValueError(err_pattern.format(txt=txt))
        err_unset_arg = lambda arg: ValueError(err_pattern.format(txt=f'`{arg}`'))

        # 1. Maybe default split.
        if not hasattr(self, 'split'):
            if not isinstance(split, Auto):
                self.split = split
            else:
                raise err_unset_arg('split')

        # 2. Maybe derive local.
        if not hasattr(self, 'local'):
            if self.remote is not None:
                self.local = _derive_local(self.remote, self.split)
            else:
                raise err_unset('`remote` and/or `local`')

        # 3.A. Maybe default init args.
        if not hasattr(self, 'allow_schema_mismatch'):
            if not isinstance(allow_schema_mismatch, Auto):
                self.allow_schema_mismatch = _get_bool(allow_schema_mismatch)
            else:
                raise err_unset_arg('allow_schema_mismatch')

        if not hasattr(self, 'allow_unsafe_types'):
            if not isinstance(allow_unsafe_types, Auto):
                self.allow_unsafe_types = _get_bool(allow_unsafe_types)
            else:
                raise err_unset_arg('allow_unsafe_types')

        if not hasattr(self, 'allow_unchecked_resumption'):
            if not isinstance(allow_unchecked_resumption, Auto):
                self.allow_unchecked_resumption = _get_bool(allow_unchecked_resumption)
            else:
                raise err_unset_arg('allow_unchecked_resumption')

        # 3.B. Maybe default download args.
        if not hasattr(self, 'download_retry'):
            if not isinstance(download_retry, Auto):
                self.download_retry = _get_count(download_retry)
            else:
                raise err_unset_arg('download_retry')

        if not hasattr(self, 'download_timeout'):
            if not isinstance(download_timeout, Auto):
                self.download_timeout = _get_duration(download_timeout)
            else:
                raise err_unset_arg('download_timeout')

        if not hasattr(self, 'download_max_size'):
            if not isinstance(download_max_size, Auto):
                self.download_max_size = _get_bytes(download_max_size)
            else:
                raise err_unset_arg('download_max_size')

        if not hasattr(self, 'check_hashes'):
            if not isinstance(validate_hash, Auto):
                self.check_hashes = _get_hash_algos(validate_hash)
            else:
                raise err_unset_arg('validate_hash')

        # 3.C. Maybe default phaser (phase keeper), then delete self.keep_zip.
        if not hasattr(self, 'keep_phases'):
            if not isinstance(keep_phases, Auto):
                self.keep_phases = _get_phaser(keep_phases)
            elif isinstance(self.keep_zip, bool):
                self.keep_phases = _get_keep_zip(self.keep_zip)
            else:
                raise err_unset_arg('keep_phases')

        # We delete self.keep_zip because files are maximally three-phase not two-phase now, and
        # keep_zip might mean the user wants to literally keep zip phases only (but there is known
        # good reason to want this), or it might mean the user wants to keep the original phases
        # of files instead, zipped or not (for which the knob is self.keep_phases.storage).
        del self.keep_zip

        # 4. Derive safe_keep_phases from keep_phases.
        self.safe_keep_phases = self.keep_phases.to_safe()
