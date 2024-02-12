# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Configures a Stream directory."""

import os
from collections.abc import Sequence as SequenceClass
from copy import deepcopy
from hashlib import blake2s
from tempfile import gettempdir
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from streaming.format.base.phaser import Phaser
from streaming.util.auto import Auto
from streaming.util.shorthand import normalize_bytes, normalize_count, normalize_duration

__all__ = ['StreamDirConf']


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
    return os.path.join(gettempdir(), 'streaming', 'local', hex_digest)


def _get_bool(arg: bool) -> bool:
    """Normalize a bool arg.

    Args:
        arg (bool): Input arg.

    Returns:
        bool: Normalized arg.
    """
    return arg


def _get_bytes(arg: Union[None, str, int]) -> Optional[int]:
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


def _get_duration(arg: Union[None, str, float]) -> float:
    """Normalize a duration arg.

    Args:
        arg (None | str | float): Input arg.

    Returns:
        float: Normalized arg.
    """
    return normalize_duration(arg) if arg is not None else float('inf')


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


def _get_keep_phases_kwarg(txt: str) -> Tuple[str, bool]:
    """Parse one ``keep_phases`` kwarg.

    Args:
        txt (str): Input arg.

    Returns:
        Tuple[str, bool]: Normalized arg.
    """
    if txt.startswith('+'):
        pair = txt[1:], True
    elif txt.startswith('-'):
        pair = txt[1:], False
    else:
        pair = txt, True
    return pair


def _get_keep_phases(arg: Union[None, str, Sequence[str], Dict[str, bool], Phaser]) -> Phaser:
    """Normalize a keep phases arg.

    Args:
        arg (None | str | Sequence[str] | Dict[str, bool] | Phaser): Input arg.

    Returns:
        Phaser: Normalized arg.
    """
    if arg is None:
        return Phaser()
    elif isinstance(arg, str):
        key, value = _get_keep_phases_kwarg(arg)
        kwargs = {key: value}
        return Phaser(**kwargs)
    elif isinstance(arg, dict):
        return Phaser(**arg)
    elif isinstance(arg, SequenceClass):
        pairs = []
        for txt in arg:
            pair = _get_keep_phases_kwarg(txt)
            pairs.append(pair)
        kwargs = dict(zip(*pairs))
        return Phaser(**kwargs)
    else:
        return arg


def _get_keep_zip(arg: bool) -> Phaser:
    """Normalize a keep zip arg.

    Args:
        arg (bool): Whether to keep zip.

    Returns:
        Phaser: Normalized arg.
    """
    return Phaser(persistent=arg)


def _get_phaser(
    init_keep_phases: Union[None, str, Sequence[str], Dict[str, bool], Phaser, Auto],
    init_keep_zip: Optional[bool],
    default_keep_phases: Union[None, str, Sequence[str], Dict[str, bool], Phaser, Auto],
    default_keep_zip: Optional[bool],
) -> Phaser:
    """Given all Phaser-related args, determine the Phaser.

    Args:
        init_keep_phases (None | str | Sequence[str] | Dict[str, bool] | Phaser | Auto):
            ``keep_phases`` as provided up front at Stream init time.
        init_keep_zip (None | bool): ``keep_zip`` as provided up front at Stream init time.
        default_keep_phases (None | str | Sequence[str] | Dict[str, bool] | Phaser | Auto):
            ``keep_phases`` as provided later in ``Stream.apply_defaults()`` at StreamingDataset
            init time.
        default_keep_zip (None | bool): ``keep_zip`` as provided later in
            ``Stream.apply_defaults()`` at StreamingDataset init time.
    """
    # There is no legitimate reason why you would be mixing old-style and new-style args. Such
    # cases are bugs. Let's rule them all out first.
    is_set = lambda arg: (arg is not None) and not isinstance(arg, Auto)
    set_keep_phases = is_set(init_keep_phases) or is_set(default_keep_phases)
    set_keep_zip = is_set(init_keep_zip) or is_set(default_keep_zip)
    if set_keep_phases and set_keep_zip:
        clean = lambda txt: ' '.join(txt.split())
        raise ValueError(
            clean('''
            The argument `keep_zip` is deprecated.

            Please update your code to use `keep_phases` instead.

            If you wanted to cache the original, potentially compressed, forms of shard files,
            which is the phase that is stored and streamed, use `keep_phases="persistent"'.

            If you wanted to cache the compressed forms of shard files, which are not required to
            exist, use `keep_phases="zip"`.
        '''))

    # First priority is Stream init keep_phases.
    if init_keep_phases is not None and not isinstance(init_keep_phases, Auto):
        return _get_keep_phases(init_keep_phases)

    # Second priority is StreamingDataset init (Stream default) keep_phases.
    if default_keep_phases is not None and not isinstance(default_keep_phases, Auto):
        return _get_keep_phases(default_keep_phases)

    # Third priority is Stream init keep_zip.
    if init_keep_zip is not None:
        return _get_keep_zip(init_keep_zip)

    # Fourth priority is Stream init keep_zip.
    if default_keep_zip is not None:
        return _get_keep_zip(default_keep_zip)

    # Default.
    return Phaser()


class StreamDirConf:
    """Configures a Stream.

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
        download_max_size (None | str | int | Auto): Maximum size of an individual download, if any.
            This is used to prevent over-large shard files from cripping Streaming performance. Set
            to ``None`` for no size limit. Set to ``str`` to specify the limit using Streaming
            bytes shorthand. Set to ``int`` to specify bytes. Set to ``Auto`` to inherit from
            StreamingDataset. Defaults to ``Auto()``.
        validate_hash (None | str | Sequence[str] | Auto): Ranked list of hashing algorithms to
            apply if expected digest is available. Set to ``Auto`` to inherit from
            StreamingDataset. Defaults to ``Auto()``.
        keep_phases (None | str | Sequence[str] | Dict[str, bool] | Phaser | Auto): After a phase
            transition of a shard file, do we keep the old form of the file or garbage collect it?
            Provided as one of: (1) ``None`` for defaults, (2) the single use case or phase to keep,
            (3) a sequence giving the use cases or phases to keep, (4) Phaser kwargs (a mapping of
            use case or phase to whether it must be kept, (5) a Phaser object, or ``Auto`` to
            inherit from StreamingDataset. All code paths result in a ``Phaser``. Defaults to
            ``Auto()``.
        kwargs (Any): Any unsupported (for forward compat) or deprecated args.
    """

    index_relative_path = 'index.json'

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
        keep_phases: Union[None, str, Sequence[str], Dict[str, bool], Phaser, Auto] = Auto(),
        **kwargs: Any,
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

        # 3. Maybe derive remote/local index paths.
        if self.remote is None:
            self.remote_index_path = None
        elif hasattr(self, 'split'):
            self.remote_index_path = os.path.join(self.remote, self.split or '',
                                                  self.index_relative_path)
        if hasattr(self, 'local') and hasattr(self, 'split'):
            self.local_index_path = os.path.join(self.local, self.split or '',
                                                 self.index_relative_path)

        # 5. Maybe override init args.
        if not isinstance(allow_schema_mismatch, Auto):
            self.allow_schema_mismatch = _get_bool(allow_schema_mismatch)
        if not isinstance(allow_unsafe_types, Auto):
            self.allow_unsafe_types = _get_bool(allow_unsafe_types)
        if not isinstance(allow_unchecked_resumption, Auto):
            self.allow_unchecked_resumption = _get_bool(allow_unchecked_resumption)

        # 6. Maybe override download args.
        if not isinstance(download_retry, Auto):
            self.download_retry = _get_count(download_retry)
        if not isinstance(download_timeout, Auto):
            self.download_timeout = _get_duration(download_timeout)
        if not isinstance(download_max_size, Auto):
            self.download_max_size = _get_bytes(download_max_size)
        if not isinstance(validate_hash, Auto):
            self.check_hashes = _get_hash_algos(validate_hash)

        # 7. Maybe override phaser (phase keeper), then cache self.keep_zip in case needed.
        self.init_keep_phases = keep_phases
        self.init_keep_zip = kwargs.get('keep_zip')
        self.phaser: Phaser
        self.safe_phaser: Phaser

    def apply_defaults(
        self,
        *,
        split: Union[None, str, Auto] = Auto(),
        allow_schema_mismatch: Union[bool, Auto] = Auto(),
        allow_unsafe_types: Union[bool, Auto] = Auto(),
        allow_unchecked_resumption: Union[bool, Auto] = Auto(),
        download_retry: Union[str, int, Auto] = Auto(),
        download_timeout: Union[None, str, float, Auto] = Auto(),
        download_max_size: Union[None, str, int, Auto] = Auto(),
        validate_hash: Union[None, str, Sequence[str], Auto] = Auto(),
        keep_phases: Union[None, str, Sequence[str], Dict[str, bool], Phaser] = None,
        **kwargs: Any,
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
            download_max_size (None | str | int | Auto): Maximum size of an individual download, if
                any. This is used to prevent over-large shard files from cripping Streaming
                performance. Set to ``None`` for no size limit. Set to ``str`` to specify the limit
                using Streaming bytes shorthand. Set to ``int`` to specify bytes. Set to ``Auto``
                to not inherit from StreamingDataset. Defaults to ``Auto()``.
            validate_hash (None | str | Sequence[str] | Auto): Ranked list of hashing algorithms to
                apply if expected digest is available. Set to ``Auto`` to not inherit from
                StreamingDataset. Defaults to ``Auto()``.
            keep_phases (None | str | Sequence[str] | Dict[str, bool] | Phaser | Auto): After a
                phase transition of a shard file, do we keep the old form of the file or garbage
                collect it? Provided as one of: (1) ``None`` for defaults, (2) the single use case
                or phase to keep, (3) a sequence giving the use cases or phases to keep, (4)
                Phaser kwargs (a mapping of use case or phase to whether it must be kept, (5) a
                Phaser object, or ``Auto`` to inherit from StreamingDataset. All code paths result
                in a ``Phaser``. Defaults to ``Auto()``.
            kwargs (Any): Deprecated arguments, if any: ``keep_zip``.
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

        # 3. Maybe derive remote/local index paths.
        if not hasattr(self, 'remote_index_path'):
            if self.remote is None:
                self.remote_index_path = None
            else:
                self.remote_index_path = os.path.join(self.remote, self.split or '',
                                                      self.index_relative_path)
        if not hasattr(self, 'local_index_path'):
            self.local_index_path = os.path.join(self.local, self.split or '',
                                                 self.index_relative_path)

        # 4. Maybe default init args.
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

        # 5. Maybe default download args.
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

        # 6. Maybe default phaser (phase keeper), then delete self.keep_zip.
        keep_zip = kwargs.get('keep_zip')
        self.phaser = _get_phaser(self.init_keep_phases, self.init_keep_zip, keep_phases, keep_zip)
        del self.init_keep_phases
        del self.init_keep_zip

        # 7. Derive `safe_phaser` from phaser.
        self.safe_phaser = deepcopy(self.phaser)
        if self.remote in {None, self.local}:
            self.safe_phaser.persistent = True
