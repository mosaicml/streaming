# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Some extras which wrap and/or complement the Streaming storage API."""

import os
import re
from re import Pattern
from time import sleep, time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from streaming.hashing import get_hash
from streaming.storage.download import download_file
from streaming.storage.upload import CloudUploader
from streaming.util.pretty import normalize_bytes, normalize_duration

__all__ = [
    'wait_for_file_to_exist', 'walk_prefix', 'walk_dir', 'list_dataset_files',
    'smart_download_file', 'file_exists'
]


def wait_for_file_to_exist(filename: str, poll_interval: float, timeout: float,
                           err_msg: str) -> None:
    """Wait for the file to exist till timeout seconds. Raise an Exception after that.

    File must be local.

    Args:
        filename (str): A file name
        poll_interval (float): Number of seconds to wait before next polling
        timeout (float): Number of seconds to wait for a file to exist before raising an exception
        err_msg (str): Error message description for an exception

    Raises:
        RuntimeError: Raise an Exception if file does not exist after timeout
    """
    start_time = time()
    while True:
        sleep(poll_interval)
        if os.path.exists(filename):
            sleep(poll_interval)
            break
        dt = time() - start_time
        if dt > timeout:
            raise RuntimeError(f'{err_msg}' + f'{timeout:.3f} < {dt:.3f} secs.')


def _normalize_path(path: str) -> Tuple[str, bool]:
    """Analyze the path, returning URI scheme-normalized form and whether is on the local fs.

    Args:
        path (str): Path to analyze.

    Returns:
        Tuple[str, bool]: Normalized path, and whether it is local.
    """
    obj = urlparse(path)
    if obj.scheme == '':
        is_local = True
    elif obj.scheme == 'file':
        is_local = True
        path = obj.path
    else:
        is_local = False
    return path, is_local


def _normalize_dir(dirname: str) -> str:
    """Normalize a dirname to contain one trailing slash.

    Args:
        dirname (str): Directory path.

    Returns:
        str: Normalized directory path.
    """
    return dirname.rstrip(os.path.sep) + os.path.sep


def walk_prefix(prefix: str) -> List[str]:
    """Recursively list all file paths matching a prefix in sorted order.

    Notes:
      * If you choose a non-directory as a prefix, returned paths will indeed be relative to your
        non-directory, which may seem funky.
      * There is some special case handling so that if your path is a local directory with or
        without a trailing slash, returned paths will nevertheless never start with a slash, lest
        they assume "absolute" power.

    Args:
        prefix (str): Path prefix.

    Returns:
        List[str]: All file paths under the prefix, which are all relative to the given prefix.
    """
    prefix, is_local = _normalize_path(prefix)

    if is_local:
        # Prefix points to local filesystem.
        prefix_rel_files = []
        if os.path.isdir(prefix):
            # Prefix is a directory, so include everything under the directory.
            root = _normalize_dir(prefix)
            for abs_dir, _, file_bases in os.walk(root):
                root_rel_dir = abs_dir.lstrip(root)
                for base in file_bases:
                    root_rel_file = os.path.join(root_rel_dir, base)
                    prefix_rel_files.append(root_rel_file)
        else:
            # Prefix has other stuff tacked onto it after the directory, so include everything
            # under the prefix's parent directory which also matches the prefix's basename.
            root = os.path.dirname(prefix)
            for abs_dir, _, file_bases in os.walk(root):
                for base in file_bases:
                    abs_file = os.path.join(abs_dir, base)
                    if abs_file.startswith(prefix):
                        prefix_rel_file = abs_file.lstrip(prefix)
                        prefix_rel_files.append(prefix_rel_file)
    else:
        # Prefix points to some non-local storage.
        neither = CloudUploader.get(prefix, exist_ok=True)
        prefix_rel_files = neither.list_objects(prefix)

    # TODO: verify all implementations do a global sort on returned paths, then remove this line.
    return sorted(prefix_rel_files)


def walk_dir(root: str) -> List[str]:
    """Recursively list the given directory in sorted order.

    Notes:
      * Supported across various storage backends, including local filesystem.
      * Root must be a directory, not a generic path prefix, to make the local case nicer.
      * There seems to be inconsistency in list_objects() about what the returned paths are
        relative to: cwd, the given root, some local... let's just wrap it for our purposes.

    Args:
        root (str): Root directory to walk.

    Returns:
        List[str]: File paths, which are relative to the given root.
    """
    obj = urlparse(root)
    if obj.scheme == '':
        is_local = True
    elif obj.scheme == 'file':
        is_local = True
        root = obj.path
    else:
        is_local = False

    if is_local:
        if not os.path.isdir(root):
            raise ValueError(f'Path is not a directory: {root}.')
        paths = []
        for sub_root, _, file_basenames in os.walk(root):
            sub_path = sub_root.lstrip(root)
            paths += [os.path.join(sub_path, name) for name in file_basenames]
    else:
        neither = CloudUploader.get(root, exist_ok=True)
        paths = neither.list_objects(root)

    return sorted(paths)


def _filter(keep: Optional[Union[str, Pattern, Callable[[str], bool]]],
            paths: Optional[Iterable[str]]) -> Iterable[str]:
    """Filter the given paths according to the pattern or predicate.

    Args:
        keep (Union[str, Pattern, Callable[[str], bool]], optional): A regex or Callable which is
            applied to each path, keeping or dropping it. If not provided, do no filtering.
        paths (Iterable[str], optional): Iterable of paths to filter. If empty, is the empty list.
    """
    paths = paths or []
    if not keep:
        pass
    elif isinstance(keep, str):
        keep_regex = re.compile(keep)
        paths = filter(keep_regex.match, paths)
    elif isinstance(keep, Pattern):
        paths = filter(keep.match, paths)
    elif isinstance(keep, Callable):
        paths = filter(keep, paths)
    else:
        raise ValueError(f'Unsupported type of keep: {keep}.')
    yield from paths


def _get_overlap(want: Set[str], have: Set[str]) -> Dict[str, Any]:
    """Get the overlap between two sets for informational/debugging purposes.

    Args:
        want (Set[str]): What we want.
        have (Set[str]): What we have.

    Returns:
        Dict[str, Any]: Information about overlaps.
    """
    return {
        'present': len(want & have),
        'missing': len(want.difference(have)),
        'ignored': len(have.difference(want)),
    }


def list_dataset_files(
        *,
        local: str,
        remote: Optional[str] = None,
        split: Optional[str] = None,
        paths: Optional[Iterable[str]] = None,
        keep: Optional[Union[str, Pattern, Callable[[str], bool]]] = None) -> List[str]:
    """Collect all/certain local/remote dataset files, which are then filtered.

    Args:
        local (str): Local dataset root.
        remote (str, optional): Remote dataset root, if we have a remote.
        split (str, optional): Split subdir, if used.
        paths (Iterable[str], optional): Iterable of paths relative to dataset root (i.e.,
            local/remote + split). These are then filtered by the keep predicate, if any. If not
            provided, defaults to a sorted, recursive listing of all dataset files. Such a listing
            treats remote as authoritative if provided, else uses local. Defaults to ``None``.
        keep (Union[str, Pattern, Callable[[str], bool]], optional): A regex or Callable which is
            applied to each path in order to keep or drop it from the listing. If not provided, no
            filtering is performed to paths. Defaults to ``None``.

    Returns:
        List[str]: List of paths, relative to dataset root, ordered by ``paths``.
    """
    # Tack on the split dir, if any.
    if split:
        local = os.path.join(local, split)
        if remote:
            remote = os.path.join(remote, split)

    # If no paths Iterable was not provided, list all the files, filter, and we're done.
    if paths is None:
        root = remote if remote else local
        paths = walk_dir(root)
        return list(_filter(keep, paths))

    # If we were indeed provided explicit paths, cross-check those against a listing of local
    # before we start assuming everything is fine.
    want_paths = list(_filter(keep, paths))
    want_paths_set = set(want_paths)
    have_local_paths_set = set(walk_dir(local))
    if want_paths_set.issubset(have_local_paths_set):  # All exist in local?
        return want_paths

    # If local is incomplete, and there is no remote, give up.
    if not remote:
        obj = _get_overlap(want_paths_set, have_local_paths_set)
        raise ValueError(f'Local does not contain all listed shards, and no remote was ' +
                         f'provided. Overlap of listed vs local: {obj["present"]} present, ' +
                         f'{obj["missing"]} missing, {obj["ignored"]} ignored.')

    # Explicit paths, incomplete local, but we do have a remote to fall back to. Let's cross-check
    # against that.
    have_remote_paths_set = set(walk_dir(remote))
    if want_paths_set.issubset(have_remote_paths_set):
        return want_paths

    # Both local and remote do not contain all the needed files, so give up.
    l_obj = _get_overlap(want_paths_set, have_local_paths_set)
    r_obj = _get_overlap(want_paths_set, have_remote_paths_set)
    raise ValueError(f'Neither local nor remote contains all shards listed. Overlap of listed ' +
                     f'vs local: {l_obj["present"]} present, {l_obj["missing"]} missing, ' +
                     f'{l_obj["ignored"]} ignored. Overlap of listed vs remote: ' +
                     f'{r_obj["present"]} present, {r_obj["missing"]} missing, ' +
                     f'{r_obj["ignored"]} ignored.')


def smart_download_file(*,
                        remote: str,
                        local: str,
                        timeout: Union[float, str] = 60,
                        size: Optional[Union[int, str]] = None,
                        max_size: Optional[Union[int, str]] = None,
                        hashes: Optional[Dict[str, str]] = None) -> None:
    """Download a file from the remote path to the local path, with size/hash checks.

    Args:
        remote (str): Remote path.
        local (str): Local path.
        timeout (Union[float, str]): Maximum time to download, in seconds. Defaults to ``60``.
        size (Union[int, str], optional): Expected file size. This check is a weak but fast/cheap
            way to detect overwrites, truncation, tampering, and corruption. Defaults to ``None``.
        max_size (Union[int, str], optional): Maximum file size. This check is a fast/cheap way to
            prevent the user from inadvertently using shards that are far too large for Streaming
            purposes, which is non-obvious and would result in a terrible user experience. Defaults
            to ``None``.
        hashes (Dict[str, str], optional): Hashes to check, as a dict of hash algo name to expected
            hex digest. These checks are a very strong but slow/expensive way to detect changes to
            data. See our benchmarks for more details. Defaults to ``None``.
    """
    # Download.
    want_timeout = normalize_duration(timeout)
    download_file(remote, local, want_timeout)

    # Size checks.
    if size is not None or max_size is not None:
        have_size = os.stat(local).st_size

        # Exact size check.
        if size is not None:
            want_size = normalize_bytes(size)
            if want_size != have_size:
                raise ValueError(
                    f'The file as downloaded does not match the expected size: remote path = ' +
                    f'{remote}, local path = {local}, expected size = {want_size}, got size = ' +
                    f'{have_size}.')

        # Size limit check.
        if max_size is not None:
            want_max_size = normalize_bytes(max_size)
            if want_max_size < have_size:
                raise ValueError(
                    f'The file is too large for efficient use by Streaming, please reduce shard ' +
                    f'size: remote path = {remote}, local path = {local}, maximum size = ' +
                    f'{want_max_size}, got size = {have_size}.')

    # Hash checks.
    if hashes:
        data = open(local, 'rb').read()
        for hash_algo in sorted(hashes):
            want_hex_digest = hashes[hash_algo]
            have_hex_digest = get_hash(hash_algo, data)
            if want_hex_digest != have_hex_digest:
                raise ValueError(
                    f'The file as downloaded does not match the expected hash: remote path = ' +
                    f'{remote}, local path = {local}, hash algo = {hash_algo}, expected hex ' +
                    f'digest = {want_hex_digest}, got digest = {have_hex_digest}.')


def file_exists(*,
                path: str,
                local: str,
                remote: Optional[str] = None,
                split: Optional[str] = None) -> bool:
    """Determine whether the file path exists across local and/or remote.

    Args:
        path (str): File path relative to local and/or remote.
        local (str): Local root.
        remote (str, optional): Remote root.
        split (str, optional): Dataset split, if applicable.

    Returns:
        bool: Whether file exists locally and/or remotely.
    """
    local_filename = os.path.join(local, split or '', path)
    filenames = walk_prefix(local_filename)
    if filenames and filenames[0] == local_filename:
        return True

    if remote:
        remote_path = os.path.join(remote, split or '', path)
        paths = walk_prefix(remote_path)
        if paths and paths[0] == remote_path:
            return True

    return False
