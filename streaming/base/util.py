# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for datasets."""

import collections.abc
import functools
import json
import logging
import os
import random
import shutil
import tempfile
import urllib.parse
from collections import OrderedDict
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from pathlib import Path
from time import sleep, time
from typing import Any, Callable, List, Sequence, Tuple, Type, TypeVar, Union, cast, overload

import torch.distributed as dist

from streaming.base.constant import SHM_TO_CLEAN
from streaming.base.distributed import get_local_rank, maybe_init_dist
from streaming.base.format.index import get_index_basename
from streaming.base.shared.prefix import _get_path

logger = logging.getLogger(__name__)

TCallable = TypeVar('TCallable', bound=Callable)

__all__ = [
    'get_list_arg', 'wait_for_file_to_exist', 'bytes_to_int', 'number_abbrev_to_int',
    'clean_stale_shared_memory', 'get_import_exception_message', 'merge_index', 'retry'
]


def get_list_arg(text: str) -> List[str]:
    """Pass a list as a command-line flag.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: Splits, if any.
    """
    return text.split(',') if text else []


def wait_for_file_to_exist(filename: str, poll_interval: float, timeout: float,
                           err_msg: str) -> None:
    """Wait for the file to exist till timeout seconds. Raise an Exception after that.

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


def bytes_to_int(bytes_str: Union[int, str]) -> int:
    """Convert human readable byte format to an integer.

    Args:
        bytes_str (Union[int, str]): Value to convert.

    Raises:
        ValueError: Invalid byte suffix.

    Returns:
        int: Integer value of bytes.
    """
    #input is already an int
    if isinstance(bytes_str, int) or isinstance(bytes_str, float):
        return int(bytes_str)

    units = {
        'kb': 1024,
        'mb': 1024**2,
        'gb': 1024**3,
        'tb': 1024**4,
        'pb': 1024**5,
        'eb': 1024**6,
        'zb': 1024**7,
        'yb': 1024**8,
    }
    # Convert a various byte types to an integer
    for suffix in units:
        bytes_str = bytes_str.lower().strip()
        if bytes_str.lower().endswith(suffix):
            try:
                return int(float(bytes_str[0:-len(suffix)]) * units[suffix])
            except ValueError:
                raise ValueError(''.join([
                    f'Unsupported value/suffix {bytes_str}. Supported suffix are ',
                    f'{["b"] + list(units.keys())}.'
                ]))
    else:
        # Convert bytes to an integer
        if bytes_str.endswith('b') and bytes_str[0:-1].isdigit():
            return int(bytes_str[0:-1])
        # Convert string representation of a number to an integer
        elif bytes_str.isdigit():
            return int(bytes_str)
        else:
            raise ValueError(''.join([
                f'Unsupported value/suffix {bytes_str}. Supported suffix are ',
                f'{["b"] + list(units.keys())}.'
            ]))


def number_abbrev_to_int(abbrev_str: Union[int, str]) -> int:
    """Convert human readable number abbreviations to an integer.

    Args:
        abbrev_str (Union[int, str]): Value to convert.

    Raises:
        ValueError: Invalid number suffix.

    Returns:
        int: Integer value of number abbreviation.
    """
    #input is already an int
    if isinstance(abbrev_str, int) or isinstance(abbrev_str, float):
        return int(abbrev_str)

    units = {
        'k': 10**3,
        'm': 10**6,
        'b': 10**9,
        't': 10**12,
    }
    # Convert a various abbreviation types to an integer
    for suffix in units:
        abbrev_str = abbrev_str.lower().strip()
        if abbrev_str.lower().endswith(suffix):
            try:
                return int(float(abbrev_str[0:-len(suffix)]) * units[suffix])
            except ValueError:
                raise ValueError(''.join([
                    f'Unsupported value/suffix {abbrev_str}. Supported suffix are ',
                    f'{list(units.keys())}.'
                ]))
    else:
        # Convert string representation of a number to an integer
        if abbrev_str.isdigit():
            return int(abbrev_str)
        else:
            raise ValueError(''.join([
                f'Unsupported value/suffix {abbrev_str}. Supported suffix are ',
                f'{list(units.keys())}.'
            ]))


def clean_stale_shared_memory() -> None:
    """Clean up all the leaked shared memory.

    In case of a distributed run, clean up happens on local rank 0 while other local ranks wait for
    the local rank 0 to finish.
    """
    # Initialize torch.distributed ourselves, if necessary.
    destroy_dist = maybe_init_dist()

    # Perform clean up on local rank 0
    if get_local_rank() == 0:
        for prefix_int in range(1000000):
            leaked_shm = False
            for shm_name in SHM_TO_CLEAN:
                name = _get_path(prefix_int, shm_name)
                try:
                    shm = BuiltinSharedMemory(name, True, 4)
                except FileExistsError:
                    shm = BuiltinSharedMemory(name, False, 4)
                    leaked_shm = True
                finally:
                    shm.close()  # pyright: ignore
                    shm.unlink()
            # Come out of loop if no leaked shared memory
            if not leaked_shm:
                break

    # Sync all ranks
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Delete the process group if Streaming initialized it.
    if destroy_dist:
        dist.destroy_process_group()


def get_import_exception_message(package_name: str, extra_deps: str) -> str:
    """Get import exception message.

    Args:
        package_name (str): Package name.

    Returns:
        str: Exception message.
    """
    return f'Streaming was installed without {package_name} support. ' + \
            f'To use {package_name} related packages with Streaming, run ' + \
            f'`pip install \'mosaicml-streaming[{package_name}]\'`.'


def merge_index(*args: Any, **kwargs: Any):
    r"""Merge index.json from partitions to form a global index.json.

    This can be called as

        merge_index(index_file_urls, out, keep_local, download_timeout)

        merge_index(out, keep_local, download_timeout)

    The first signature takes in a list of index files URLs of MDS partitions.
    The second takes the root of a MDS dataset and parse the partition folders from there.

    Args:
        index_file_urls (List[Union[str, Tuple[str,str]]]): index.json from all the partitions.
            Each element can take the form of a single path string or a tuple string.

            1. If ``index_file_urls`` is a List of local URLs, merge locally without download.
            2. If ``index_file_urls`` is a List of tuple (local, remote) URLs, check if local index.json are missing, download before merging.
            3. If ``index_file_urls`` is a List of remote URLs, download all and merge.

        out (Union[str, Tuple[str,str]]): folder that contain MDS partitions and to put the merged index file

            1. A local directory, merge index happens locally.
            2. A remote directory, download all the sub-directories index.json, merge locally and upload.
            3. A tuple (local_dir, remote_dir), check if local index.json exist, download if not.

        keep_local (bool): Keep local copy of the merged index file. Defaults to ``True``.
        download_timeout (int): The allowed time for downloading each json file. Defaults to 60.
    """
    if isinstance(args[0], list) and len(args) + len(kwargs) in [2, 3, 4]:
        return _merge_index_from_list(*args, **kwargs)
    elif (isinstance(args[0], str) or
          isinstance(args[0], tuple)) and len(args) + len(kwargs) in [1, 2, 3]:
        return _merge_index_from_root(*args, **kwargs)
    raise ValueError(f'Invalid arguments to merge_index: {args}, {kwargs}')


def _merge_index_from_list(index_file_urls: Sequence[Union[str, Tuple[str, str]]],
                           out: Union[str, Tuple[str, str]],
                           keep_local: bool = True,
                           download_timeout: int = 60) -> None:
    """Merge index.json from a list of index files of MDS directories to create joined index.

    Args:
        index_file_urls (Union[str, Tuple[str,str]]): index.json from all the partitions
            each element can take the form of a single path string or a tuple string.

            The pattern of index_file_urls and corresponding reaction is one of:
            1. All URLS are str (local). All URLS are accessible locally -> no download
            2. All URLS are tuple (local, remote). All URLS are accessible locally -> no download
            3. All URLS are tuple (local, remote). Download URL that is not accessible locally
            4. All URLS are str (remote) -> download all

        out (Union[str, Tuple[str, str]]): path to put the merged index file
        keep_local (bool): Keep local copy of the merged index file. Defaults to ``True``
        download_timeout (int): The allowed time for downloading each json file. Defaults to 60.
    """
    from streaming.base.storage.download import download_file
    from streaming.base.storage.upload import CloudUploader

    if not index_file_urls or not out:
        logger.warning('Either index_file_urls or out are None. ' +
                       'Need to specify both `index_file_urls` and `out`. ' + 'No index merged')
        return

    # This is the index json file name, e.g., it is index.json as of 0.6.0
    index_basename = get_index_basename()

    cu = CloudUploader.get(out, keep_local=True, exist_ok=True)

    # Remove duplicates, and strip '/' from right if any
    index_file_urls = list(OrderedDict.fromkeys(index_file_urls))
    urls = []
    for url in index_file_urls:
        if isinstance(url, str):
            urls.append(url.rstrip('/').strip())
        else:
            urls.append((url[0].rstrip('/').strip(), url[1].rstrip('/').strip()))

    # Prepare a temp folder to download index.json from remote if necessary. Removed in the end.
    with tempfile.TemporaryDirectory() as temp_root:
        logging.warning(f'A temporary folder {temp_root} is created to store index files')

        # Copy files to a temporary directory. Download if necessary
        partitions = []
        for url in urls:
            if isinstance(url, tuple):
                src = url[0] if os.path.exists(url[0]) else url[1]
            else:
                src = url

            obj = urllib.parse.urlparse(src)
            scheme, bucket, path = obj.scheme, obj.netloc, obj.path
            if scheme == '' and bucket == '' and path == '':
                raise FileNotFoundError(
                    f'Check data availability! local index {url[0]} is not accessible.' +
                    f'remote index {url[1]} does not have a valid url format')
            dest = os.path.join(temp_root, path.lstrip('/'))

            try:
                download_file(src, dest, download_timeout)
            except Exception as ex:
                raise RuntimeError(f'Failed to download index.json: {src} to {dest}') from ex

            if not os.path.exists(dest):
                raise FileNotFoundError(f'Index file {dest} does not exist or not accessible.')

            partitions.append(dest)

        # merge shards from all index files
        shards = []
        for partition_index in partitions:
            p = Path(partition_index)
            obj = json.load(open(partition_index))
            for i in range(len(obj['shards'])):
                shard = obj['shards'][i]
                for key in ('raw_data', 'zip_data', 'raw_meta', 'zip_meta'):
                    if shard.get(key):
                        basename = shard[key]['basename']
                        obj['shards'][i][key]['basename'] = os.path.join(
                            os.path.basename(p.parent), basename)
            shards += obj['shards']

        # Save merged index locally
        obj = {
            'version': 2,
            'shards': shards,
        }
        merged_index_path = os.path.join(temp_root, index_basename)
        with open(merged_index_path, 'w') as outfile:
            json.dump(obj, outfile)

        # Move merged index from temp path to local part in out
        # Upload merged index to remote if out has remote part
        shutil.move(merged_index_path, cu.local)
        if cu.remote is not None:
            cu.upload_file(index_basename)

        # Clean up
        if not keep_local:
            shutil.rmtree(cu.local, ignore_errors=True)


def _not_merged_index(index_file_path: str, out: str):
    """Check if index_file_path is the merged index at folder out.

    Args:
        index_file_path (str): the path to index.json file
        out (str): remote or local url of a folder
    Return:
        (bool): no if index.json sits in out instead of in the subfolders of out
    """
    prefix = str(urllib.parse.urlparse(out).path)
    return os.path.dirname(index_file_path).strip('/') != prefix.strip('/')


def _format_remote_index_files(remote: str, files: List[str]) -> List[str]:
    """Formats the remote index files by appending the remote URL scheme and netloc to each file.

    Args:
        remote (str): The remote URL.
        files (list[str]): The list of files.

    Returns:
        list[str]: The formatted remote index files.
    """
    remote_index_files = []
    obj = urllib.parse.urlparse(remote)
    for file in files:
        if file.endswith(get_index_basename()) and _not_merged_index(file, remote):
            join_char = '://'
            if obj.scheme == 'dbfs':
                path = Path(remote)
                prefix = os.path.join(path.parts[0], path.parts[1])
                if prefix == 'dbfs:/Volumes':
                    join_char = ':/'

            remote_index_files.append(obj.scheme + join_char + os.path.join(obj.netloc, file))
    return remote_index_files


def _merge_index_from_root(out: Union[str, Tuple[str, str]],
                           keep_local: bool = True,
                           download_timeout: int = 60) -> None:
    """Merge index.json given the root of MDS dataset. Write merged index to the root folder.

    Args:
        out (Union[str, Tuple[str,str]]): folder that contain MDS partitions.
            :A local directory, merge index happens locally
            :A remote directory, download all the sub-directories index.json in a temporary
                sub-directories, merge locally, and then upload it to out location
            :A (local_dir, remote_dir), check if sub-directories index.json file present locally
                If yes, then merge locally and upload to remote_dir .
                If not, download all the sub-directories index.json from remote to local,
                merge locally, and upload to remote_dir .
        keep_local (bool): Keep local copy of the merged index file. Defaults to ``True``
        download_timeout (int): The allowed time for downloading each json file. Defaults to 60.
    """
    from streaming.base.storage.upload import CloudUploader

    if not out:
        logger.warning('No MDS dataset folder specified, no index merged')
        return

    cu = CloudUploader.get(out, exist_ok=True, keep_local=True)

    local_index_files = []
    cl = CloudUploader.get(cu.local, exist_ok=True, keep_local=True)
    for file in cl.list_objects():
        if file.endswith('.json') and _not_merged_index(file, cu.local):
            local_index_files.append(file)

    if cu.remote:
        remote_index_files = _format_remote_index_files(cu.remote, cu.list_objects())
        if len(local_index_files) == len(remote_index_files):
            _merge_index_from_list(list(zip(local_index_files, remote_index_files)),
                                   out,
                                   keep_local=keep_local,
                                   download_timeout=download_timeout)
        else:
            _merge_index_from_list(remote_index_files,
                                   out,
                                   keep_local=keep_local,
                                   download_timeout=download_timeout)
        return

    _merge_index_from_list(local_index_files,
                           out,
                           keep_local=keep_local,
                           download_timeout=download_timeout)


@overload
def retry(
    exc_class: Union[Type[Exception], Sequence[Type[Exception]]] = ...,
    num_attempts: int = ...,
    initial_backoff: float = ...,
    max_jitter: float = ...,
) -> Callable[[TCallable], TCallable]:
    ...


@overload
def retry(exc_class: TCallable) -> TCallable:
    # Use the decorator without parenthesis
    ...


# error: Type "(TCallable@retry) -> TCallable@retry" cannot be assigned to type
# "(func: Never) -> Never"
def retry(  # type: ignore
    exc_class: Union[TCallable, Type[Exception], Sequence[Type[Exception]]] = Exception,
    num_attempts: int = 3,
    initial_backoff: float = 1.0,
    max_jitter: float = 0.5,
):
    """Decorator to retry a function with backoff and jitter.

    Attempts are spaced out with
    ``initial_backoff * 2**num_attempts + random.random() * max_jitter`` seconds.

    Example:
        .. testcode::

            from streaming.base.util import retry

            num_tries = 0

            @retry(RuntimeError, num_attempts=3, initial_backoff=0.1)
            def flaky_function():
                global num_tries
                if num_tries < 2:
                    num_tries += 1
                    raise RuntimeError("Called too soon!")
                return "Third time's a charm."

            print(flaky_function())

    .. testoutput::

        Third time's a charm.

    Args:
        exc_class (Type[Exception] | Sequence[Type[Exception]]], optional): The exception class or
            classes to retry. Defaults to Exception.
        num_attempts (int, optional): The total number of attempts to make. Defaults to 3.
        initial_backoff (float, optional): The initial backoff, in seconds. Defaults to 1.0.
        max_jitter (float, optional): The maximum amount of random jitter to add. Defaults to 0.5.

            Increasing the ``max_jitter`` can help prevent overloading a resource when multiple
            processes in parallel are calling the same underlying function.
    """
    if num_attempts < 1:
        raise ValueError('num_attempts must be at-least 1')

    def wrapped_func(func: TCallable) -> TCallable:

        @functools.wraps(func)
        def new_func(*args: Any, **kwargs: Any):
            i = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exc_class as e:
                    if i + 1 == num_attempts:
                        logger.debug(f'Attempt {i + 1}/{num_attempts} failed with: {e}')
                        raise e
                    else:
                        sleep(initial_backoff * 2**i + random.random() * max_jitter)
                        logger.debug(f'Attempt {i + 1}/{num_attempts} failed with: {e}')
                        i += 1

        return cast(TCallable, new_func)

    if not isinstance(exc_class, collections.abc.Sequence) and not (isinstance(
            exc_class, type) and issubclass(exc_class, Exception)):
        # Using the decorator without (), like @retry_with_backoff
        func = cast(TCallable, exc_class)
        exc_class = Exception

        return wrapped_func(func)

    return wrapped_func
