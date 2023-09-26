# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for datasets."""

import json
import logging
import os
import shutil
import tempfile
import urllib.parse
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from time import sleep, time
from typing import List, Tuple, Union

import torch.distributed as dist

from streaming.base.constant import SHM_TO_CLEAN
from streaming.base.distributed import get_local_rank, maybe_init_dist
from streaming.base.format.index import get_index_basename
from streaming.base.shared.prefix import _get_path

logger = logging.getLogger(__name__)

__all__ = ['get_list_arg']


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


def merge_index(folder_urls: List[Union[str, Tuple[str, str]]],
                out: Union[str, Tuple[str, str]],
                *,
                keep_local: bool = True,
                overwrite: bool = True,
                download_timeout: int = 100) -> int:
    """Merge index.json from a list of remote or local directories.

    Args:
        folder_urls (Iterable): folders that contain index.json for the partition
            each element can take the form of a single path string or a tuple string

            for each  url in folder_urls, if url is
                1. tuple (local, remote): check if local is accessible.
                    -> Yes: use local index to merge
                    -> No:  download from remote first, then merge
                2. str (local path): use local path to merge.
                    raise FileNotFoundError if any local index is not accessible
                3. str (remote url): download to a temp directory first, then merge

        out (Union[str, Tuple[str, str]]): path to put the merged index file
        keep_local (bool): Keep local copy of the merged index file. Defaults to ``True``
        overwrite (bool): Overwrite merged index file in out if there exists one.Defaults to ``True``
        download_timeout (int): The allowed time for downloading each json file
            defaults to 60, same as streaming.download_file

    Returns:
        int: count of files downloaded during function call
    """
    # Import here to avoid circular import error
    from streaming.base.storage.download import download_file
    from streaming.base.storage.upload import CloudUploader

    if not folder_urls:
        logger.warning('No partitions exist, no index merged')
        return 0

    # This is the index json file name, e.g., it is index.json as of 0.6.0
    index_basename = get_index_basename()

    cu = CloudUploader.get(out, keep_local=True, exist_ok=True)
    if os.path.exists(os.path.join(cu.local, index_basename)) and overwrite:
        logger.warning('Merged index already exists locally. no index merged if overwrite=False')
        return 0

    # Remove '/' from right, so os.path.basename gives relative path to each folder
    urls = []
    for url in folder_urls:
        if type(url) is str:
            urls.append(url.rstrip('/').strip())
        else:
            urls.append((url[0].rstrip('/').strip(), url[1].rstrip('/').strip()))

    # Determine if we need to call download_file.
    download = False
    for url in urls:
        local = remote = url
        if type(url) is tuple:
            # If driver cannot access the local path, download = True
            download = not os.path.exists(url[0])
        else:
            # If url is a remote, download = True, False otherwise
            download = urllib.parse.urlparse(url).scheme != ''

        # As long as one index file needs download, we download them all to keep it simple
        if download:
            break

    print('download = ', download)
    # Prepare a temp folder to download index.json rom remote if necessary. Removed in the end.
    with tempfile.TemporaryDirectory() as temp_root:

        # container for absolute local folder path
        partitions = []
        n_downloads = 0
        for url in urls:
            local = remote = url

            if download:
                # If download is needed, download url from remote to temp_root
                path = urllib.parse.urlparse(remote).path
                local = os.path.join(temp_root, path.lstrip('/'))
                try:
                    remote_url = os.path.join(remote, index_basename)
                    local_path = os.path.join(local, index_basename)
                    download_file(remote_url, local_path, download_timeout)
                    n_downloads += 1
                except Exception as ex:
                    raise RuntimeError(f'failed to download index.json {url}') from ex

            if not (os.path.exists(local)):
                raise FileNotFoundError(
                    'Folder {local} does not exit or cannot be acceessed by the current process')
            partitions.append(local)

        # merge index files into shards
        shards = []
        for partition in partitions:
            partition_index = f'{partition}/{index_basename}'
            mds_partition_basename = os.path.basename(partition)
            obj = json.load(open(partition_index))
            for i in range(len(obj['shards'])):
                shard = obj['shards'][i]
                for key in ('raw_data', 'zip_data'):
                    if shard.get(key):
                        basename = shard[key]['basename']
                        obj['shards'][i][key]['basename'] = os.path.join(
                            mds_partition_basename, basename)
            shards += obj['shards']

        # Save merged index locally
        obj = {
            'version': 2,
            'shards': shards,
        }
        merged_index_path = os.path.join(temp_root, index_basename)
        with open(merged_index_path, 'w') as outfile:
            json.dump(obj, outfile)

        # Upload merged index to remote if out has remote part
        # Otherwise, move it from temp root to out location
        shutil.move(merged_index_path, cu.local)
        if cu.remote is not None:
            cu.upload_file(index_basename)

        # Clean up
        # shutil.rmtree(temp_root, ignore_errors=True)
        if not keep_local:
            shutil.rmtree(cu.local, ignore_errors=True)

    return n_downloads
