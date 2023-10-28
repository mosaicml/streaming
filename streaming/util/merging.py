# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Merging serialized streaming datasets."""

import json
import logging
import os
import shutil
import tempfile
import urllib.parse
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Tuple, Union

from streaming.base.format.index import get_index_basename

__all__ = ['merge_index']

logger = logging.getLogger(__name__)


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


def _merge_index_from_list(index_file_urls: List[Union[str, Tuple[str, str]]],
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

    def not_merged_index(index_file_path: str, out: str):
        """Check if index_file_path is the merged index at folder out.

        Args:
            index_file_path (str): the path to index.json file
            out (str): remote or local url of a folder
        Return:
            (bool): no if index.json sits in out instead of in the subfolders of out
        """
        prefix = str(urllib.parse.urlparse(out).path)
        return os.path.dirname(index_file_path).strip('/') != prefix.strip('/')

    if not out:
        logger.warning('No MDS dataset folder specified, no index merged')
        return

    cu = CloudUploader.get(out, exist_ok=True, keep_local=True)

    local_index_files = []
    cl = CloudUploader.get(cu.local, exist_ok=True, keep_local=True)
    for file in cl.list_objects():
        if file.endswith('.json') and not_merged_index(file, cu.local):
            local_index_files.append(file)

    if cu.remote:
        obj = urllib.parse.urlparse(cu.remote)
        remote_index_files = []
        for file in cu.list_objects():
            if file.endswith(get_index_basename()) and not_merged_index(file, cu.remote):
                join_char = '//'
                if obj.scheme == 'dbfs':
                    path = Path(cu.remote)
                    prefix = os.path.join(path.parts[0], path.parts[1])
                    if prefix == 'dbfs:/Volumes':
                        join_char = '/'
                remote_index_files.append(obj.scheme + join_char + os.path.join(obj.netloc, file))
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
