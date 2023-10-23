# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for HF datasets Ingestion."""

# Improvements on snapshot_download:
# 1. Enable resume = True. retry when bad network happens
# 2. Disable progress_bar to prevent browser/terminal crash
# 3. Add a monitor to print out file stats every 15 seconds

import logging
import os
import threading
import time
from typing import Any, List, Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from streaming.base.util import retry

logger = logging.getLogger(__name__)


class FolderObserver:
    """A wrapper class of WatchDog."""

    def __init__(self, directory: str):
        """Specify the download directory to monitor."""
        patterns = ['*']
        ignore_patterns = None
        ignore_directories = False
        case_sensitive = True
        self.average_file_size = 0

        self.file_count = 0
        self.file_size = 0

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory
        self.get_directory_info()

        self.my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns,
                                                            ignore_directories, case_sensitive)
        self.my_event_handler.on_created = self.on_created
        self.my_event_handler.on_deleted = self.on_deleted
        self.my_event_handler.on_modified = self.on_modified
        self.my_event_handler.on_moved = self.on_moved

        go_recursively = True
        self.observer = Observer()
        self.observer.schedule(self.my_event_handler, directory, recursive=go_recursively)
        self.tik = time.time()

    def start(self):
        return self.observer.start()

    def stop(self):
        return self.observer.stop()

    def join(self):
        return self.observer.join()

    def get_directory_info(self):
        self.file_count = file_count = 0
        self.file_size = total_file_size = 0
        for root, _, files in os.walk(self.directory):
            for file in files:
                file_count += 1
                file_path = os.path.join(root, file)
                total_file_size += os.path.getsize(file_path)
        self.file_count, self.file_size = file_count, total_file_size

    def on_created(self, event: Any):
        self.file_count += 1

    def on_deleted(self, event: Any):
        print(type(event))
        self.file_count -= 1

    def on_modified(self, event: Any):
        print(type(event))
        pass

    def on_moved(self, event: Any):
        print(type(event))
        pass


def monitor_directory_changes(interval: int = 5):
    """Dataset downloading monitor. Keep file counts N and file size accumulation.

    Approximate dataset size by N * avg file size.
    """

    def decorator(func: Any):

        def wrapper(repo_id: str, local_dir: str, max_workers: int, token: str,
                    allow_patterns: Optional[List[str]], *args: Any, **kwargs: Any):
            event = threading.Event()
            observer = FolderObserver(local_dir)

            def beautify(kb: int):
                mb = kb // (1024)
                gb = mb // (1024)
                if gb >= 1:
                    return str(gb) + 'GB'
                elif mb >= 1:
                    return str(mb) + 'MB'
                else:
                    return str(kb) + 'KB'

            def monitor_directory():
                observer.start()
                while not event.is_set():
                    try:
                        elapsed_time = int(time.time() - observer.tik)
                        if observer.file_size > 1e9:  # too large to keep an accurate count of the file size
                            if observer.average_file_size == 0:
                                observer.average_file_size = observer.file_size // observer.file_count
                                logger.warning(
                                    f'approximately: average file size = {beautify(observer.average_file_size//1024)}'
                                )
                            kb = observer.average_file_size * observer.file_count // 1024
                        else:
                            observer.get_directory_info()
                            b = observer.file_size
                            kb = b // 1024

                        sz = beautify(kb)
                        cnt = observer.file_count

                        if elapsed_time % 10 == 0:
                            logger.warning(
                                f'Downloaded {cnt} files, Total approx file size = {sz}, Time Elapsed: {elapsed_time} seconds.'
                            )

                        if elapsed_time > 0 and elapsed_time % 120 == 0:
                            observer.get_directory_info(
                            )  # Get the actual stats by walking through the directory
                            observer.average_file_size = observer.file_size // observer.file_count
                            logger.warning(
                                f'update average file size to {beautify(observer.average_file_size//1024)}'
                            )

                        time.sleep(1)
                    except Exception as exc:
                        # raise RuntimeError("Something bad happened") from exc
                        logger.warning(str(exc))
                        time.sleep(1)
                        continue

            monitor_thread = threading.Thread(target=monitor_directory)
            monitor_thread.start()

            try:
                result = func(repo_id, local_dir, max_workers, token, allow_patterns, *args,
                              **kwargs)
                return result
            finally:
                observer.get_directory_info(
                )  # Get the actual stats by walking through the directory
                logger.warning(
                    f'Done! Downloaded {observer.file_count} files, Total file size = {beautify(observer.file_size//1024)}, Time Elapsed: {int(time.time() - observer.tik)} seconds.'
                )
                observer.stop()
                observer.join()

                event.set()
                monitor_thread.join()

        return wrapper

    return decorator


@monitor_directory_changes()
@retry([Exception, RuntimeError], num_attempts=10, initial_backoff=10)
def hf_snapshot(repo_id: str, local_dir: str, max_workers: int, token: str,
                allow_patterns: Optional[List[str]]):
    """API call to HF snapshot_download.

    which internally use hf_hub_download
    """
    print(
        f'Now start to download {repo_id} to {local_dir}, with allow_patterns = {allow_patterns}')
    output = snapshot_download(repo_id,
                               repo_type='dataset',
                               local_dir=local_dir,
                               local_dir_use_symlinks=False,
                               max_workers=max_workers,
                               resume_download=True,
                               token=token,
                               allow_patterns=allow_patterns)
    return output


def download_hf_dataset(local_cache_directory: str,
                        prefix: str,
                        submixes: List[str],
                        token: str,
                        max_workers: int = 32,
                        allow_patterns: Optional[List[str]] = None) -> None:
    """Disable progress bar and call hf_snapshot.

    Args:
        local_cache_directory (str): local output directory the dataset will be written to.
        prefix (str): HF namespace, allenai for example.
        submixes (List): a list of repos within HF namespace, c4 for example.
        token (str): HF access toekn.
        max_workers (int): number of processors to parallelize downloading.
        allow_patterns (List): only files matching the pattern will be download. E.g., "en/*" along with allenai/c4 means to download allenai/c4/en folder only.
    """
    disable_progress_bars()
    for submix in submixes:
        repo_id = os.path.join(prefix, submix)
        local_dir = os.path.join(local_cache_directory, submix)

        _ = hf_snapshot(
            repo_id,
            local_dir,
            max_workers,
            token,
            allow_patterns=allow_patterns,
        )


if __name__ == '__main__':
    #download_hf_dataset(local_cache_directory="/tmp/xiaohan/cifar10_1233", prefix="", submixes=["cifar10"], max_workers=32)
    download_hf_dataset(local_cache_directory='/tmp/xiaohan/c4_1316',
                        prefix='allenai/',
                        submixes=[
                            'c4',
                        ],
                        allow_patterns=['en/*'],
                        max_workers=1,
                        token='MY_HUGGINGFACE_ACCESS_TOKEN'
                       )  # 32 seems to be a sweet point, beyond 32 downloading is not smooth
