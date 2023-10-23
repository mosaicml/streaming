# Improvements on snapshot_download:
# 1. Enable resume = True. retry when bad network happens
# 2. Disable progress_bar to prevent browser/terminal crash
# 3. Add a monitor to print out file stats every 15 seconds

import os
import time
from huggingface_hub import snapshot_download
from pyspark.sql.functions import concat_ws
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars
import asyncio
import threading

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from streaming.base.util import retry


class FolderObserver:
    def __init__(self, directory):
        patterns = ["*"]
        ignore_patterns = None
        ignore_directories = False
        case_sensitive = True
        self.average_file_size = 0

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory
        self.get_directory_info()

        self.my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)
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
        file_count = 0
        total_file_size = 0
        for root, _, files in os.walk(self.directory):
            for file in files:
                file_count += 1
                file_path = os.path.join(root, file)
                total_file_size += os.path.getsize(file_path)
        self.file_count, self.file_size = file_count, total_file_size

    def on_created(self, event):
        self.file_count += 1

    def on_deleted(self, event):
        self.file_count -= 1

    def on_modified(self, event):
        pass

    def on_moved(self, event):
        pass


def monitor_directory_changes(interval=5):
    def decorator(func):
        def wrapper(repo_id, local_dir, max_workers, token, allow_patterns, *args, **kwargs):
            event = threading.Event()
            start_time = time.time()  # Capture the start time
            observer = FolderObserver(local_dir)

            def beautify(kb):
                mb = kb //(1024)
                gb = mb //(1024)
                return str(mb)+'MB' if mb >= 1 else str(kb) + 'KB'

            def monitor_directory():
                observer.start()
                while not event.is_set():
                    try:
                        elapsed_time = int(time.time() - observer.tik)
                        if observer.file_size > 1e9: # too large to keep an accurate count of the file size
                            if observer.average_file_size == 0:
                                observer.average_file_size = observer.file_size // observer.file_count
                                print(f"approximately: average file size = {beautify(observer.average_file_size//1024)}")
                            kb = observer.average_file_size * observer.file_count // 1024
                        else:
                            observer.get_directory_info()
                            b = observer.file_size
                            kb = b // 1024

                        sz = beautify(kb)
                        cnt = observer.file_count

                        if elapsed_time % 10 == 0 :
                            print(f"Downloaded {cnt} files, Total approx file size = {sz}, Time Elapsed: {elapsed_time} seconds.")

                        if elapsed_time > 0 and elapsed_time % 120 == 0:
                            observer.get_directory_info() # Get the actual stats by walking through the directory
                            observer.average_file_size = observer.file_size // observer.file_count
                            print(f"update average file size to {beautify(observer.average_file_size//1024)}")

                        time.sleep(1)
                    except Exception as exc:
                        # raise RuntimeError("Something bad happened") from exc
                        print(str(exc))
                        time.sleep(1)
                        continue

            monitor_thread = threading.Thread(target=monitor_directory)
            monitor_thread.start()

            try:
                result = func(repo_id, local_dir, max_workers, token, allow_patterns, *args, **kwargs)
                return result
            finally:
                observer.get_directory_info() # Get the actual stats by walking through the directory
                print(f"Done! Downloaded {observer.file_count} files, Total file size = {beautify(observer.file_size//1024)}, Time Elapsed: {int(time.time() - observer.tik)} seconds.")
                observer.stop()
                observer.join()

                event.set()
                monitor_thread.join()

        return wrapper

    return decorator

def retry(max_retries=3, retry_delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"An exception occurred: {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            raise Exception(f"Function {func.__name__} failed after {max_retries} retries.")
        return wrapper
    return decorator

@monitor_directory_changes()
@retry(max_retries=10, retry_delay=10)
def hf_snapshot(repo_id, local_dir, max_workers, token, allow_patterns):
    print(f"Now start to download {repo_id} to {local_dir}, with allow_patterns = {allow_patterns}")
    output = snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir, local_dir_use_symlinks=False, max_workers=max_workers, resume_download=True, token=token, allow_patterns=allow_patterns)
    return output

def download_hf_dataset(local_cache_directory, prefix='', submixes =[], max_workers=32, token="", allow_patterns=None):
    disable_progress_bars()
    for submix in submixes:
        repo_id = os.path.join(prefix, submix)
        local_dir = os.path.join(local_cache_directory, submix)

        output = hf_snapshot(
            repo_id,
            local_dir,
            max_workers,
            token,
            allow_patterns=allow_patterns,
        )

if __name__ == "__main__":
    #download_hf_dataset(local_cache_directory="/tmp/xiaohan/cifar10_1233", prefix="", submixes=["cifar10"], max_workers=32)
    download_hf_dataset(
            local_cache_directory="/tmp/xiaohan/c4_1316",
            prefix = "allenai/",
            submixes = [
                "c4",
            ],
            allow_patterns=["en/*"],
            max_workers=1,
            token = "MY_HUGGINGFACE_ACCESS_TOKEN") # 32 seems to be a sweet point, beyond 32 downloading is not smooth
