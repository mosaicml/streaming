# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Base module for downloading/uploading files from/to cloud storage."""

from streaming.storage.download import (download_file, download_from_azure,
                                        download_from_azure_datalake,
                                        download_from_databricks_unity_catalog, download_from_dbfs,
                                        download_from_gcs, download_from_local, download_from_oci,
                                        download_from_s3, download_from_sftp)
from streaming.storage.extra import (file_exists, list_dataset_files, smart_download_file,
                                     wait_for_file_to_exist, walk_dir, walk_prefix)
from streaming.storage.upload import (AzureDataLakeUploader, AzureUploader, CloudUploader,
                                      GCSUploader, LocalUploader, OCIUploader, S3Uploader)

__all__ = [
    'download_file',
    'CloudUploader',
    'S3Uploader',
    'GCSUploader',
    'OCIUploader',
    'LocalUploader',
    'AzureUploader',
    'AzureDataLakeUploader',
    'download_from_s3',
    'download_from_sftp',
    'download_from_gcs',
    'download_from_oci',
    'download_from_azure',
    'download_from_azure_datalake',
    'download_from_databricks_unity_catalog',
    'download_from_dbfs',
    'download_from_local',
    'wait_for_file_to_exist',
    'walk_prefix',
    'walk_dir',
    'list_dataset_files',
    'smart_download_file',
    'file_exists',
]
