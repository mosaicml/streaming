# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import patch

import boto3
import pytest
from moto import mock_s3

from tests.common.utils import compressed_local_remote_dir  # pyright: ignore
from tests.common.utils import get_free_tcp_port  # pyright: ignore
from tests.common.utils import local_remote_dir  # pyright: ignore
from tests.test_reader import mds_dataset_dir  # pyright: ignore

MY_BUCKET = 'streaming-test-bucket'
MY_PREFIX = 'train'
GCS_URL = 'https://storage.googleapis.com'
R2_URL = 'https://r2.cloudflarestorage.com'


@pytest.fixture(scope='function')
def bucket_name():
    return MY_BUCKET


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)  # pyright: ignore
def pytest_runtest_call(item: Any):
    # Launch a custom function for distributed tests
    if getattr(item.cls, 'is_dist_test', False):
        dist_test_class = item.cls()
        dist_test_class._run_test(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


@pytest.fixture(scope='session', autouse=True)
def azure_credentials():
    """Mocked azure Credentials."""
    os.environ['AZURE_ACCOUNT_NAME'] = 'testing'
    os.environ['AZURE_ACCOUNT_ACCESS_KEY'] = 'testing'


@pytest.fixture(scope='session', autouse=True)
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'


@pytest.fixture()
def s3_client(aws_credentials: Any):
    with mock_s3():
        conn = boto3.client('s3', region_name='us-east-1')
        yield conn


@pytest.fixture()
def s3_test(s3_client: Any, bucket_name: str):
    s3_client.create_bucket(Bucket=bucket_name)
    yield


@pytest.mark.usefixtures('s3_client', 's3_test')
def test_list_s3_buckets():
    client = boto3.client('s3', region_name='us-east-1')
    buckets = client.list_buckets()
    assert buckets['Buckets'][0]['Name'] == 'streaming-test-bucket'


@pytest.fixture(scope='session', autouse=True)
def gcs_credentials():
    """Mocked GCS Credentials for moto."""
    os.environ['GCS_KEY'] = 'testing'
    os.environ['GCS_SECRET'] = 'testing'


@pytest.fixture()
def gcs_client(gcs_credentials: Any):
    # Have to inline this, as the URL-param is not available as a context decorator
    with patch.dict(os.environ, {'MOTO_S3_CUSTOM_ENDPOINTS': GCS_URL}):
        # Mock needs to be started after the environment variable is patched in
        with mock_s3():
            conn = boto3.client('s3',
                                region_name='us-east-1',
                                endpoint_url=GCS_URL,
                                aws_access_key_id=os.environ['GCS_KEY'],
                                aws_secret_access_key=os.environ['GCS_SECRET'])
            yield conn


@pytest.fixture()
def gcs_test(gcs_client: Any, bucket_name: str):
    gcs_client.create_bucket(Bucket=bucket_name)
    yield


@pytest.mark.usefixtures('gcs_client', 'gcs_test')
def test_list_gcs_buckets():
    client = boto3.client('s3',
                          region_name='us-east-1',
                          endpoint_url=GCS_URL,
                          aws_access_key_id=os.environ['GCS_KEY'],
                          aws_secret_access_key=os.environ['GCS_SECRET'])
    buckets = client.list_buckets()
    assert buckets['Buckets'][0]['Name'] == MY_BUCKET


@pytest.fixture()
def r2_credentials():
    """Mocked R2 Credentials for moto."""
    os.environ['S3_ENDPOINT_URL'] = R2_URL
    yield
    # Line after `yield` gets called at the end of test function.
    del os.environ['S3_ENDPOINT_URL']


@pytest.fixture()
def r2_client(r2_credentials: Any):
    # Have to inline this, as the URL-param is not available as a context decorator
    with patch.dict(os.environ, {'MOTO_S3_CUSTOM_ENDPOINTS': R2_URL}):
        # Mock needs to be started after the environment variable is patched in
        with mock_s3():
            conn = boto3.client('s3', region_name='us-east-1', endpoint_url=R2_URL)
            yield conn


@pytest.fixture()
def r2_test(r2_client: Any, bucket_name: str):
    r2_client.create_bucket(Bucket=bucket_name)
    yield


@pytest.mark.usefixtures('r2_client', 'r2_test')
def test_list_r2_buckets():
    client = boto3.client('s3', region_name='us-east-1', endpoint_url=R2_URL)
    buckets = client.list_buckets()
    assert buckets['Buckets'][0]['Name'] == MY_BUCKET
