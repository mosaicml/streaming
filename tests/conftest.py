# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

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


@pytest.fixture(scope='class', autouse=True)
def azure_credentials():
    """Mocked azure Credentials."""
    os.environ['AZURE_ACCOUNT_NAME'] = 'testing'
    os.environ['AZURE_ACCOUNT_ACCESS_KEY'] = 'testing'


@pytest.fixture(scope='class', autouse=True)
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'


@pytest.fixture(scope='class', autouse=True)
def hf_credentials():
    """Mocked HF Credentials."""
    os.environ['HF_TOKEN'] = 'testing'


@pytest.fixture()
def s3_client(aws_credentials: Any):
    with mock_aws():
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
def clear_environ_credentials():
    """Clears all cloud provider credentials for testing."""
    os.environ.pop('GCS_KEY', None)
    os.environ.pop('GCS_SECRET', None)
    os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
    os.environ.pop('AWS_ACCESS_KEY_ID', None)
    os.environ.pop('AWS_SECRET_ACCESS_KEY', None)
    os.environ.pop('AWS_SECURITY_TOKEN', None)
    os.environ.pop('AWS_SESSION_TOKEN', None)
    os.environ.pop('AZURE_ACCOUNT_NAME', None)
    os.environ.pop('AZURE_ACCOUNT_ACCESS_KEY', None)


@pytest.fixture()
def gcs_hmac_credentials():
    """Mocked GCS Credentials for moto."""
    os.environ['GCS_KEY'] = 'hmac_key_testing'
    os.environ['GCS_SECRET'] = 'hmac_secret_testing'
    yield
    del os.environ['GCS_KEY']
    del os.environ['GCS_SECRET']


@pytest.fixture()
def gcs_service_account_credentials():
    """Mocked GCS Credentials for service level account."""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account_testing'
    yield
    del os.environ['GOOGLE_APPLICATION_CREDENTIALS']


@pytest.fixture()
def gcs_hmac_client(gcs_hmac_credentials: Any):
    # Have to inline this, as the URL-param is not available as a context decorator
    with patch.dict(os.environ, {'MOTO_S3_CUSTOM_ENDPOINTS': GCS_URL}):
        # Mock needs to be started after the environment variable is patched in
        with mock_aws():
            conn = boto3.client(
                's3',
                region_name='us-east-1',
                endpoint_url=GCS_URL,
                aws_access_key_id=os.environ['GCS_KEY'],
                aws_secret_access_key=os.environ['GCS_SECRET'],
            )
            yield conn


@pytest.fixture()
def gcs_test(gcs_hmac_client: Any, bucket_name: str):
    gcs_hmac_client.create_bucket(Bucket=bucket_name)
    yield


@pytest.mark.usefixtures('gcs_hmac_client', 'gcs_test')
def test_list_gcs_buckets():
    client = boto3.client(
        's3',
        region_name='us-east-1',
        endpoint_url=GCS_URL,
        aws_access_key_id=os.environ['GCS_KEY'],
        aws_secret_access_key=os.environ['GCS_SECRET'],
    )
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
        with mock_aws():
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


@pytest.fixture()
def alipan_credentials():
    """Mocked alipan Credentials."""
    os.environ['ALIPAN_WEB_REFRESH_TOKEN'] = 'testing'
    yield
    del os.environ['ALIPAN_WEB_REFRESH_TOKEN']
