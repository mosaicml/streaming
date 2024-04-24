import oci
import urllib
import os

local = "myfile.json"
remote = "oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/0pt8/train/index.json"

OCI_CONFIG = oci.config.from_file()
#no_retry = oci.retry.NoneRetryStrategy()
OCI_CLIENT = oci.object_storage.ObjectStorageClient(
    config=OCI_CONFIG, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
OCI_NAMESPACE = OCI_CLIENT.get_namespace().data

try:
    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 'oci':
        raise ValueError(
            f'Expected obj.scheme to be `oci`, instead, got {obj.scheme} for remote={remote}')

    bucket_name = obj.netloc.split('@' + OCI_NAMESPACE)[0]
    # Remove leading and trailing forward slash from string
    object_path = obj.path.strip('/')
    object_details = OCI_CLIENT.get_object(OCI_NAMESPACE, bucket_name, object_path)
    local_tmp = local + '.tmp'
    with open(local_tmp, 'wb') as f:
        for chunk in object_details.data.raw.stream(2048**2, decode_content=False):
            f.write(chunk)
    os.rename(local_tmp, local)
except Exception as e:
    raise e
finally:
    object_details.data.close()
