import tempfile
from google.cloud.storage import Client
from streaming.base.storage.upload import GCSUploader


def run_official_fn():
    bucket_name = "mosaicml-composer-tests"
    storage_client = Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix='streaming')
    # Note: The call returns a response only when the iterator is consumed.
    print('-----blobs in gcs-----')
    for blob in blobs:
        print(blob.name)

def main():
    with tempfile.TemporaryDirectory() as file_path:
        filename = 'random_file.txt'
        print(file_path, filename)
        remote = 'gs://mosaicml-composer-tests/streaming'
        gcsw = GCSUploader(out=(file_path, remote), keep_local=True)
        # gcsw.check_bucket_exists(remote)
        with open(file_path+'/'+filename, 'wb') as tmp:
            tmp.write(b'Hello world!')
        gcsw.upload_file(filename)
        run_official_fn()

if __name__ == "__main__":
    main()