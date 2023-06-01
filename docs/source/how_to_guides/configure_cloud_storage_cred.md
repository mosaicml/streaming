# Configure Cloud Storage Credentials

Streaming dataset supports the following cloud storage providers to stream your data directly to your instance.
- Amazon S3
- Any S3 compatible object store
- Google Cloud Storage
- Oracle Cloud Storage
- Azure Blob Storage

## Amazon S3

For an S3 bucket with public access, no additional setup is required, simply specify the S3 URI of the resource.

### MosaicML platform

For [MosaicML platform](https://www.mosaicml.com/cloud) users, follow the steps mentioned in the [AWS S3](https://mcli.docs.mosaicml.com/en/latest/secrets/s3.html) MCLI doc on how to configure the cloud provider credentials.

### Others

First, make sure the `awscli` is installed, and then run `aws configure` to create the config and credential files:

```
python -m pip install awscli
aws configure
```

```{note}
The requested credentials can be retrieved through your [AWS console](https://aws.amazon.com/console/), typically under “Command line or programmatic access”.
```

Your config and credentials files should follow the standard structure output by `aws configure`:

`~/.aws/config`

```
[default]
region=us-west-2
output=json

```

`~/.aws/credentials`

```
[default]
aws_access_key_id=AKIAIOSFODNN7EXAMPLE
aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

```

More details about the authentication can be found [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).

### Requester Pays Bucket

If the bucket you are accessing is a [Requester Pays](https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html) bucket, then set the below environment variable by providing a bucket name. If there are more than one requester pays bucket, provide each one separated by a comma.

````{tabs}
```{code-tab} py
import os
os.environ['MOSAICML_STREAMING_AWS_REQUESTER_PAYS'] = 'streaming-bucket'

# For more than one requester pays bucket
os.environ['MOSAICML_STREAMING_AWS_REQUESTER_PAYS'] = 'streaming-bucket,another-bucket'
```

```{code-tab} sh
export MOSAICML_STREAMING_AWS_REQUESTER_PAYS='streaming-bucket'

# For more than one requester pays bucket
export MOSAICML_STREAMING_AWS_REQUESTER_PAYS='streaming-bucket,another-bucket'
```
````

## Any S3 compatible object store
For any S3 compatible object store such as [Cloudflare R2](https://www.cloudflare.com/products/r2/), [Coreweave](https://docs.coreweave.com/storage/object-storage), [Backblaze b2](https://www.backblaze.com/b2/cloud-storage.html), etc., setup your credentials as mentioned in the above `Amazon S3` section. The only difference is you must set your object store endpoint url. To do this, you need to set the ``S3_ENDPOINT_URL`` environment variable.

Below is one such example, which sets a R2 `endpoint url` in your run environment.

```{note}
Your endpoint url is `https://<accountid>.r2.cloudflarestorage.com`. The account ID can be retrieved through your [Cloudflare console](https://dash.cloudflare.com/).
```

````{tabs}
```{code-tab} py
import os
os.environ['S3_ENDPOINT_URL'] = 'https://<accountid>.r2.cloudflarestorage.com'
```

```{code-tab} sh
export S3_ENDPOINT_URL='https://<accountid>.r2.cloudflarestorage.com'
```
````

## Google Cloud Storage

### MosaicML platform

For [MosaicML platform](https://www.mosaicml.com/cloud) users, follow the steps mentioned in the [Google Cloud Storage](https://mcli.docs.mosaicml.com/en/latest/secrets/gcp.html) MCLI doc on how to configure the cloud provider credentials.

### Others

Streaming dataset supports [GCP user credentials](https://cloud.google.com/storage/docs/authentication#user_accounts) or [HMAC keys for User account](https://cloud.google.com/storage/docs/authentication/hmackeys).  Users must set their GCP `user access key` and GCP `user access secret` in the run environment.

From the Google Cloud console, navigate to `Google Storage` > `Settings (Left vertical pane)` > `Interoperability` > `Service account HMAC` > `User account HMAC` > `Access keys for your user account` > `Create a key`.

````{tabs}
```{code-tab} py
import os
os.environ['GCS_KEY'] = 'AKIAIOSFODNN7EXAMPLE'
os.environ['GCS_SECRET'] = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
```

```{code-tab} sh
export GCS_KEY='AKIAIOSFODNN7EXAMPLE'
export GCS_SECRET='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
```
````

## Oracle Cloud Storage

### MosaicML platform

For [MosaicML platform](https://www.mosaicml.com/cloud) users, follow the steps mentioned in the [Oracle Cloud Storage](https://mcli.docs.mosaicml.com/en/latest/secrets/oci.html) MCLI doc on how to configure the cloud provider credentials.

### Others

To set up OCI SSH keys and SDK, please read the Oracle Cloud Infrastructure documentation [here](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/devguidesetupprereq.htm).

Specifically:

1. To generate the required keys and OCIDs, follow the instructions [here](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm#Required_Keys_and_OCIDs).
2. To get the SDK/CLI configuration files, follow the link [here](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm#SDK_and_CLI_Configuration_File).

A sample config file (`~/.oci/config`) would look like this:

```
[DEFAULT]
user=ocid1.user.oc1..<unique_ID>
fingerprint=<your_fingerprint>
key_file=~/.oci/oci_api_key.pem
tenancy=ocid1.tenancy.oc1..<unique_ID>
region=us-ashburn-1

```

The key file (`~/.oci/oci_api_key.pem`) is a PEM file that would look like a typical RSA private key file. The streaming dataset authenticates the credentials by reading the `~/.oci/config` and `~/.oci/oci_api_key.pem`.

## Azure Blob Storage

If you wish to create a new storage account, you can use the [Azure Portal](https://docs.microsoft.com/azure/storage/common/storage-quickstart-create-account?tabs=azure-portal), [Azure PowerShell](https://docs.microsoft.com/azure/storage/common/storage-quickstart-create-account?tabs=azure-powershell), or [Azure CLI](https://docs.microsoft.com/azure/storage/common/storage-quickstart-create-account?tabs=azure-cli):

```
# Create a new resource group to hold the storage account -
# if using an existing resource group, skip this step
az group create --name my-resource-group --location westus2

# Create the storage account
az storage account create -n my-storage-account-name -g my-resource-group
```

Users must set their Azure `account name` and Azure `account access key` in the run environment.

The `account access key` can be found in the Azure Portal under the `"Access Keys"` section or by running the following Azure CLI command:

```
az storage account keys list -g MyResourceGroup -n MyStorageAccount
```

````{tabs}
```{code-tab} py
os.environ['AZURE_ACCOUNT_NAME'] = 'test'
os.environ['AZURE_ACCOUNT_ACCESS_KEY'] = 'NN1KHxKKkj20ZO92EMiDQjx3wp2kZG4UUvfAGlgGWRn6sPRmGY/TEST/Dri+ExAmPlEExAmPlExA+ExAmPlExA=='
```

```{code-tab} sh
export AZURE_ACCOUNT_NAME='test'
export AZURE_ACCOUNT_ACCESS_KEY='NN1KHxKKkj20ZO92EMiDQjx3wp2kZG4UUvfAGlgGWRn6sPRmGY/TEST/Dri+ExAmPlEExAmPlExA+ExAmPlExA=='
```
````
