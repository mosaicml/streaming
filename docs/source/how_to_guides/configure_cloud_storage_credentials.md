# Configure Cloud Storage Credentials

Streaming dataset supports the following cloud storage providers to stream your data directly to your instance.
- [Amazon S3](#amazon-s3)
- [Any S3 compatible object store](#any-s3-compatible-object-store)
- [Google Cloud Storage](#google-cloud-storage)
- [Oracle Cloud Storage](#oracle-cloud-storage)
- [Azure Blob Storage](#azure-blob-storage-and-azure-datalake)
- [Databricks](#databricks)

## Amazon S3

For an S3 bucket with public access, no additional setup is required, simply specify the S3 URI of the resource.

### Mosaic AI Training

For [Mosaic AI Training](https://docs.mosaicml.com/projects/mcli/en/latest/) users, follow the steps mentioned in the [AWS S3](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/s3.html) MCLI documentation page on how to configure the cloud provider credentials.

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
region=<your region, e.g. us-west-2>
output=json

```

`~/.aws/credentials`

```
[default]
aws_access_key_id=<key ID>
aws_secret_access_key=<application key>

```

More details about the authentication can be found [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).

Alternatively, this can also be set through [environment variables](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html).

````{tabs}
```{code-tab} py
import os
os.environ["AWS_ACCESS_KEY_ID"] = '<key ID>'
os.environ["AWS_SECRET_ACCESS_KEY"] = '<application key>'
os.environ["AWS_DEFAULT_REGION"] = '<your region, e.g. us-west-2>'
```

```{code-tab} sh
export AWS_ACCESS_KEY_ID='<key ID>'
export AWS_SECRET_ACCESS_KEY='<application key>'
export AWS_DEFAULT_REGION='<your region, e.g. us-west-2>'
```
````


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

### Canned ACL
Canned ACLs (Access Control Lists) are predefined sets of permissions in AWS S3 that you can apply to your objects. These can simplify the process of managing access to your S3 resources. Examples of canned ACLs include `private`, `public-read`, `public-read-write`, `authenticated-read`, etc. You can set a canned ACL for your S3 objects by using the `S3_CANNED_ACL` environment variable. This allows you to manage access permissions to your S3 resources in a simplified manner.

````{tabs}
```{code-tab} py
import os
os.environ['S3_CANNED_ACL'] = 'authenticated-read'
```

```{code-tab} sh
export S3_CANNED_ACL='authenticated-read'
```
````

## Any S3 compatible object store
For any S3 compatible object store such as [Cloudflare R2](https://www.cloudflare.com/products/r2/), [Coreweave](https://docs.coreweave.com/storage/object-storage), [Backblaze b2](https://www.backblaze.com/b2/cloud-storage.html), etc., set up your credentials as mentioned in the above `Amazon S3` section. Alternatively, you may use the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variable names to specify your credentials, even though you are not using AWS. The only difference is that you must set your object store endpoint url. To do this, you need to set the ``S3_ENDPOINT_URL`` environment variable.

Below are the examples of setting an R2 or Backblaze `endpoint url` in your run environment.

```{note}
R2: Your endpoint url is `https://<accountid>.r2.cloudflarestorage.com`. The account ID can be retrieved through your [Cloudflare console](https://dash.cloudflare.com/).
Backblaze: Your endpoint url is 'https://s3.<your region>.backblazeb2.com'. The region can be retrieved through your [Backblaze console](https://secure.backblaze.com/b2_buckets.htm).
```

````{tabs}
```{code-tab} py
import os
# If using R2
os.environ['S3_ENDPOINT_URL'] = 'https://<accountid>.r2.cloudflarestorage.com'
# If using Backblaze
os.environ['S3_ENDPOINT_URL'] = 'https://s3.<your region>.backblazeb2.com'
```

```{code-tab} sh
# If using R2
export S3_ENDPOINT_URL='https://<accountid>.r2.cloudflarestorage.com'
# If using Backblaze
export S3_ENDPOINT_URL='https://s3.<your region>.backblazeb2.com'
```
````


Note that even with S3 compatible object stores, URLs should be of the form `s3://<bucket name>/<path within the bucket>` and use the `s3://` path prefix, instead of `<endpoint url>/<bucket name>/<path within the bucket>`.


## Google Cloud Storage

### Mosaic AI Training

For [Mosaic AI Training](https://docs.mosaicml.com/projects/mcli/en/latest/) users, follow the steps mentioned in the [Google Cloud Storage](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/gcp.html) MCLI documentation page on how to configure the cloud provider credentials.


### GCP User Auth Credentials Mounted as Environment Variables

Streaming dataset supports [GCP user credentials](https://cloud.google.com/storage/docs/authentication#user_accounts) or [HMAC keys for User account](https://cloud.google.com/storage/docs/authentication/hmackeys).  Users must set their GCP `user access key` and GCP `user access secret` in the run environment.

From the Google Cloud console, navigate to `Google Storage` > `Settings (Left vertical pane)` > `Interoperability` > `Service account HMAC` > `User account HMAC` > `Access keys for your user account` > `Create a key`.

````{tabs}
```{code-tab} py
import os
os.environ['GCS_KEY'] = 'EXAMPLEFODNN7EXAMPLE'
os.environ['GCS_SECRET'] = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
```

```{code-tab} sh
export GCS_KEY='EXAMPLEFODNN7EXAMPLE'
export GCS_SECRET='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
```
````


###  GCP Application Default Credentials

Streaming dataset supports the use of Application Default Credentials (ADC) to authenticate you with Google Cloud. When
no HMAC keys are given (see above), it will attempt to authenticate using ADC. This will, in order, check

1. a key-file whose path is given in the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
2. a key-file in the Google cloud configuration directory.
3. the Google App Engine credentials.
4. the GCE Metadata Service credentials.

See the [Google Cloud Docs](https://cloud.google.com/docs/authentication/provide-credentials-adc) for more details.

To explicitly use the `GOOGLE_APPLICATION_CREDENTIALS` (point 1 above), users must set their GCP `account credentials`
to point to their credentials file in the run environment.

````{tabs}
```{code-tab} py
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'KEY_FILE'
```

```{code-tab} sh
export GOOGLE_APPLICATION_CREDENTIALS='KEY_FILE'
```
````


## Oracle Cloud Storage

### Mosaic AI Training

For [Mosaic AI Training](https://docs.mosaicml.com/projects/mcli/en/latest/) users, follow the steps mentioned in the [Oracle Cloud Storage](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/oci.html) MCLI documentation page on how to configure the cloud provider credentials.

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

## Azure Blob Storage and Azure DataLake

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

## Databricks

To authenticate Databricks access for both Unity Catalog and Databricks File System (DBFS), users must set their Databricks host (`DATABRICKS_HOST`) and access token (`DATABRICKS_TOKEN`) in the run environment.

See the [Databricks documentation](https://docs.databricks.com/en/dev-tools/auth.html#databricks-personal-access-token-authentication) for instructions on how to create a personal access token.

### Mosaic AI Training

For [Mosaic AI Training](https://docs.mosaicml.com/projects/mcli/en/latest/) users, follow the steps mentioned in the [Databricks](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/databricks.html) MCLI documentation page on how to configure the credentials.

### Others

````{tabs}
```{code-tab} py
os.environ['DATABRICKS_HOST'] = 'hostname'
os.environ['DATABRICKS_TOKEN'] = 'token key'
```

```{code-tab} sh
export DATABRICKS_HOST='hostname'
export DATABRICKS_TOKEN='token key'
```
````


## Alipan

To authenticate Alipan access, users must set their Alipan refresh token (`ALIPAN_WEB_REFRESH_TOKEN`) in the run environment.

To get the refresh token from the Alipan website, the user needs to login to the [Alipan website](https://www.alipan.com/drive), go to the console of the browser's devTools, and pass the code below into the console to get the refresh token.

```javascript
JSON.parse(localStorage.token).refresh_token;
```

Then set the refresh token in the run environment.

````{tabs}
```{code-tab} py
import os
os.environ['ALIPAN_WEB_REFRESH_TOKEN'] = 'refresh_token'
```

```{code-tab} sh
export ALIPAN_WEB_REFRESH_TOKEN='refresh_token'
```
````

### Encryption Data

If you want to encrypt data in cloud storage, you can set the environment variable below to encrypt and decrypt the data.

````{tabs}
```{code-tab} py
import os
os.environ['ALIPAN_ENCRYPT_PASSWORD'] = 'encryption_key'

# For uploading, the encryption type must be set as one of `Simple`, `ChaCha20`, `AES256CBC`
# When downloading, this environment variable is not required
os.environ['ALIPAN_ENCRYPT_TYPE'] = 'AES256CBC'
```

```{code-tab} sh
export ALIPAN_ENCRYPT_PASSWORD='encryption_key'

# For uploading, the encryption type must be set as one of `Simple`, `ChaCha20`, `AES256CBC`
# When downloading, this environment variable is not required
export ALIPAN_ENCRYPT_TYPE='AES256CBC'
```
````
