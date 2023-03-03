# Configure Cloud Storage Credentials

Streaming dataset supports the following cloud storage providers to stream your data directly to your instance.
- Amazon S3
- Google Cloud Storage
- Oracle Cloud Storage

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

More details on these files can be found [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html). The streaming dataset reads the `~/.aws/config` and `~/.aws/credentials` to authenticate your credentials and stream data into your instance.

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

The above step will add two environment variables `GCS_KEY` and `GCS_SECRET` to your runs and the streaming dataset fetches those environment variables for authentication and stream data into your instance.

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
