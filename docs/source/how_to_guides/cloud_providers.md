# Cloud provider

Streaming dataset supports the following cloud storage providers: [AWS S3](https://aws.amazon.com/s3/), [GCP Storage](https://cloud.google.com/storage), and [OCI Cloud Storage](https://www.oracle.com/cloud/storage/) to stream your data directly to your instance. For [MCLOUD](https://www.mosaicml.com/cloud) users, follow the steps mentioned in this [AWS S3](https://mcli.docs.mosaicml.com/en/latest/secrets/s3.html), [GCP Storage](https://mcli.docs.mosaicml.com/en/latest/secrets/gcp.html), and [OCI Cloud Storage](https://mcli.docs.mosaicml.com/en/latest/secrets/oci.html) doc on how to configure the cloud provider credentials.

## AWS S3

For S3 bucket with a public access, skip rest of the below steps since no credentials setup is required.

First, make sure the `awscli` is installed, and then run `aws configure` to create the config and credential files:

```
python -m pip install awscli
aws configure
```

Note: the requested credentials can be retrieved through your [AWS console](https://aws.amazon.com/console/), typically under “Command line or programmatic access”.

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

## GCP

Streaming dataset support [GCP user credentials](https://cloud.google.com/storage/docs/authentication#user_accounts), where you need to set your GCP user access key and GCP user access secret as environment variables for your runs. You can set these environment variables in your main python script as such.

```python
import os
os.envionment['GCS_KEY'] = 'AKIAIOSFODNN7EXAMPLE'
os.envionment['GCS_SECRET'] = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
```

Or you can also export the environment from the shell command

```bash
export GCS_KEY='AKIAIOSFODNN7EXAMPLE'
export GCS_SECRET='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
```

The above step will add two environment variables `GCS_KEY` and `GCS_SECRET` to your runs and the streaming dataset fetches those environment variables for authentication and stream data into your instance.

## OCI

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
