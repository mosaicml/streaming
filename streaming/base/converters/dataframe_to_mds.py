# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A utility to convert spark dataframe to MDS."""

import logging
import os
import shutil
import uuid
from collections.abc import Iterable
from tempfile import mkdtemp
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from streaming.base.util import get_import_exception_message
from streaming.base.util import merge_index as do_merge_index

try:
    from pyspark import TaskContext
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        BooleanType,
        ByteType,
        DateType,
        DayTimeIntervalType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        NullType,
        ShortType,
        StringType,
        StructField,
        StructType,
        TimestampNTZType,
        TimestampType,
    )
except ImportError as e:
    e.msg = get_import_exception_message(e.name, extra_deps="spark")  # pyright: ignore
    raise e

from streaming import MDSWriter
from streaming.base.format.index import get_index_basename
from streaming.base.format.mds.encodings import _encodings
from streaming.base.storage.download import CloudDownloader
from streaming.base.storage.upload import CloudUploader

logger = logging.getLogger(__name__)

SPARK_TO_MDS = {
    ByteType(): "uint8",
    ShortType(): "uint16",
    IntegerType(): "int32",
    LongType(): "int64",
    FloatType(): "float32",
    DoubleType(): "float64",
    DecimalType(): "str_decimal",
    StringType(): "str",
    BinaryType(): "bytes",
    BooleanType(): None,
    TimestampType(): None,
    TimestampNTZType(): None,
    DateType(): None,
    DayTimeIntervalType(): None,
    ArrayType(IntegerType()): "ndarray:int32",
    ArrayType(ShortType()): "ndarray:int16",
    ArrayType(LongType()): "ndarray:int64",
    ArrayType(FloatType()): "ndarray:float32",
    ArrayType(DoubleType()): "ndarray:float64",
}


def is_json_compatible(data_type: Any):
    """Recursively check if a given PySpark DataType is JSON compatible.

    JSON = Union[Dict[str, 'JSON'], List['JSON'], str, float, int, bool, None]

    Args:
        data_type (Any): A pyspark schema for a column of the input spark dataframe.

    Returns:
        (bool): True if data_type is JSON compatible.
    """
    if isinstance(data_type, StructType):
        return all(is_json_compatible(field.dataType) for field in data_type.fields)
    elif isinstance(data_type, ArrayType):
        return is_json_compatible(data_type.elementType)
    elif isinstance(data_type, MapType):
        return is_json_compatible(data_type.keyType) and is_json_compatible(
            data_type.valueType
        )
    elif isinstance(
        data_type, (StringType, IntegerType, FloatType, BooleanType, NullType)
    ):
        return True
    else:
        return False


def infer_dataframe_schema(
    dataframe: DataFrame, user_defined_cols: Optional[dict[str, Any]] = None
) -> Optional[dict]:
    """Retrieve schema to construct a dictionary or do sanity check for dataframe_to_mds.

    Args:
        dataframe (spark dataframe): dataframe to inspect schema
        user_defined_cols (Optional[Dict[str, Any]]): user specified schema for dataframe_to_mds

    Returns:
        If user_defined_cols is None, return schema_dict (dict): column name and dtypes that are
        supported by MDSWriter, else None

    Raises:
        ValueError if any of the datatypes are unsupported by dataframe_to_mds.
    """

    def map_spark_dtype(spark_data_type: Any) -> str:
        """Map spark data type to mds supported types.

        Args:
            spark_data_type: https://spark.apache.org/docs/latest/sql-ref-datatypes.html

        Returns:
            str: corresponding mds datatype for input.

        Raises:
            raise ValueError if no mds datatype is found for input type
        """
        if issubclass(type(spark_data_type), DecimalType):
            mds_type = SPARK_TO_MDS.get(DecimalType(), None)
        else:
            mds_type = SPARK_TO_MDS.get(spark_data_type, None)

        if mds_type is None:
            raise ValueError(f"{spark_data_type} is not supported by dataframe_to_mds")
        return mds_type

    # user has provided schema, we just check if mds supports the dtype
    if user_defined_cols is not None:
        mds_supported_dtypes = set(filter(bool, SPARK_TO_MDS.values()))

        for col_name, user_dtype in user_defined_cols.items():
            if col_name not in dataframe.columns:
                raise ValueError(
                    f"{col_name} is not a column of input dataframe: {dataframe.columns}"
                )

            if user_dtype.startswith("ndarray:"):
                parts = user_dtype.split(":")
                if len(parts) == 3:
                    user_dtype = ":".join(parts[:-1])

            actual_spark_dtype = dataframe.schema[col_name].dataType

            if user_dtype not in mds_supported_dtypes:
                if user_dtype == "json":
                    if is_json_compatible(actual_spark_dtype):
                        continue
                    else:
                        raise ValueError(f"{col_name} can not be encoded by MDS JSON.")
                elif user_dtype in ("pil", "png", "jpeg", "pkl"):
                    if isinstance(actual_spark_dtype, BinaryType):
                        continue
                    else:
                        raise ValueError(
                            f"Non-binary col {col_name} cannot encode as {user_dtype}."
                        )
                raise ValueError(f"{user_dtype} is not supported by dataframe_to_mds")

            mapped_mds_dtype = map_spark_dtype(actual_spark_dtype)
            if user_dtype != mapped_mds_dtype:
                raise ValueError(
                    f"Mismatched types: column name `{col_name}` is `{mapped_mds_dtype}` in "
                    + f"DataFrame but `{user_dtype}` in user_defined_cols"
                )
        return None

    schema = dataframe.schema
    schema_dict = {}

    for field in schema:
        dtype = map_spark_dtype(field.dataType)
        if dtype.split(":")[0] in _encodings:
            schema_dict[field.name] = dtype
        else:
            raise ValueError(f"{dtype} is not supported by dataframe_to_mds")
    return schema_dict


def dataframe_to_mds(
    dataframe: DataFrame,
    merge_index: bool = True,
    mds_kwargs: Optional[dict[str, Any]] = None,
    udf_iterable: Optional[Callable] = None,
    udf_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """Execute a Spark DataFrame to MDS conversion process.

    This method orchestrates the conversion of a Spark DataFrame into MDS format.
    It processes the input data in a distributed manner, applying a user-defined
    iterable function if provided, and writes the results to an MDS-compatible format.
    The converted data is saved to mds_path

    Key Features:
    - Aggressive cleanup of local worker files to minimize disk usage on large datasets.
    - Downloads partition indices to the driver for merging, allowing for headless
      execution on workers.

    Args:
        dataframe (pyspark.sql.DataFrame): A DataFrame containing the data.
        merge_index (bool): Whether to merge MDS index files. Defaults to ``True``.
        mds_kwargs (dict): Arguments for MDSWriter. Must contain 'out' and 'columns'.
            Refer to https://docs.mosaicml.com/projects/streaming/en/stable/
            api_reference/generated/streaming.MDSWriter.html
        udf_iterable (Callable or None): A user-defined function that returns an iterable
            over the dataframe. udf_kwargs is the k-v args for the method.
            Defaults to ``None``.
        udf_kwargs (Dict): Additional keyword arguments to pass to the udf_iterable.
            Defaults to an empty dictionary.

    Returns:
        mds_path (str or (str, str)): The local and remote path where the MDS dataset
            was saved.
    """

    # Worker Logic
    def write_mds(iterator: Iterable):
        """Worker function: writes iterable to MDS datasets locally.

        Returns remote index path.
        """
        context = TaskContext.get()

        if context is None:
            raise RuntimeError("TaskContext.get() returns None")

        task_id = context.taskAttemptId()

        # Create unique task subdirectory to prevent collisions
        # We use the configured output paths
        # local: /tmp/dataset/task_123
        # remote: s3://bucket/dataset/task_123
        task_sub_path = f"task_{task_id}"

        task_local = os.path.join(mds_path[0], task_sub_path)
        task_remote = os.path.join(mds_path[1], task_sub_path) if mds_path[1] else ""

        if task_remote:
            task_out = (task_local, task_remote)
        else:
            task_out = task_local

        # Prepare kwargs for this specific task
        partition_kwargs = mds_kwargs.copy()
        partition_kwargs["out"] = task_out

        # We enforce keep_local=False for the workers to save disk space during massive jobs.
        # The driver will handle the merge by downloading indices later.
        partition_kwargs["keep_local"] = False

        # Write data
        with MDSWriter(**partition_kwargs) as mds_writer:
            for pdf in iterator:
                if udf_iterable is not None:
                    records = udf_iterable(pdf, **udf_kwargs)
                else:
                    records = pdf.to_dict("records")

                if not isinstance(records, Iterable):
                    raise TypeError(
                        f"udf_iterable must return an Iterable, got {type(records)}"
                    )

                for sample in records:
                    mds_writer.write(sample)

        # Return the location of the index file.
        # Since keep_local=False, the local file is deleted. We return the remote path.
        # If running purely locally (no remote), we return the local path.
        index_filename = get_index_basename()

        if task_remote:
            index_path = os.path.join(task_remote, index_filename)
        else:
            index_path = os.path.join(task_local, index_filename)

        yield pd.DataFrame({"mds_index_path": [index_path]})

    # Input Validation & Setup
    if dataframe is None or dataframe.isEmpty():
        raise ValueError("Input dataframe is None or Empty!")

    if not mds_kwargs:
        mds_kwargs = {}

    if not udf_kwargs:
        udf_kwargs = {}

    if "out" not in mds_kwargs:
        raise ValueError("`out` and `columns` need to be specified in `mds_kwargs`")

    # Default compression
    if "compression" not in mds_kwargs:
        logger.info("Defaulting to zstd compression")
        mds_kwargs["compression"] = "zstd"

    # Schema Inference
    if udf_iterable is not None:
        if "columns" not in mds_kwargs:
            raise ValueError(
                "If udf_iterable is specified, user must provide correct `columns` in mds_kwargs"
            )
        logger.warning(
            "With udf_iterable defined, it's up to the user's discretion to provide "
            + "mds_kwargs[columns]'"
        )
    else:
        if "columns" not in mds_kwargs:
            logger.warning(
                "Columns arg is missing from mds_kwargs. Auto-inferring schema from DataFrame."
            )
            mds_kwargs["columns"] = infer_dataframe_schema(dataframe)
            logger.warning(f"Auto inferred schema: {mds_kwargs['columns']}")
        else:
            # Validate existing columns against dataframe
            infer_dataframe_schema(dataframe, mds_kwargs["columns"])

    out = mds_kwargs["out"]

    # Handle local vs remote path parsing using CloudUploader logic
    # We pass keep_local=False here because we handle the merge via download later
    cu = CloudUploader.get(out, keep_local=keep_local)

    if cu.remote is None:
        # Handling purely local paths (or FUSE mounted paths)
        # If dataframe_to_mds is being called, this is in a distributed Spark env.
        # If there is no remote, it's because the given out path is local, which does not
        # in general make sense, unless this 'local' path is FUSE-mounted distributed
        # storage such as /dbfs or /Volumes in Databricks for example.
        # It's not wrong in this case, but probably nevertheless desirable to specify a local temp
        # path explicitly, to interpret the FUSE-mounted path as remote
        logger.warning(
            f"Path {cu.local} is interpreted as a local path. If this is actually "
            + "mounted distributed storage, it will work, but consider also specifying "
            + 'a local temp path. Pass a (local, remote) tuple as "out", as in '
            + f'("/local_disk0/my_tmp", "{cu.local}")'
        )
        mds_path = (cu.local, "")
        remote_root = cu.local  # For local execution, "remote" is just the output dir
    else:
        mds_path = (cu.local, cu.remote)
        remote_root = cu.remote

    # Distributed Execution
    # Upload intermediate shards to remote_root and aggressively purge intermediate
    # local shards on executors to minimize disk usage.
    # Write the metadata (paths to index files) to a remote_root location.
    unique_run_id = str(uuid.uuid4())[:8]
    metadata_dump_path = os.path.join(remote_root, f"_spark_metadata_{unique_run_id}")

    logger.info(f"Starting distributed write. Metadata buffer: {metadata_dump_path}")

    # partition schema
    result_schema = StructType([StructField("mds_index_path", StringType(), False)])

    try:
        # Trigger the Spark job
        (
            dataframe.mapInPandas(func=write_mds, schema=result_schema)
            .write.mode("overwrite")
            .parquet(metadata_dump_path)
        )

        logger.info("Distributed MDS write complete. Starting index merge phase.")

        # Merge Index Phase
        if merge_index:
            spark = dataframe.sparkSession

            # Read the list of remote index files from the temporary parquet dump
            metadata_df = spark.read.parquet(metadata_dump_path)
            # Collect only the shard index file paths (strings)
            remote_indices = [row["mds_index_path"] for row in metadata_df.collect()]

            # Prepare a local temporary directory on the driver to download indices
            tmpdir = mkdtemp()
            driver_tmpdir = os.path.join(tmpdir, f"merger_tmp_{unique_run_id}")

            if os.path.exists(driver_tmpdir):
                shutil.rmtree(driver_tmpdir)
            os.makedirs(driver_tmpdir)

            tuples_for_merger = []

            logger.info(f"Downloading {len(remote_indices)} index files for merging...")

            try:
                for remote_file in remote_indices:
                    # Construct a local mirror path:
                    # Remote: s3://bucket/data/task_0/index.json
                    # Local:  ./merger_tmp/task_0/index.json

                    # Robust relative path extraction
                    if mds_path[1]:  # Remote exists
                        rel_path = remote_file.replace(mds_path[1], "").strip("/")
                    else:
                        rel_path = remote_file.replace(mds_path[0], "").strip("/")

                    local_dest = os.path.join(driver_tmpdir, rel_path)
                    os.makedirs(os.path.dirname(local_dest), exist_ok=True)

                    # If we are in remote mode, we must download.
                    # If purely local mode, the file exists on disk (if shared fs)
                    # or we copy it.
                    if mds_path[1]:
                        CloudDownloader.get(remote_file).download(
                            remote_file, local_dest
                        )
                    else:
                        # Fallback for shared-fs/local setups where keep_local=False might have deleted it
                        # If keep_local=False ran locally, the file might be gone unless the user
                        # used a shared mount that persists.
                        # Assuming standard object store usage here.
                        if not os.path.exists(remote_file):
                            logger.warning(
                                f"Index file missing for local merge: {remote_file}"
                            )
                        else:
                            shutil.copy(remote_file, local_dest)

                    tuples_for_merger.append((local_dest, remote_file))

                if cu.remote is not None:
                    # remote mode
                    keep_local = False
                else:
                    keep_local = True

                do_merge_index(
                    tuples_for_merger, out, keep_local=keep_local, download_timeout=60
                )
                logger.info("Master index merged and uploaded successfully.")

            finally:
                # Cleanup driver temporary directory
                shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Failed during MDS conversion: {e}")
        raise e
    finally:
        # Global Cleanup
        # Remove the temporary metadata dump folder from the remote storage
        # Note: We need a way to delete the folder. CloudUploader/Downloader usually handles files.
        # This is a best-effort cleanup for the parquet dump.
        try:
            # If `remote_root` supports filesystem operations (like dbfs or local), shutil works.
            # If it is s3://, gs:// shutil fails.
            # In a strict production environment, we might leave this or use cloud-specific delete.
            # For this PR, we attempt local cleanup if applicable.
            if os.path.exists(metadata_dump_path):
                shutil.rmtree(metadata_dump_path, ignore_errors=True)
        except Exception:
            pass

    return mds_path
