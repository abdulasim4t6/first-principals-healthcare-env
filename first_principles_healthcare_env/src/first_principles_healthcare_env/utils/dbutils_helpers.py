"""
Helper functions for working with Databricks utilities (dbutils).
"""
from typing import Optional, List, Dict, Any
import os


def get_dbutils():
    """
    Get the Databricks utilities object.

    Returns:
        dbutils: The Databricks utilities object
    """
    if 'dbutils' in globals():
        return dbutils
    else:
        from pyspark.sql import SparkSession
        from pyspark.dbutils import DBUtils
        spark = SparkSession.builder.getOrCreate()
        return DBUtils(spark)


def mount_s3_bucket(bucket_name: str, mount_point: str, aws_access_key: Optional[str] = None,
                    aws_secret_key: Optional[str] = None) -> None:
    """
    Mount an S3 bucket to DBFS.

    Args:
        bucket_name: Name of the S3 bucket
        mount_point: DBFS path to mount the bucket to
        aws_access_key: AWS access key (optional, will use instance profile if not provided)
        aws_secret_key: AWS secret key (optional, will use instance profile if not provided)
    """
    dbutils = get_dbutils()

    # Check if the mount point already exists
    mounts = dbutils.fs.mounts()
    for mount in mounts:
        if mount.mountPoint == mount_point:
            print(f"Mount point {mount_point} already exists")
            return

    # Mount the bucket
    if aws_access_key and aws_secret_key:
        dbutils.fs.mount(
            source=f"s3a://{bucket_name}",
            mount_point=mount_point,
            extra_configs={
                "fs.s3a.access.key": aws_access_key,
                "fs.s3a.secret.key": aws_secret_key
            }
        )
    else:
        # Use instance profile
        dbutils.fs.mount(
            source=f"s3a://{bucket_name}",
            mount_point=mount_point
        )

    print(f"Mounted s3://{bucket_name} at {mount_point}")


def unmount_s3_bucket(mount_point: str) -> None:
    """
    Unmount an S3 bucket from DBFS.

    Args:
        mount_point: DBFS path where the bucket is mounted
    """
    dbutils = get_dbutils()

    # Check if the mount point exists
    mounts = dbutils.fs.mounts()
    for mount in mounts:
        if mount.mountPoint == mount_point:
            dbutils.fs.unmount(mount_point)
            print(f"Unmounted {mount_point}")
            return

    print(f"Mount point {mount_point} does not exist")


def create_directory_if_not_exists(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Path to create
    """
    dbutils = get_dbutils()

    try:
        dbutils.fs.ls(path)
    except:
        dbutils.fs.mkdirs(path)
        print(f"Created directory {path}")


def list_files(path: str, recursive: bool = False) -> List[str]:
    """
    List files in a directory.

    Args:
        path: Path to list files from
        recursive: Whether to list files recursively

    Returns:
        List[str]: List of file paths
    """
    dbutils = get_dbutils()

    files = []

    def _list_files(p):
        for file_info in dbutils.fs.ls(p):
            if file_info.isDir() and recursive:
                _list_files(file_info.path)
            else:
                files.append(file_info.path)

    _list_files(path)
    return files


def get_notebook_context() -> Dict[str, Any]:
    """
    Get the current notebook context.

    Returns:
        Dict[str, Any]: Dictionary with notebook context information
    """
    dbutils = get_dbutils()

    return {
        "notebook_path": dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get(),
        "notebook_id": dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookId().get(),
        "notebook_url": dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get(),
        "cluster_id": dbutils.notebook.entry_point.getDbutils().notebook().getContext().clusterId().get(),
        "spark_version": dbutils.notebook.entry_point.getDbutils().notebook().getContext().sparkVersion().get()
    }


def get_secrets(scope: str, key: str) -> str:
    """
    Get a secret from Databricks secrets.

    Args:
        scope: Scope of the secret
        key: Key of the secret

    Returns:
        str: Value of the secret
    """
    dbutils = get_dbutils()

    return dbutils.secrets.get(scope=scope, key=key)
