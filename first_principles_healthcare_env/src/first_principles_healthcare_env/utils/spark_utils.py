"""
Utility functions for working with Spark.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pyspark.sql.types as T
from typing import List, Dict, Any, Optional


def get_spark_session(app_name: str = "Healthcare Data Pipeline") -> SparkSession:
    """
    Get or create a Spark session.

    Args:
        app_name: Name of the Spark application

    Returns:
        SparkSession: The Spark session
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()


def create_database(spark: SparkSession, db_name: str, if_not_exists: bool = True) -> None:
    """
    Create a database in Spark.

    Args:
        spark: SparkSession object
        db_name: Name of the database to create
        if_not_exists: Whether to use IF NOT EXISTS clause
    """
    query = f"CREATE DATABASE {('IF NOT EXISTS' if if_not_exists else '')} {db_name}"
    spark.sql(query)
    print(f"Database {db_name} created")


def table_exists(spark: SparkSession, table_name: str) -> bool:
    """
    Check if a table exists.

    Args:
        spark: SparkSession object
        table_name: Name of the table to check

    Returns:
        bool: True if the table exists, False otherwise
    """
    tables = spark.sql("SHOW TABLES").select(
        "tableName").rdd.flatMap(lambda x: x).collect()
    return table_name in tables


def column_exists(df: DataFrame, column_name: str) -> bool:
    """
    Check if a column exists in a DataFrame.

    Args:
        df: DataFrame to check
        column_name: Name of the column to check

    Returns:
        bool: True if the column exists, False otherwise
    """
    return column_name in df.columns


def add_surrogate_key(df: DataFrame, key_column: str = "surrogate_key") -> DataFrame:
    """
    Add a surrogate key to a DataFrame.

    Args:
        df: DataFrame to add surrogate key to
        key_column: Name of the surrogate key column

    Returns:
        DataFrame: DataFrame with surrogate key column
    """
    return df.withColumn(key_column, F.monotonically_increasing_id())


def add_partition_columns(df: DataFrame, date_col: str = None) -> DataFrame:
    """
    Add partition columns (year, month, day) to a DataFrame.

    Args:
        df: DataFrame to add partition columns to
        date_col: Name of the date column to extract partitions from

    Returns:
        DataFrame: DataFrame with partition columns
    """
    if date_col is None:
        date_col = next(
            (col for col in df.columns if "date" in col.lower()), None)

    if date_col is None:
        raise ValueError("No date column found and none specified")

    return df.withColumn("year", F.year(F.col(date_col))) \
             .withColumn("month", F.month(F.col(date_col))) \
             .withColumn("day", F.dayofmonth(F.col(date_col)))


def print_dataframe_info(df: DataFrame, sample_size: int = 5) -> None:
    """
    Print information about a DataFrame.

    Args:
        df: DataFrame to print information about
        sample_size: Number of rows to sample
    """
    print(f"DataFrame has {df.count()} rows and {len(df.columns)} columns")
    print("Schema:")
    df.printSchema()
    print(f"Sample of {sample_size} rows:")
    df.show(sample_size, truncate=False)


def write_to_delta(df: DataFrame, path: str, mode: str = "overwrite", partition_by: List[str] = None) -> None:
    """
    Write a DataFrame to a Delta table.

    Args:
        df: DataFrame to write
        path: Path to save the Delta table
        mode: Write mode (append, overwrite, etc.)
        partition_by: List of columns to partition by
    """
    writer = df.write.format("delta").mode(
        mode).option("overwriteSchema", "true")

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.save(path)
