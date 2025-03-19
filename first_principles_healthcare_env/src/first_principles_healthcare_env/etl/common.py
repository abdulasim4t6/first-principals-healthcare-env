"""
Common utilities for ETL/ELT processes in Databricks.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
import datetime


def get_spark_session() -> SparkSession:
    """
    Get or create a Spark session.

    Returns:
        SparkSession: The Spark session
    """
    return SparkSession.builder \
        .appName("Healthcare ETL") \
        .getOrCreate()


def add_audit_columns(df: DataFrame) -> DataFrame:
    """
    Add audit columns to a DataFrame.

    Args:
        df: DataFrame to add audit columns to

    Returns:
        DataFrame: DataFrame with audit columns
    """
    return df.withColumn("etl_timestamp", F.current_timestamp()) \
             .withColumn("etl_date", F.current_date()) \
             .withColumn("etl_batch_id", F.lit(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))


def validate_dataframe(df: DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        bool: True if all required columns are present, False otherwise
    """
    df_columns = df.columns
    missing_columns = [
        col for col in required_columns if col not in df_columns]

    if missing_columns:
        print(f"Validation failed. Missing columns: {missing_columns}")
        return False
    return True


def write_to_delta(df: DataFrame, path: str, table_name: str, mode: str = "overwrite") -> None:
    """
    Write a DataFrame to a Delta table.

    Args:
        df: DataFrame to write
        path: Path to save the Delta table
        table_name: Name of the table to create/update
        mode: Write mode (append, overwrite, etc.)
    """
    # Add audit columns
    df_with_audit = add_audit_columns(df)

    # Write to Delta
    df_with_audit.write.format("delta") \
        .mode(mode) \
        .option("overwriteSchema", "true") \
        .save(path)

    # Create or replace table
    spark = SparkSession.getActiveSession()
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table_name} USING DELTA LOCATION '{path}'")

    print(f"Data successfully written to {table_name} at {path}")


def cleanup_table_history(table_name: str, retain_days: int = 30) -> None:
    """
    Clean up the history of a Delta table to save storage.

    Args:
        table_name: Name of the table to clean up
        retain_days: Number of days of history to retain
    """
    spark = SparkSession.getActiveSession()
    spark.sql(f"VACUUM {table_name} RETAIN {retain_days} DAYS")
    print(
        f"Successfully vacuumed {table_name}, retaining {retain_days} days of history")
