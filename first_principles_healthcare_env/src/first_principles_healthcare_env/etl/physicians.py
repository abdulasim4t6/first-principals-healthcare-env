"""
ETL/ELT module for processing physician data in Databricks with PySpark.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, TimestampType


def get_schema() -> StructType:
    """
    Returns the schema for physician data.

    Returns:
        StructType: Schema for physician data
    """
    return StructType([
        StructField("physician_id", StringType(), False),
        StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True),
        StructField("specialty", StringType(), True),
        StructField("license_number", StringType(), True),
        StructField("email", StringType(), True),
        StructField("contact_number", StringType(), True),
        StructField("department", StringType(), True),
        StructField("hire_date", DateType(), True),
        StructField("status", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True)
    ])


def extract_physicians(spark: SparkSession, source_path: str) -> DataFrame:
    """
    Extract physician data from source.

    Args:
        spark: SparkSession object
        source_path: Path to the source data

    Returns:
        DataFrame: Extracted physician data
    """
    return spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .schema(get_schema()) \
        .load(source_path)


def transform_physicians(df: DataFrame) -> DataFrame:
    """
    Transform physician data.

    Args:
        df: DataFrame with physician data

    Returns:
        DataFrame: Transformed physician data
    """
    # Clean and transform physician data
    transformed_df = df.withColumn("full_name",
                                   F.concat(F.col("first_name"), F.lit(" "), F.col("last_name"))) \
        .withColumn("years_employed",
                    F.floor(F.datediff(F.current_date(), F.col("hire_date")) / 365.25)) \
        .withColumn("processed_timestamp", F.current_timestamp())

    # Remove any duplicates
    transformed_df = transformed_df.dropDuplicates(["physician_id"])

    return transformed_df


def load_physicians(df: DataFrame, target_path: str, table_name: str = "physicians") -> None:
    """
    Load transformed physician data into the target.

    Args:
        df: DataFrame with transformed physician data
        target_path: Path to the target location
        table_name: Name of the table to create/update
    """
    # Write to Delta table
    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(target_path)

    # Create or replace table
    spark = SparkSession.getActiveSession()
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table_name} USING DELTA LOCATION '{target_path}'")


def run_pipeline(spark: SparkSession, source_path: str, target_path: str, table_name: str = "physicians") -> None:
    """
    Run the complete ETL/ELT pipeline for physician data.

    Args:
        spark: SparkSession object
        source_path: Path to the source data
        target_path: Path to the target location
        table_name: Name of the table to create/update
    """
    # Extract
    physicians_df = extract_physicians(spark, source_path)

    # Transform
    transformed_df = transform_physicians(physicians_df)

    # Load
    load_physicians(transformed_df, target_path, table_name)

    print(
        f"Physician ETL completed: data loaded to {target_path} and registered as table {table_name}")
