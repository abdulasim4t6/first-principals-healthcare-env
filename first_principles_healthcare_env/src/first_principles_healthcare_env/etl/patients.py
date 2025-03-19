"""
ETL/ELT module for processing patient data in Databricks with PySpark.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, TimestampType


def get_schema() -> StructType:
    """
    Returns the schema for patient data.

    Returns:
        StructType: Schema for patient data
    """
    return StructType([
        StructField("patient_id", StringType(), False),
        StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True),
        StructField("date_of_birth", DateType(), True),
        StructField("gender", StringType(), True),
        StructField("contact_number", StringType(), True),
        StructField("email", StringType(), True),
        StructField("address", StringType(), True),
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("zip_code", StringType(), True),
        StructField("insurance_provider", StringType(), True),
        StructField("insurance_id", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True)
    ])


def extract_patients(spark: SparkSession, source_path: str) -> DataFrame:
    """
    Extract patient data from source.

    Args:
        spark: SparkSession object
        source_path: Path to the source data

    Returns:
        DataFrame: Extracted patient data
    """
    return spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .schema(get_schema()) \
        .load(source_path)


def transform_patients(df: DataFrame) -> DataFrame:
    """
    Transform patient data.

    Args:
        df: DataFrame with patient data

    Returns:
        DataFrame: Transformed patient data
    """
    # Clean and transform patient data
    transformed_df = df.withColumn("full_name",
                                   F.concat(F.col("first_name"), F.lit(" "), F.col("last_name"))) \
        .withColumn("age",
                    F.floor(F.datediff(F.current_date(), F.col("date_of_birth")) / 365.25)) \
        .withColumn("processed_timestamp", F.current_timestamp())

    return transformed_df


def load_patients(df: DataFrame, target_path: str, table_name: str = "patients") -> None:
    """
    Load transformed patient data into the target.

    Args:
        df: DataFrame with transformed patient data
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


def run_pipeline(spark: SparkSession, source_path: str, target_path: str, table_name: str = "patients") -> None:
    """
    Run the complete ETL/ELT pipeline for patient data.

    Args:
        spark: SparkSession object
        source_path: Path to the source data
        target_path: Path to the target location
        table_name: Name of the table to create/update
    """
    # Extract
    patients_df = extract_patients(spark, source_path)

    # Transform
    transformed_df = transform_patients(patients_df)

    # Load
    load_patients(transformed_df, target_path, table_name)

    print(
        f"Patient ETL completed: data loaded to {target_path} and registered as table {table_name}")
