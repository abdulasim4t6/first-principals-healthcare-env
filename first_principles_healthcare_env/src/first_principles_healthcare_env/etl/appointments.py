"""
ETL/ELT module for processing appointment data in Databricks with PySpark.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, TimestampType, BooleanType


def get_schema() -> StructType:
    """
    Returns the schema for appointment data.

    Returns:
        StructType: Schema for appointment data
    """
    return StructType([
        StructField("appointment_id", StringType(), False),
        StructField("patient_id", StringType(), False),
        StructField("physician_id", StringType(), False),
        StructField("appointment_date", DateType(), True),
        StructField("appointment_time", TimestampType(), True),
        StructField("duration_minutes", IntegerType(), True),
        StructField("reason", StringType(), True),
        StructField("status", StringType(), True),
        StructField("no_show", BooleanType(), True),
        StructField("notes", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True)
    ])


def extract_appointments(spark: SparkSession, source_path: str) -> DataFrame:
    """
    Extract appointment data from source.

    Args:
        spark: SparkSession object
        source_path: Path to the source data

    Returns:
        DataFrame: Extracted appointment data
    """
    return spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .schema(get_schema()) \
        .load(source_path)


def transform_appointments(df: DataFrame, patients_df: DataFrame = None, physicians_df: DataFrame = None) -> DataFrame:
    """
    Transform appointment data.

    Args:
        df: DataFrame with appointment data
        patients_df: Optional DataFrame with patient data for enrichment
        physicians_df: Optional DataFrame with physician data for enrichment

    Returns:
        DataFrame: Transformed appointment data
    """
    # Clean and transform appointment data
    transformed_df = df.withColumn("day_of_week",
                                   F.date_format(F.col("appointment_date"), "EEEE")) \
        .withColumn("is_weekend",
                    F.when(F.date_format(F.col("appointment_date"), "u").isin(
                        ["6", "7"]), True)
                    .otherwise(False)) \
        .withColumn("processed_timestamp", F.current_timestamp())

    # Join with patient and physician data if available
    if patients_df is not None:
        transformed_df = transformed_df.join(
            patients_df.select("patient_id", "full_name").withColumnRenamed(
                "full_name", "patient_name"),
            on="patient_id",
            how="left"
        )

    if physicians_df is not None:
        transformed_df = transformed_df.join(
            physicians_df.select("physician_id", "full_name", "specialty").withColumnRenamed(
                "full_name", "physician_name"),
            on="physician_id",
            how="left"
        )

    return transformed_df


def load_appointments(df: DataFrame, target_path: str, table_name: str = "appointments") -> None:
    """
    Load transformed appointment data into the target.

    Args:
        df: DataFrame with transformed appointment data
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


def run_pipeline(spark: SparkSession, source_path: str, target_path: str,
                 patients_table: str = None, physicians_table: str = None,
                 table_name: str = "appointments") -> None:
    """
    Run the complete ETL/ELT pipeline for appointment data.

    Args:
        spark: SparkSession object
        source_path: Path to the source data
        target_path: Path to the target location
        patients_table: Optional name of patients table for data enrichment
        physicians_table: Optional name of physicians table for data enrichment
        table_name: Name of the table to create/update
    """
    # Extract
    appointments_df = extract_appointments(spark, source_path)

    # Get reference data if tables are provided
    patients_df = None
    if patients_table:
        patients_df = spark.table(patients_table)

    physicians_df = None
    if physicians_table:
        physicians_df = spark.table(physicians_table)

    # Transform
    transformed_df = transform_appointments(
        appointments_df, patients_df, physicians_df)

    # Load
    load_appointments(transformed_df, target_path, table_name)

    print(
        f"Appointments ETL completed: data loaded to {target_path} and registered as table {table_name}")
