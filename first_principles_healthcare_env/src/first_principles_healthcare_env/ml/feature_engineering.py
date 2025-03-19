"""
Feature engineering module for healthcare ML models.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, FloatType
from typing import List, Dict, Any, Optional


def extract_datetime_features(df: DataFrame, datetime_col: str = "appointment_time") -> DataFrame:
    """
    Extract datetime features from a timestamp column.

    Args:
        df: DataFrame with timestamp column
        datetime_col: Name of the timestamp column

    Returns:
        DataFrame: DataFrame with datetime features
    """
    return df.withColumn("hour_of_day", F.hour(F.col(datetime_col))) \
             .withColumn("day_of_week", F.date_format(F.col(datetime_col), "EEEE")) \
             .withColumn("day_of_month", F.dayofmonth(F.col(datetime_col))) \
             .withColumn("month", F.month(F.col(datetime_col))) \
             .withColumn("quarter", F.quarter(F.col(datetime_col))) \
             .withColumn("year", F.year(F.col(datetime_col))) \
             .withColumn("is_weekend",
                         F.when(F.date_format(F.col(datetime_col),
                                "u").isin(["6", "7"]), True)
                         .otherwise(False)) \
             .withColumn("season", F.when(F.month(F.col(datetime_col)).isin([12, 1, 2]), "Winter")
                         .when(F.month(F.col(datetime_col)).isin([3, 4, 5]), "Spring")
                         .when(F.month(F.col(datetime_col)).isin([6, 7, 8]), "Summer")
                         .otherwise("Fall"))


def calculate_appointment_lead_time(df: DataFrame,
                                    appointment_date_col: str = "appointment_date",
                                    created_at_col: str = "created_at") -> DataFrame:
    """
    Calculate the lead time in days between when an appointment was created and when it is scheduled.

    Args:
        df: DataFrame with appointment data
        appointment_date_col: Name of the appointment date column
        created_at_col: Name of the created_at column

    Returns:
        DataFrame: DataFrame with appointment lead time
    """
    return df.withColumn("appointment_lead_days",
                         F.datediff(F.col(appointment_date_col), F.col(created_at_col)).cast(IntegerType()))


def calculate_patient_history(df: DataFrame, patient_id_col: str = "patient_id",
                              appointment_date_col: str = "appointment_date",
                              no_show_col: str = "no_show") -> DataFrame:
    """
    Calculate patient history features like previous appointments and no-show rate.

    Args:
        df: DataFrame with appointment data
        patient_id_col: Name of the patient ID column
        appointment_date_col: Name of the appointment date column
        no_show_col: Name of the no-show column

    Returns:
        DataFrame: DataFrame with patient history features
    """
    # Create a window spec for patient history
    window_spec = Window.partitionBy(patient_id_col) \
                        .orderBy(appointment_date_col) \
                        .rowsBetween(Window.unboundedPreceding, -1)

    # Calculate previous appointments and no-shows
    return df.withColumn("previous_appointments", F.count("*").over(window_spec)) \
             .withColumn("previous_no_shows", F.sum(F.col(no_show_col).cast(IntegerType())).over(window_spec)) \
             .withColumn("no_show_rate",
                         F.when(F.col("previous_appointments") > 0,
                                (F.col("previous_no_shows") / F.col("previous_appointments")).cast(FloatType()))
                         .otherwise(0.0))


def calculate_patient_age(df: DataFrame,
                          dob_col: str = "date_of_birth",
                          as_of_date_col: str = "appointment_date") -> DataFrame:
    """
    Calculate patient age as of a certain date.

    Args:
        df: DataFrame with patient data
        dob_col: Name of the date of birth column
        as_of_date_col: Name of the reference date column

    Returns:
        DataFrame: DataFrame with patient age
    """
    return df.withColumn("patient_age",
                         F.floor(F.datediff(F.col(as_of_date_col), F.col(dob_col)) / 365.25).cast(IntegerType()))


def generate_features(appointments_df: DataFrame,
                      patients_df: DataFrame,
                      physicians_df: DataFrame) -> DataFrame:
    """
    Generate features for appointment no-show prediction.

    Args:
        appointments_df: DataFrame with appointment data
        patients_df: DataFrame with patient data
        physicians_df: DataFrame with physician data

    Returns:
        DataFrame: DataFrame with features for ML
    """
    # Join appointments with patients and physicians
    df = appointments_df.join(
        patients_df,
        on="patient_id",
        how="left"
    ).join(
        physicians_df.select("physician_id", "specialty"),
        on="physician_id",
        how="left"
    )

    # Extract features
    df = extract_datetime_features(df)
    df = calculate_appointment_lead_time(df)
    df = calculate_patient_history(df)
    df = calculate_patient_age(df)

    # Create final feature set
    feature_cols = [
        "appointment_id",
        "patient_id",
        "physician_id",
        "appointment_date",
        "appointment_time",
        "patient_age",
        "gender",
        "specialty",
        "appointment_lead_days",
        "previous_no_shows",
        "previous_appointments",
        "no_show_rate",
        "day_of_week",
        "is_weekend",
        "hour_of_day",
        "season",
        "no_show"
    ]

    # Select and rename columns
    return df.select([
        F.col(c) if c in df.columns else F.lit(None).alias(c)
        for c in feature_cols
    ]).withColumnRenamed("gender", "patient_gender") \
      .withColumnRenamed("hour_of_day", "appointment_hour") \
      .withColumnRenamed("specialty", "physician_specialty")


def create_feature_table(spark: SparkSession,
                         features_df: DataFrame,
                         target_table: str = "appointment_features",
                         target_path: str = None) -> None:
    """
    Create a feature table for ML.

    Args:
        spark: SparkSession object
        features_df: DataFrame with features
        target_table: Name of the target table
        target_path: Path to save the Delta table
    """
    if target_path is None:
        target_path = f"dbfs:/FileStore/tables/{target_table}"

    # Write to Delta
    features_df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(target_path)

    # Create or replace table
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {target_table} USING DELTA LOCATION '{target_path}'")

    print(
        f"Feature table created at {target_path} and registered as {target_table}")
