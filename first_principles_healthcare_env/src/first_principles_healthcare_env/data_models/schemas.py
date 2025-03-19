"""
Data schemas for the healthcare data project.
"""
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DateType, TimestampType, BooleanType, FloatType
)

# Patient Schema
patient_schema = StructType([
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

# Physician Schema
physician_schema = StructType([
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

# Appointment Schema
appointment_schema = StructType([
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

# Feature Schema for ML
feature_schema = StructType([
    StructField("appointment_id", StringType(), False),
    StructField("patient_id", StringType(), False),
    StructField("physician_id", StringType(), False),
    StructField("appointment_date", DateType(), True),
    StructField("appointment_time", TimestampType(), True),
    StructField("patient_age", IntegerType(), True),
    StructField("patient_gender", StringType(), True),
    StructField("physician_specialty", StringType(), True),
    StructField("appointment_lead_days", IntegerType(), True),
    StructField("previous_no_shows", IntegerType(), True),
    StructField("previous_appointments", IntegerType(), True),
    StructField("no_show_rate", FloatType(), True),
    StructField("day_of_week", StringType(), True),
    StructField("is_weekend", BooleanType(), True),
    StructField("appointment_hour", IntegerType(), True),
    StructField("season", StringType(), True),
    StructField("no_show", BooleanType(), True)
])

# Prediction Schema
prediction_schema = StructType([
    StructField("appointment_id", StringType(), False),
    StructField("patient_id", StringType(), False),
    StructField("appointment_date", DateType(), True),
    StructField("no_show_probability", FloatType(), True),
    StructField("prediction", BooleanType(), True),
    StructField("model_version", StringType(), True),
    StructField("prediction_timestamp", TimestampType(), True)
])
