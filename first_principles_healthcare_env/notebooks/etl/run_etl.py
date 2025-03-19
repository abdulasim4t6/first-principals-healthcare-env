# Databricks notebook source
# MAGIC %md
# MAGIC # Healthcare ETL Pipeline
# MAGIC
# MAGIC This notebook runs the complete ETL pipeline for the healthcare data, processing:
# MAGIC - Patient data
# MAGIC - Physician data
# MAGIC - Appointment data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Import required libraries
from first_principles_healthcare_env.config.config import get_config
from first_principles_healthcare_env.etl.patients import run_pipeline as run_patients_pipeline
from first_principles_healthcare_env.etl.physicians import run_pipeline as run_physicians_pipeline
from first_principles_healthcare_env.etl.appointments import run_pipeline as run_appointments_pipeline
from first_principles_healthcare_env.utils.spark_utils import get_spark_session, create_database
from pyspark.sql import SparkSession
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ETL Pipeline")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

# Set parameters
dbutils.widgets.dropdown("env", "dev", ["dev", "prod"], "Environment")
env = dbutils.widgets.get("env")

# Set source paths
data_root = "dbfs:/data"
patients_source = f"{data_root}/patients"
physicians_source = f"{data_root}/physicians"
appointments_source = f"{data_root}/appointments"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Configuration and Spark Session

# COMMAND ----------

# Get configuration for the environment
config = get_config(env)
print(f"Running in {env} environment with schema: {config.db.schema}")

# Initialize Spark session
spark = get_spark_session(config.spark.app_name)
spark.sparkContext.setLogLevel(config.spark.log_level)

# Create database if it doesn't exist
create_database(spark, config.db.schema)

# Use the database
spark.sql(f"USE {config.db.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run ETL Pipeline

# COMMAND ----------

# Log start of job
job_start_time = datetime.now()
logger.info(f"Starting ETL pipeline at {job_start_time}")

try:
    # Run patients ETL
    logger.info("Starting patients ETL pipeline")
    run_patients_pipeline(
        spark=spark,
        source_path=patients_source,
        target_path=config.storage.patients_path,
        table_name=f"{config.db.schema}.{config.tables.patients_table}"
    )
    logger.info("Completed patients ETL pipeline")

    # Run physicians ETL
    logger.info("Starting physicians ETL pipeline")
    run_physicians_pipeline(
        spark=spark,
        source_path=physicians_source,
        target_path=config.storage.physicians_path,
        table_name=f"{config.db.schema}.{config.tables.physicians_table}"
    )
    logger.info("Completed physicians ETL pipeline")

    # Run appointments ETL
    logger.info("Starting appointments ETL pipeline")
    run_appointments_pipeline(
        spark=spark,
        source_path=appointments_source,
        target_path=config.storage.appointments_path,
        patients_table=f"{config.db.schema}.{config.tables.patients_table}",
        physicians_table=f"{config.db.schema}.{config.tables.physicians_table}",
        table_name=f"{config.db.schema}.{config.tables.appointments_table}"
    )
    logger.info("Completed appointments ETL pipeline")

    # Log job completion
    job_end_time = datetime.now()
    job_duration = (job_end_time - job_start_time).total_seconds()
    logger.info(
        f"ETL pipeline completed at {job_end_time}. Duration: {job_duration:.2f} seconds")

except Exception as e:
    # Log error
    logger.error(f"Error in ETL pipeline: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Results

# COMMAND ----------

# Display the tables
print("\nPatients Table:")
display(spark.table(
    f"{config.db.schema}.{config.tables.patients_table}").limit(5))

print("\nPhysicians Table:")
display(spark.table(
    f"{config.db.schema}.{config.tables.physicians_table}").limit(5))

print("\nAppointments Table:")
display(spark.table(
    f"{config.db.schema}.{config.tables.appointments_table}").limit(5))

# COMMAND ----------

logger.info("ETL notebook execution completed successfully")
