"""
Daily ETL job for healthcare data.
"""
from pyspark.sql import SparkSession
from datetime import datetime, timedelta
import logging

from first_principles_healthcare_env.config.config import get_config
from first_principles_healthcare_env.etl.patients import run_pipeline as run_patients_pipeline
from first_principles_healthcare_env.etl.physicians import run_pipeline as run_physicians_pipeline
from first_principles_healthcare_env.etl.appointments import run_pipeline as run_appointments_pipeline
from first_principles_healthcare_env.utils.spark_utils import get_spark_session, create_database


def main(env: str = "dev",
         patients_source: str = None,
         physicians_source: str = None,
         appointments_source: str = None):
    """
    Main job function to run the daily ETL pipeline.

    Args:
        env: Environment (dev, prod, etc.)
        patients_source: Source path for patients data
        physicians_source: Source path for physicians data
        appointments_source: Source path for appointments data
    """
    # Get configuration for the environment
    config = get_config(env)

    # Initialize Spark session
    spark = get_spark_session(config.spark.app_name)
    spark.sparkContext.setLogLevel(config.spark.log_level)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create database if it doesn't exist
    create_database(spark, config.db.schema)

    # Use the database
    spark.sql(f"USE {config.db.schema}")

    # Set default source paths if not provided
    if patients_source is None:
        patients_source = "dbfs:/data/patients"

    if physicians_source is None:
        physicians_source = "dbfs:/data/physicians"

    if appointments_source is None:
        appointments_source = "dbfs:/data/appointments"

    # Log start of job
    job_start_time = datetime.now()
    logger.info(f"Starting daily ETL job at {job_start_time}")

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
            f"Daily ETL job completed at {job_end_time}. Duration: {job_duration:.2f} seconds")

    except Exception as e:
        # Log error and raise
        logger.error(f"Error in daily ETL job: {str(e)}")
        raise


if __name__ == "__main__":
    # This will be used when the script is run directly
    main()
