"""
Machine learning training job for healthcare data.
"""
from pyspark.sql import SparkSession
from datetime import datetime
import logging
import mlflow
import mlflow.spark

from first_principles_healthcare_env.config.config import get_config
from first_principles_healthcare_env.utils.spark_utils import get_spark_session
from first_principles_healthcare_env.ml.feature_engineering import generate_features, create_feature_table
from first_principles_healthcare_env.ml.models import train_with_mlflow, load_model, predict_no_shows


def main(env: str = "dev"):
    """
    Main job function to run the ML training pipeline.

    Args:
        env: Environment (dev, prod, etc.)
    """
    # Get configuration for the environment
    config = get_config(env)

    # Initialize Spark session
    spark = get_spark_session(config.spark.app_name)
    spark.sparkContext.setLogLevel(config.spark.log_level)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Use the database
    spark.sql(f"USE {config.db.schema}")

    # Log start of job
    job_start_time = datetime.now()
    logger.info(f"Starting ML training job at {job_start_time}")

    try:
        # Load data
        logger.info("Loading data from tables")
        appointments_df = spark.table(
            f"{config.db.schema}.{config.tables.appointments_table}")
        patients_df = spark.table(
            f"{config.db.schema}.{config.tables.patients_table}")
        physicians_df = spark.table(
            f"{config.db.schema}.{config.tables.physicians_table}")

        # Generate features
        logger.info("Generating features for ML")
        features_df = generate_features(
            appointments_df, patients_df, physicians_df)

        # Create feature table
        logger.info("Creating feature table")
        create_feature_table(
            spark=spark,
            features_df=features_df,
            target_table=f"{config.db.schema}.{config.tables.features_table}",
            target_path=f"{config.storage.base_path}/features"
        )

        # Train model
        logger.info("Training ML model")
        mlflow.set_experiment(config.ml.mlflow_experiment_name)

        run_id = train_with_mlflow(
            df=features_df,
            categorical_cols=None,  # Use default from function
            numerical_cols=None,  # Use default from function
            target_col=config.ml.target_col,
            experiment_name=config.ml.mlflow_experiment_name
        )

        # Load the trained model
        logger.info(f"Loading trained model from run {run_id}")
        model_uri = f"runs:/{run_id}/model"
        model = load_model(model_uri)

        # Make predictions on upcoming appointments
        logger.info("Making predictions on upcoming appointments")
        upcoming_appointments = appointments_df.filter(
            "appointment_date > current_date() AND status = 'Scheduled'"
        )

        if upcoming_appointments.count() > 0:
            # Generate features for upcoming appointments
            upcoming_features = generate_features(
                upcoming_appointments, patients_df, physicians_df)

            # Make predictions
            predictions = predict_no_shows(model, upcoming_features)

            # Save predictions
            predictions.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(f"{config.storage.base_path}/predictions")

            # Create or replace table
            spark.sql(f"""
                CREATE TABLE IF NOT EXISTS {config.db.schema}.appointment_predictions 
                USING DELTA 
                LOCATION '{config.storage.base_path}/predictions'
            """)

            logger.info(
                f"Generated {predictions.count()} predictions for upcoming appointments")
        else:
            logger.info("No upcoming appointments to predict")

        # Log model URI for downstream use
        logger.info(f"Model URI: {model_uri}")

        # Log job completion
        job_end_time = datetime.now()
        job_duration = (job_end_time - job_start_time).total_seconds()
        logger.info(
            f"ML training job completed at {job_end_time}. Duration: {job_duration:.2f} seconds")

    except Exception as e:
        # Log error and raise
        logger.error(f"Error in ML training job: {str(e)}")
        raise


if __name__ == "__main__":
    # This will be used when the script is run directly
    main()
