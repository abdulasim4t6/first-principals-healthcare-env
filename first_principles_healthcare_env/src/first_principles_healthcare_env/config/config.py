"""
Configuration settings for the healthcare data project.
"""
import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DatabaseConfig:
    """Database configuration."""
    catalog: str = "main"
    schema: str = "healthcare"


@dataclass
class StorageConfig:
    """Storage configuration."""
    base_path: str = "dbfs:/FileStore/tables/first_principles_healthcare"
    patients_path: str = "dbfs:/FileStore/tables/first_principles_healthcare/patients"
    physicians_path: str = "dbfs:/FileStore/tables/first_principles_healthcare/physicians"
    appointments_path: str = "dbfs:/FileStore/tables/first_principles_healthcare/appointments"
    ml_models_path: str = "dbfs:/FileStore/tables/first_principles_healthcare/ml_models"


@dataclass
class TableConfig:
    """Table configuration."""
    patients_table: str = "patients"
    physicians_table: str = "physicians"
    appointments_table: str = "appointments"
    features_table: str = "appointment_features"


@dataclass
class SparkConfig:
    """Spark configuration."""
    app_name: str = "Healthcare Data Pipeline"
    log_level: str = "WARN"


@dataclass
class MLConfig:
    """Machine Learning configuration."""
    target_col: str = "no_show"
    features: list = None
    test_size: float = 0.2
    random_state: int = 42
    mlflow_experiment_name: str = "healthcare_no_show_prediction"

    def __post_init__(self):
        if self.features is None:
            self.features = [
                "patient_age",
                "appointment_lead_days",
                "previous_no_shows",
                "is_weekend",
                "appointment_hour",
                "season"
            ]


@dataclass
class Config:
    """Main configuration class."""
    env: str
    db: DatabaseConfig = None
    storage: StorageConfig = None
    tables: TableConfig = None
    spark: SparkConfig = None
    ml: MLConfig = None

    def __post_init__(self):
        self.db = self.db or DatabaseConfig()
        self.storage = self.storage or StorageConfig()
        self.tables = self.tables or TableConfig()
        self.spark = self.spark or SparkConfig()
        self.ml = self.ml or MLConfig()


def get_config(env: str = None) -> Config:
    """
    Get the configuration for the specified environment.

    Args:
        env: Environment name (dev, prod, etc.)

    Returns:
        Config: Configuration for the specified environment
    """
    if env is None:
        env = os.environ.get("ENV", "dev")

    if env == "dev":
        return Config(
            env=env,
            db=DatabaseConfig(schema="healthcare_dev"),
            storage=StorageConfig(
                base_path="dbfs:/FileStore/tables/first_principles_healthcare_dev"),
            spark=SparkConfig(log_level="INFO")
        )
    elif env == "prod":
        return Config(
            env=env,
            db=DatabaseConfig(schema="healthcare"),
            storage=StorageConfig(
                base_path="dbfs:/FileStore/tables/first_principles_healthcare"),
            spark=SparkConfig(log_level="WARN")
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


# Default configuration
config = get_config()
