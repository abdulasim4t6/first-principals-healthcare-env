"""
Machine learning models for healthcare predictions.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline, PipelineModel
from typing import List, Dict, Any, Optional, Tuple
import mlflow
import mlflow.spark
from datetime import datetime


def prepare_training_data(df: DataFrame,
                          categorical_cols: List[str] = None,
                          numerical_cols: List[str] = None,
                          target_col: str = "no_show") -> Tuple[Pipeline, DataFrame, List[str]]:
    """
    Prepare training data for ML.

    Args:
        df: DataFrame with features
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        target_col: Name of the target column

    Returns:
        Tuple[Pipeline, DataFrame, List[str]]: Feature preparation pipeline, prepared DataFrame, feature names
    """
    if categorical_cols is None:
        categorical_cols = [
            "patient_gender",
            "physician_specialty",
            "day_of_week",
            "season"
        ]

    if numerical_cols is None:
        numerical_cols = [
            "patient_age",
            "appointment_lead_days",
            "previous_no_shows",
            "previous_appointments",
            "no_show_rate",
            "appointment_hour"
        ]

    # Handle missing values
    df_clean = df
    for col in numerical_cols:
        df_clean = df_clean.withColumn(col, F.coalesce(F.col(col), F.lit(0)))

    # Create string indexers for categorical columns
    indexers = [
        StringIndexer(
            inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]

    # Create one-hot encoders for indexed columns
    encoders = [
        OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec")
        for col in categorical_cols
    ]

    # Create vector assembler for all features
    assembler_inputs = [
        f"{col}_vec" for col in categorical_cols] + numerical_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

    # Create pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    # Fit the pipeline to the data
    pipeline_model = pipeline.fit(df_clean)

    # Transform the data
    df_transformed = pipeline_model.transform(df_clean)

    return pipeline, df_transformed, assembler_inputs


def train_random_forest(df: DataFrame,
                        feature_col: str = "features",
                        target_col: str = "no_show",
                        max_depth: int = 10,
                        num_trees: int = 100) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        df: DataFrame with features
        feature_col: Name of the feature column
        target_col: Name of the target column
        max_depth: Maximum depth of trees
        num_trees: Number of trees

    Returns:
        RandomForestClassifier: Trained Random Forest classifier
    """
    # Create classifier
    rf = RandomForestClassifier(
        featuresCol=feature_col,
        labelCol=target_col,
        maxDepth=max_depth,
        numTrees=num_trees,
        seed=42
    )

    # Train the model
    return rf.fit(df)


def train_gbt(df: DataFrame,
              feature_col: str = "features",
              target_col: str = "no_show",
              max_depth: int = 5,
              max_iter: int = 50) -> GBTClassifier:
    """
    Train a Gradient-Boosted Tree classifier.

    Args:
        df: DataFrame with features
        feature_col: Name of the feature column
        target_col: Name of the target column
        max_depth: Maximum depth of trees
        max_iter: Maximum number of iterations

    Returns:
        GBTClassifier: Trained GBT classifier
    """
    # Create classifier
    gbt = GBTClassifier(
        featuresCol=feature_col,
        labelCol=target_col,
        maxDepth=max_depth,
        maxIter=max_iter,
        seed=42
    )

    # Train the model
    return gbt.fit(df)


def train_with_mlflow(df: DataFrame,
                      categorical_cols: List[str] = None,
                      numerical_cols: List[str] = None,
                      target_col: str = "no_show",
                      experiment_name: str = "healthcare_no_show_prediction") -> str:
    """
    Train a model with MLflow tracking.

    Args:
        df: DataFrame with features
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        target_col: Name of the target column
        experiment_name: Name of the MLflow experiment

    Returns:
        str: ID of the MLflow run
    """
    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Prepare data
    pipeline, train_transformed, feature_names = prepare_training_data(
        train_df, categorical_cols, numerical_cols, target_col
    )

    # Apply same transformations to test data
    test_transformed = pipeline.fit(test_df).transform(test_df)

    # Create evaluators
    evaluator_auc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol=target_col,
        metricName="areaUnderROC"
    )

    evaluator_acc = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol=target_col,
        metricName="accuracy"
    )

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("num_trees", 100)
        mlflow.log_param("train_size", train_df.count())
        mlflow.log_param("test_size", test_df.count())

        # Train model
        rf_model = train_random_forest(train_transformed)

        # Make predictions
        predictions = rf_model.transform(test_transformed)

        # Evaluate model
        auc = evaluator_auc.evaluate(predictions)
        accuracy = evaluator_acc.evaluate(predictions)

        # Log metrics
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", accuracy)

        # Log feature importance
        for i, feature in enumerate(rf_model.featureImportances):
            if i < len(feature_names):
                mlflow.log_metric(
                    f"importance_{feature_names[i]}", float(feature))

        # Create pipeline with feature processing and model
        model_pipeline = Pipeline(stages=[pipeline, rf_model])

        # Log model
        mlflow.spark.log_model(model_pipeline, "model")

        print(f"Model trained with AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    return run_id


def load_model(model_uri: str) -> PipelineModel:
    """
    Load a trained model from MLflow.

    Args:
        model_uri: URI of the model in MLflow

    Returns:
        PipelineModel: Loaded model
    """
    return mlflow.spark.load_model(model_uri)


def predict_no_shows(model: PipelineModel, df: DataFrame) -> DataFrame:
    """
    Predict appointment no-shows.

    Args:
        model: Trained model
        df: DataFrame with features

    Returns:
        DataFrame: DataFrame with predictions
    """
    # Make predictions
    predictions = model.transform(df)

    # Add timestamp
    predictions = predictions.withColumn(
        "prediction_timestamp", F.current_timestamp())

    # Select relevant columns
    return predictions.select(
        "appointment_id",
        "patient_id",
        "appointment_date",
        "probability",
        "prediction",
        "prediction_timestamp"
    ).withColumn(
        "no_show_probability",
        F.element_at(F.col("probability"), 2)
    ).drop("probability")
