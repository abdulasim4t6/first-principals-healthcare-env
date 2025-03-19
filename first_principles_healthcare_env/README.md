# First Principles Healthcare Environment

A scalable and robust healthcare data processing and ML platform built on Databricks and deployed in AWS.

## Project Overview

This project provides an ETL/ELT data pipeline and machine learning workflows for healthcare data, focusing on patients, physicians, and appointments. It's designed to help healthcare providers predict appointment no-shows and optimize their scheduling processes.

### Key Features

- **ETL/ELT Pipelines**: Efficient data processing for patient, physician, and appointment data
- **ML Capabilities**: Predicting appointment no-shows with machine learning models
- **Scheduled Jobs**: Automated ETL and ML training workflows
- **Scalability**: Built on Databricks with Apache Spark for large-scale data processing
- **AWS Integration**: Designed to work in the AWS cloud environment

## Project Structure

```
first_principles_healthcare_env/
│
├── .databricks/           # Databricks configuration
├── src/                   # Source code
│   └── first_principles_healthcare_env/
│       ├── config/        # Configuration settings
│       ├── data_models/   # Data schemas and models
│       ├── etl/           # ETL/ELT processing modules
│       ├── jobs/          # Scheduled job definitions
│       ├── ml/            # Machine learning modules
│       └── utils/         # Utility functions
│
├── notebooks/             # Databricks notebooks
│   ├── etl/               # ETL notebooks
│   └── ml/                # ML notebooks
│
├── resources/             # Databricks resources (jobs, pipelines)
├── tests/                 # Test modules
├── data/                  # Sample and test data
│   ├── patients/
│   ├── physicians/
│   └── appointments/
│
└── setup.py               # Package setup script
```

## Data Model

The project works with the following core data entities:

### Patients
Patient demographic and insurance information.

### Physicians
Physician details including specialties and departments.

### Appointments
Appointment records linking patients with physicians, including status and no-show flags.

## Prerequisites

- Databricks workspace (AWS deployment)
- Python 3.8+
- Apache Spark 3.3+
- Delta Lake

## Setup and Deployment

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/first-principles-healthcare-env.git
   cd first-principles-healthcare-env
   ```

2. Install development dependencies:
   ```
   pip install -r requirements-dev.txt
   ```

3. Configure Databricks CLI:
   ```
   databricks configure
   ```

4. Deploy to your Databricks workspace:
   ```
   databricks bundle deploy --target dev
   ```

## Workflows

### ETL Pipeline

The ETL pipeline processes the healthcare data in the following sequence:
1. Extract and transform patient data
2. Extract and transform physician data
3. Extract, transform, and enrich appointment data

The pipeline runs daily to ensure the data is always up-to-date.

### ML Workflow

The ML workflow:
1. Generates features from processed data
2. Trains a model to predict appointment no-shows
3. Evaluates model performance
4. Makes predictions on upcoming appointments
5. Logs models and metrics to MLflow

## Development

For local development:

1. Install development requirements:
   ```
   pip install -r requirements-dev.txt
   ```

2. Setup Databricks Connect for local development:
   ```
   databricks configure --profile dev
   databricks-connect configure
   ```

3. Run tests:
   ```
   pytest tests/
   ```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Databricks for their excellent platform
- The Apache Spark community
