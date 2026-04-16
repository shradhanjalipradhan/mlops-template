# MLOps Template - Production ML Pipeline with Drift Monitoring

[![CI](https://github.com/shradhanjalipradhan/mlops-template/actions/workflows/ci.yml/badge.svg)](https://github.com/shradhanjalipradhan/mlops-template/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

A plug-and-play, fully local MLOps platform demonstrating production-grade ML engineering: model training, experiment tracking, real-time inference, drift detection, automated retraining, CI/CD, and observability.

## What This Template Includes

This repository implements an end-to-end ML lifecycle for a customer churn prediction use case:

| Layer | What it does |
|---|---|
| Data | Synthetic data generation, validation, and train/validation/test split |
| Training | XGBoost training with MLflow tracking and model registration |
| Serving | FastAPI inference API with Prometheus metrics and prediction logging |
| Drift detection | Scheduled drift checks against a reference dataset |
| Retraining | Airflow DAGs for ingestion, training, drift checks, and retraining |
| Observability | Grafana dashboards backed by Prometheus |
| CI/CD | GitHub Actions for linting, tests, image builds, and image publishing |

## Architecture

```text
Raw Data -> [Airflow: ingest_validate_dag]
              |
              v
          [Airflow: train_register_dag]
              |
              v
          MLflow Registry (Production model)
              |
              v
          FastAPI /predict -> SQLite prediction log
              |                    |
              v                    v
          Prometheus         [Airflow: drift_check_dag]
                                   |
                                   v
                           Drift report and alerts
                                   |
                                   v
                           [Airflow: retrain_dag]
```

## Quick Start

Prerequisites:

- Docker Desktop or Docker Engine with Compose
- Python 3.11+
- Git
- Bash for `scripts/bootstrap.sh` if you are on Windows (`Git Bash` or `WSL` is fine)

```bash
git clone https://github.com/shradhanjalipradhan/mlops-template.git
cd mlops-template
bash scripts/bootstrap.sh
make demo
```

The bootstrap script creates `.env`, installs Python dependencies, starts the local stack, generates synthetic data, and seeds the first model in MLflow.

## Service Map

| Service | URL | Credentials |
|---|---|---|
| FastAPI inference | http://localhost:8000/docs | none |
| MLflow UI | http://localhost:5000 | none |
| Airflow | http://localhost:8080 | `admin` / `admin` |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| Prometheus | http://localhost:9090 | none |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |

## Make Targets

```bash
make bootstrap         # first-time setup
make up                # start all services
make down              # stop and remove all services and volumes
make train             # trigger training DAG in Airflow
make test              # run unit tests
make test-integration  # run integration tests
make demo              # end-to-end demo flow
make load-test         # send 500 prediction requests
make trigger-drift     # trigger the drift check DAG
```

## Repository Structure

```text
mlops-template/
|-- .github/workflows/          # CI and CD workflows
|-- configs/                    # model, training, and drift config
|-- dags/                       # Airflow DAG definitions
|-- data/                       # generated datasets and reference data
|-- infra/
|   |-- docker-compose.yml      # full local stack
|   |-- docker-compose.ci.yml   # lightweight CI stack
|   |-- Dockerfile.serving
|   |-- Dockerfile.training
|   |-- Dockerfile.monitoring
|   |-- prometheus/
|   |-- grafana/
|   `-- alertmanager/
|-- scripts/
|   |-- bootstrap.sh
|   |-- generate_data.py
|   |-- seed_mlflow.py
|   |-- inject_drift.py
|   `-- load_test.py
|-- src/
|   |-- data/
|   |-- features/
|   |-- monitoring/
|   |-- serving/
|   |-- training/
|   `-- utils/
`-- tests/
    |-- integration/
    |-- unit/
    `-- conftest.py
```

## Demo Walkthrough

1. Start the stack:

   ```bash
   make up
   ```

2. Generate inference traffic:

   ```bash
   make load-test
   ```

3. Inject drift and run a drift check:

   ```bash
   python scripts/inject_drift.py
   make trigger-drift
   ```

4. Inspect outputs:

   - Grafana: http://localhost:3000
   - MLflow: http://localhost:5000
   - Airflow: http://localhost:8080

## Configuration

All project configuration lives in `configs/`:

- `model_config.yaml` for model hyperparameters and feature definitions
- `training_config.yaml` for training thresholds and paths
- `drift_config.yaml` for drift thresholds and monitoring settings

Environment variables are documented in `.env.example`.

## Testing

```bash
make test
pytest tests/unit/ --cov=src --cov-report=html
make test-integration
```

`make test-integration` uses `infra/docker-compose.ci.yml` to start a smaller stack for integration tests.

## CI/CD

CI runs on pushes and pull requests and includes:

1. Ruff linting and mypy type checking
2. Unit tests with coverage
3. Docker image builds for serving and training
4. Integration tests against the CI compose stack

CD runs on pushes to `main` and publishes three Docker images:

- `mlops-serving`
- `mlops-training`
- `mlops-monitoring`

To enable Docker Hub publishing, configure these repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## Extending The Template

- Swap `XGBClassifier` in `src/training/train.py` for another sklearn-compatible estimator.
- Add new features in `configs/model_config.yaml`, `src/features/pipeline.py`, `src/data/generator.py`, and `src/serving/schemas.py`.
- Replace the synthetic generator in `src/data/generator.py` with your real data source.
- Extend `src/monitoring/alerting.py` if you want to add Slack or other notification targets.

## Stack

| Tool | Role |
|---|---|
| Python 3.11 | Application runtime |
| XGBoost | Model training |
| scikit-learn | Feature engineering |
| MLflow | Experiment tracking and registry |
| Apache Airflow | Workflow orchestration |
| FastAPI | Inference API |
| Evidently | Drift detection |
| Prometheus | Metrics collection |
| Grafana | Dashboards |
| Alertmanager | Alert routing |
| MinIO | S3-compatible artifact storage |
| PostgreSQL | Metadata store |
| SQLite | Prediction logging |
| Docker Compose | Local deployment |
| GitHub Actions | CI/CD |
