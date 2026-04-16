# MLOps Template — Production ML Pipeline with Drift Monitoring

[![CI](https://github.com/YOUR_USERNAME/mlops-template/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/mlops-template/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A plug-and-play, fully local MLOps platform demonstrating production-grade ML engineering:
model training, experiment tracking, real-time inference, drift detection, automated retraining,
CI/CD, and full observability — **zero paid dependencies, runs on your laptop**.

---

## What is this?

This template implements the complete ML lifecycle for a **customer churn prediction** use case:

| Layer | What it does |
|---|---|
| Data | Synthetic data generator, schema validation, train/val/test split |
| Training | XGBoost + 5-fold CV, logged to MLflow, conditional promotion |
| Serving | FastAPI inference API with Prometheus metrics and prediction logging |
| Drift detection | Evidently AI compares live predictions to reference distribution hourly |
| Retraining | Alertmanager triggers Airflow DAG automatically when drift threshold is exceeded |
| Observability | Grafana dashboards: RPS, latency percentiles, drift score, label distribution |
| CI/CD | GitHub Actions: lint → unit tests → integration tests → Docker push |

---

## Architecture

```
Raw Data → [Airflow: ingest_validate_dag]
               ↓
           [Airflow: train_register_dag]
               ↓
           MLflow Registry (Production model)
               ↓
           FastAPI /predict → SQLite prediction log
               ↓                    ↓
           Prometheus          [Airflow: drift_check_dag (hourly)]
               ↓                    ↓
           Grafana             Evidently drift score
                                    ↓
                               drift > threshold?
                                    ↓ YES
                               Alertmanager → Airflow REST API
                                    ↓
                               [Airflow: retrain_dag] ──┐
                                                        ↓
                               ingest_validate → train_register
```

---

## Quick start

**Prerequisites:** Docker Desktop (or Docker Engine + Compose), Python 3.11+, Git

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/mlops-template.git
cd mlops-template

# 2. Bootstrap everything (creates .env, starts stack, trains first model)
bash scripts/bootstrap.sh

# 3. Run the full demo
make demo
```

That's it. The bootstrap script handles everything else.

---

## Service map

| Service | URL | Credentials |
|---|---|---|
| FastAPI inference | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| MinIO console | http://localhost:9001 | minioadmin / minioadmin |

---

## Available make commands

```bash
make up              # start all services
make down            # stop and remove all services + volumes
make train           # trigger training DAG in Airflow
make test            # run unit tests
make test-integration  # run integration tests (stack must be running)
make demo            # full demo: generate data → train → predict → inject drift
make load-test       # send 500 prediction requests
make trigger-drift   # manually trigger drift check DAG
```

---

## Repository structure

```
mlops-template/
├── .github/workflows/      # CI and CD GitHub Actions
├── configs/                # model, training, and drift configuration YAML
├── dags/                   # Airflow DAGs (ingest, train, drift, retrain)
├── data/                   # raw, processed, reference datasets (gitignored)
├── infra/
│   ├── docker-compose.yml      # full local stack (7 services)
│   ├── docker-compose.ci.yml   # lightweight CI stack
│   ├── Dockerfile.serving      # FastAPI inference image
│   ├── Dockerfile.training     # training image
│   ├── Dockerfile.monitoring   # drift detector image
│   ├── prometheus/             # scrape config + alert rules
│   ├── grafana/                # provisioned datasource + dashboard JSON
│   └── alertmanager/           # alert routing config
├── scripts/
│   ├── bootstrap.sh        # one-command setup
│   ├── generate_data.py    # synthetic data generator
│   ├── seed_mlflow.py      # train + register first model
│   ├── inject_drift.py     # inject drifted predictions for demo
│   └── load_test.py        # send N prediction requests
├── src/
│   ├── data/               # ingestion, validation, generator
│   ├── features/           # sklearn feature pipeline
│   ├── training/           # train, evaluate, register
│   ├── serving/            # FastAPI app, predictor, logger, metrics
│   ├── monitoring/         # Evidently drift detector, alerting
│   └── utils/              # config, logging, storage
└── tests/
    ├── unit/               # data, features, serving, drift, training tests
    ├── integration/        # live API + training pipeline tests
    └── conftest.py         # shared fixtures
```

---

## Demo walkthrough

### 1. Start the stack and open Grafana
```bash
make up
# Open http://localhost:3000 → MLOps — Churn Model dashboard
```

### 2. Generate traffic
```bash
make load-test
# Sends 500 prediction requests — watch RPS and latency panels update
```

### 3. Inject drift and trigger the retraining loop
```bash
python scripts/inject_drift.py   # shifts monthly_charge + tenure distributions
make trigger-drift               # run drift check immediately
# Watch Grafana: drift score panel climbs above 0.3
# Alertmanager fires → retrain_dag triggered → new model registered
```

### 4. Verify new model in MLflow
Open http://localhost:5000 → Experiments → churn_prediction → see the new run promoted to Production.

---

## Configuration

All configuration lives in `configs/`:

- `model_config.yaml` — hyperparameters and feature list
- `training_config.yaml` — min_auc threshold, CV folds, paths
- `drift_config.yaml` — drift threshold (default 0.3), window size, alert settings

All environment variables are documented in `.env.example`.

---

## Running tests

```bash
# Unit tests only (no Docker required)
make test

# With coverage
pytest tests/unit/ --cov=src --cov-report=html

# Integration tests (stack must be running)
make test-integration
```

---

## CI/CD

Every PR runs:
1. `ruff` linting + `mypy` type checking
2. Unit tests with coverage
3. Docker image build (serving + training)
4. Integration tests against a lightweight `docker-compose.ci.yml` stack

Merging to `main` builds and pushes all three Docker images to Docker Hub.

To enable CD, add these GitHub repository secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## Extending this template

**Add a new model type:** Replace `XGBClassifier` in `src/training/train.py` with any sklearn-compatible estimator. The rest of the pipeline is model-agnostic.

**Add a new feature:** Add the column name to `configs/model_config.yaml` and `src/features/pipeline.py`. Update `src/data/generator.py` and `src/serving/schemas.py` to include the new field.

**Change the use case:** Swap out `src/data/generator.py` for your real data source. Adjust the schema in `src/data/validation.py`. Everything downstream (training, serving, drift detection) adapts automatically.

**Add a Slack alert:** In `src/monitoring/alerting.py`, add a `requests.post` to your Slack webhook URL alongside the existing Airflow trigger.

---

## Stack

| Tool | Role | Why |
|---|---|---|
| Python 3.11 | Runtime | Ecosystem breadth |
| XGBoost | Model | Fast, interpretable, handles tabular data |
| scikit-learn | Feature pipeline | Industry-standard transforms |
| MLflow | Experiment tracking + registry | Best free all-in-one |
| Apache Airflow 2 | Orchestration | De-facto open-source scheduler |
| FastAPI | Inference API | Fastest Python API framework |
| Evidently AI | Drift detection | Purpose-built for ML monitoring |
| Prometheus | Metrics collection | Industry-standard, free |
| Grafana | Dashboards | Best free dashboard tool |
| Alertmanager | Alert routing | Native Prometheus integration |
| MinIO | Artifact storage | S3-compatible, free, local-first |
| PostgreSQL | Metadata store | Reliable, used by both Airflow and MLflow |
| SQLite | Prediction log | Zero-config, embedded, fast for writes |
| Docker Compose | Local deployment | Single command, reproducible |
| GitHub Actions | CI/CD | Free for public repos |

---

## License

MIT
