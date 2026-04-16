.PHONY: up down train test test-integration demo load-test trigger-drift bootstrap help

help:
	@echo ""
	@echo "  make bootstrap       First-time setup (creates .env, MinIO bucket)"
	@echo "  make up              Start all services"
	@echo "  make down            Stop and remove all services and volumes"
	@echo "  make train           Trigger training DAG in Airflow"
	@echo "  make test            Run unit tests"
	@echo "  make test-integration  Run integration tests (needs stack up)"
	@echo "  make demo            Run full demo: generate data → train → predict → inject drift"
	@echo "  make load-test       Send 500 prediction requests to the API"
	@echo "  make trigger-drift   Manually trigger the drift check DAG"
	@echo ""

bootstrap:
	@cp -n .env.example .env || true
	@echo "✓  .env created from .env.example"
	@bash scripts/bootstrap.sh

up:
	docker compose -f infra/docker-compose.yml up -d --build --wait
	@echo ""
	@echo "  ✓  Services ready"
	@echo "  MLflow      → http://localhost:5000"
	@echo "  Airflow     → http://localhost:8080  (admin / admin)"
	@echo "  API         → http://localhost:8000"
	@echo "  Grafana     → http://localhost:3000  (admin / admin)"
	@echo "  Prometheus  → http://localhost:9090"
	@echo "  MinIO       → http://localhost:9001"

down:
	docker compose -f infra/docker-compose.yml down -v --remove-orphans

train:
	docker compose -f infra/docker-compose.yml exec airflow-webserver \
		airflow dags trigger train_register_dag
	@echo "Training DAG triggered — check Airflow at http://localhost:8080"

test:
	pip install -e ".[dev]" -q
	pytest tests/unit/ -v --tb=short

test-integration:
	docker compose -f infra/docker-compose.ci.yml up -d --wait
	pytest tests/integration/ -v --tb=short
	docker compose -f infra/docker-compose.ci.yml down -v

demo:
	@echo "=== Step 1: Generate synthetic data ==="
	python scripts/generate_data.py
	@echo "=== Step 2: Seed MLflow with first model ==="
	python scripts/seed_mlflow.py
	@echo "=== Step 3: Send sample prediction ==="
	curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"age":45,"tenure_months":12,"monthly_charge":89.5,"num_products":2,"contract_type":"month-to-month","payment_method":"electronic_check"}' | python -m json.tool
	@echo "=== Step 4: Load test (200 requests) ==="
	python scripts/load_test.py --n 200
	@echo "=== Step 5: Inject drift ==="
	python scripts/inject_drift.py
	@echo "=== Done — open Grafana at http://localhost:3000 ==="

load-test:
	python scripts/load_test.py --n 500

trigger-drift:
	docker compose -f infra/docker-compose.yml exec airflow-webserver \
		airflow dags trigger drift_check_dag
