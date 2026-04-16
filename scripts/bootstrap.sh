#!/usr/bin/env bash
# One-shot local setup script.
# Run this once after cloning: bash scripts/bootstrap.sh

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== MLOps Template Bootstrap ===${NC}"

# 1. Check dependencies
echo "Checking dependencies..."
for cmd in docker python3 pip; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: $cmd is not installed. Please install it first."
    exit 1
  fi
done

DOCKER_COMPOSE_CMD="docker compose"
if ! docker compose version &>/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD="docker-compose"
  if ! docker-compose version &>/dev/null 2>&1; then
    echo "ERROR: Docker Compose not found. Install Docker Desktop or docker-compose."
    exit 1
  fi
fi
echo "  ✓  Docker, Python, Compose found"

# 2. Create .env if it doesn't exist
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  ✓  .env created from .env.example"
else
  echo "  ✓  .env already exists"
fi

# 3. Create data directories
mkdir -p data/raw data/processed data/reference
mkdir -p monitoring/reports
echo "  ✓  Data directories created"

# 4. Install Python dev dependencies
echo "Installing Python dependencies..."
pip install -e ".[dev]" -q
echo "  ✓  Python dependencies installed"

# 5. Start infra services
echo -e "\n${YELLOW}Starting Docker services (this may take a few minutes on first run)...${NC}"
$DOCKER_COMPOSE_CMD -f infra/docker-compose.yml up -d --build --wait
echo "  ✓  All services started"

# 6. Generate data and seed MLflow
echo -e "\n${YELLOW}Generating synthetic data and seeding first model...${NC}"
python scripts/generate_data.py --n 5000
python scripts/seed_mlflow.py
echo "  ✓  Data generated and first model registered"

echo -e "\n${GREEN}=== Bootstrap complete ===${NC}"
echo ""
echo "  MLflow    → http://localhost:5000"
echo "  Airflow   → http://localhost:8080   (admin / admin)"
echo "  API       → http://localhost:8000/docs"
echo "  Grafana   → http://localhost:3000   (admin / admin)"
echo "  MinIO     → http://localhost:9001   (minioadmin / minioadmin)"
echo ""
echo "  Next steps:"
echo "    make demo          — run full end-to-end demo"
echo "    make load-test     — send 500 prediction requests"
echo "    make trigger-drift — manually trigger drift check"
