#!/bin/bash

echo "Starting recommendation service..."

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "Project dir: $PROJECT_DIR"

# activate venv
if [ -d "env_recsys_start" ]; then
    echo "Activating virtual environment..."
    source env_recsys_start/bin/activate
else
    echo "Virtual environment not found!"
    exit 1
fi

# install dependencies if needed
echo "Checking dependencies..."
pip install -r requirements.txt >/dev/null 2>&1

# start service
echo "Launching FastAPI service..."

python -m uvicorn recommendations_service:app \
    --host 0.0.0.0 \
    --port 8000

echo "Service stopped"
