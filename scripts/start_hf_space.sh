#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/data/chroma

uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
  kill "${API_PID}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

streamlit run frontend/app.py \
  --server.address 0.0.0.0 \
  --server.port 7860
