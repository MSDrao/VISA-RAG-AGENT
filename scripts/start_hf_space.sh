#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/data/chroma

restore_chroma_snapshot() {
  if [ -z "${CHROMA_SNAPSHOT_URL:-}" ]; then
    echo "No CHROMA_SNAPSHOT_URL configured. Skipping snapshot restore."
    return
  fi

  if [ -f "/app/data/chroma/chroma.sqlite3" ]; then
    echo "Existing Chroma snapshot found. Skipping restore."
    return
  fi

  tmp_archive="/tmp/chroma_snapshot.tar.gz"
  echo "Downloading Chroma snapshot from configured URL..."
  curl -L "${CHROMA_SNAPSHOT_URL}" -o "${tmp_archive}"

  echo "Extracting Chroma snapshot..."
  rm -rf /app/data/chroma
  mkdir -p /app/data/chroma
  tar -xzf "${tmp_archive}" -C /app

  if [ ! -f "/app/data/chroma/chroma.sqlite3" ]; then
    echo "Snapshot restore finished, but chroma.sqlite3 was not found."
    exit 1
  fi

  echo "Chroma snapshot restored successfully."
}

restore_chroma_snapshot

uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
  kill "${API_PID}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

streamlit run frontend/app.py \
  --server.address 0.0.0.0 \
  --server.port 7860
