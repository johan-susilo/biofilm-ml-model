#!/usr/bin/env bash
set -e

# Convenience script to build and start the API + UI via Docker Compose

cd "$(dirname "$0")"

echo "Building containers..."
docker-compose up --build -d biofilm-api

echo "Waiting for health check..."
until curl -fsS http://localhost:8000/health >/dev/null 2>&1; do
  sleep 1
done

echo "Biofilm API is up!"
echo "   UI:  http://localhost:8000/ui"
echo "   API: http://localhost:8000/docs"
