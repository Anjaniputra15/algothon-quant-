#!/bin/bash
set -e

# Start MLflow UI in the background, logging to a file
echo "Starting MLflow UI on port 5000..."
mlflow ui --host 0.0.0.0 --port 5000 &

# Start the Next.js frontend on port 3000
echo "Starting Next.js frontend on port 3000..."
cd /app/frontend/nextjs
npm start -- --port 3000 