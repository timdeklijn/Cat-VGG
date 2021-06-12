#!/bin/sh

set -e

mlflow server \
    --backend-store-uri /mlflow
    --default-artifact-root /mlflow
    --host 0.0.0.0 \
    --port 5000
