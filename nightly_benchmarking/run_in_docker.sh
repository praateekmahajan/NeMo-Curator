#!/usr/bin/env bash
set -euo pipefail

# Run bench_driver.py inside the built container
# Usage:
#   micromamba run -n ray_curator_2506 bash nightly_benchmarking/run_in_docker.sh <image_tag> <matrix_yaml> <datasets_json>

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <image_tag> <matrix_yaml> <datasets_json>" >&2
  exit 1
fi

IMAGE_TAG="$1"
MATRIX_PATH="$2"
DATASETS_PATH="$3"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

micromamba run -n ray_curator_2506 docker run --rm --gpus all --net=host \
  -e MLFLOW_TRACKING_URI \
  -e MLFLOW_EXPERIMENT \
  -e SLACK_WEBHOOK_URL \
  -e CUDA_VISIBLE_DEVICES \
  -e RAY_BACKEND_LOG_LEVEL \
  -v /raid:/raid \
  -v "$REPO_ROOT":"$REPO_ROOT" \
  -w "$REPO_ROOT" \
  "$IMAGE_TAG" \
  python nightly_benchmarking/bench_driver.py --matrix "$MATRIX_PATH" --datasets "$DATASETS_PATH" --sink mlflow
