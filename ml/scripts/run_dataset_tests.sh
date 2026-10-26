#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$PROJECT_ROOT/ml/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment is missing: $PYTHON" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
"$PYTHON" ml/scripts/check_pose_datasets.py
"$PYTHON" ml/scripts/render_catflw_sample.py
"$PYTHON" ml/scripts/render_animalpose_sample.py

echo
echo "Done. Open these files:"
echo "  $PROJECT_ROOT/ml/outputs/catflw-face-landmarks.png"
echo "  $PROJECT_ROOT/ml/outputs/animalpose-cat-keypoints.png"
