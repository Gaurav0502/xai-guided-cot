# create .env
touch .env

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Setup root: $ROOT_DIR"

# create data directories
DIRS=(
  "data"
  "data/datasets"
  "data/dataset_config"
  "data/shap_values"
  "data/batch_outputs"
  "data/tune_config"
  "data/batches"
  "data/metrics"
)

for d in "${DIRS[@]}"; do
  mkdir -p "$ROOT_DIR/$d"
done

echo "Created directories:"
for d in "${DIRS[@]}"; do
  echo "  - $ROOT_DIR/$d"
done