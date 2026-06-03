#!/usr/bin/env bash
# Thin wrapper: run the offline research pipeline from the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python -m covariance_denoiser.cli run-offline-demo \
  --data-dir data/raw \
  --output-dir outputs/demo \
  "$@"
