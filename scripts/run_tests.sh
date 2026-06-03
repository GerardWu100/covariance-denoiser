#!/usr/bin/env bash
# Thin wrapper: run the full test suite from the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python -m pytest -v "$@"
