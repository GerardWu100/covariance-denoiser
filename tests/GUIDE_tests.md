# GUIDE_tests

## Purpose

`tests/` holds contract tests for the offline research pipeline.

## Structure

- `tests/unit/`: fast module-level tests (data, targets, features, models, notebook contract).
- `tests/integration/`: end-to-end pipeline tests that write artifacts under `outputs/`.

## Run

```bash
uv run python -m pytest -v
# or
./scripts/run_tests.sh
```
