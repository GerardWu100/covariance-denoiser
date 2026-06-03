# GUIDE_scripts

## Purpose

`scripts/` holds thin shell entry points. Reusable logic lives in `src/covariance_denoiser/`.

## Scripts

- `run_offline_demo.sh`: run the offline demo pipeline (`outputs/demo/` by default).
- `run_tests.sh`: run the full pytest suite.

## Run

```bash
chmod +x scripts/*.sh
./scripts/run_offline_demo.sh
./scripts/run_tests.sh
```
