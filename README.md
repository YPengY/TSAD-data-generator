# TSAD Data Generator

A parameter-first synthetic time series anomaly dataset generator inspired by Appendix C of the TimeRCD paper.

## Overview

This project builds synthetic time series data in four stages:

1. Stage 1 (baseline): trend + seasonality + heteroskedastic noise.
2. Stage 2 (causal context): sample a DAG and ARX parameters for multivariate dependencies.
3. Stage 3 (anomalies): sample local and seasonal anomaly events, then apply them.
4. Stage 4 (labels): create point-level and event-level labels with root-cause metadata.

The workflow is parameter-first: it samples parameters first, then realizes final sequences from those parameters.

## Implemented Features

- Trend types: increase, decrease, steady, piecewise, arima-like.
- Seasonality types: none, sine, square, triangle, wavelet-like atoms.
- Noise with optional piecewise volatility bursts.
- Causal graph sampling (DAG) and ARX simulation.
- Local anomalies with a broader archetype catalog (spikes, bursts, wide spikes, outliers, asymmetric transients, spike+level interactions).
- Seasonal anomalies over the seasonal component (waveform transforms, harmonic edits, pulse geometry edits, wavelet atom edits).
- Endogenous propagation over causal edges.
- Output as NPZ + JSON per sample.

## Project Layout

- `configs/default.json|yaml`: generation configuration.
- `scripts/generate_dataset.py`: CLI entrypoint.
- `scripts/setup_env.ps1`: environment bootstrap script.
- `src/synthtsad/pipeline.py`: end-to-end orchestration.
- `src/synthtsad/components/*`: Stage 1 modules.
- `src/synthtsad/causal/*`: Stage 2 modules.
- `src/synthtsad/anomaly/*`: Stage 3 modules.
- `src/synthtsad/labeling/labeler.py`: Stage 4 labels.
- `src/synthtsad/io/writer.py`: output writer.
- `tests/test_*.py`: pytest unit, regression, and smoke tests.
- `tests/*_console.py`: manual console debugging helpers, excluded from pytest collection.

## Environment Setup (Windows)

```powershell
cd C:\Users\Administrator\Desktop\data\synthetic_tsad
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

## Run

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_dataset.py --config .\configs\default.json --output .\outputs
```

Optional overrides:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_dataset.py --config .\configs\default.json --output .\outputs --num-samples 120 --seed 7 --num-series 6
```

`num_series` is sampled at the beginning of each sample. Use `--num-series` to force a fixed value (`min=max`).

Debug switches (temporary disable by CLI):

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_dataset.py --config .\configs\default.json --output .\outputs --disable-causal --disable-noise --disable-seasonality
```

Print effective config without generation:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_dataset.py --config .\configs\default.json --print-config
```

You can also persist these toggles in config under:

```json
"debug": {
  "enable_trend": true,
  "enable_seasonality": true,
  "enable_noise": true,
  "enable_causal": true,
  "enable_local_anomaly": true,
  "enable_seasonal_anomaly": true
}
```

ARIMA trend is implemented as differenced ARMA(p,d,q) with small orders (`p,q<=2`, `d in [1,2]` by default), configurable in `stage1.trend.arima`.

## Output Format

For each sample, two files are generated:

- `sample_XXXXXX.npz`
  - `series`: observed sequence `[T, D]`
  - `normal_series`: normal reference `[T, D]`
  - `point_mask`: anomaly mask `[T, D]`
  - `point_mask_any`: sequence-level mask `[T]`
- `sample_XXXXXX.json`
  - summary, graph, sampled parameters, sampled events, realized events, label metadata

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Pytest ignores the `tests/*_console.py` debugging helpers and only runs the actual automated test files.

## Quality Checks

Install the dev toolchain through the existing setup script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

Run the staged quality gate:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_quality.ps1
```

Apply formatting and safe auto-fixes before re-running the gate:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_quality.ps1 -Fix
```

Current scope:

- `ruff format` for formatting
- `ruff check` with `E/F/I/UP/B` and `E501` ignored for the initial rollout
- `pyright` in `standard` mode over `src/synthtsad`
