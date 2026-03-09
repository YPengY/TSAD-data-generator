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
- Local anomalies (spike/shift/shake/plateau) and seasonal anomalies.
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
- `tests/*`: smoke and behavior tests.

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
