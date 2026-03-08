# Synthetic TSAD Data Generator

A parameterized implementation framework for Appendix C (Data Generation) of the TimeRCD paper.

## What Is Implemented

- Stage 1 baseline generation: trend + seasonality + heteroskedastic noise
- Stage 2 causal context: DAG sampling + ARX-style causal propagation
- Stage 3 anomaly injection:
  - local/change anomalies (spike, shift, shake, plateau)
  - seasonal/contextual anomalies (inversion, amplitude, frequency, phase, noise)
  - endogenous propagation over causal graph (with edge gains and lags)
- Stage 4 labeling:
  - point-level mask `[T, D]`
  - sequence-level mask `[T]`
  - event table
  - root-cause and affected-node mappings
- Output format:
  - `sample_XXXXXX.npz` (series/labels arrays)
  - `sample_XXXXXX.json` (events, graph, parameters, summary)

## Project Layout

- `configs/default.json|yaml`: generation controls
- `scripts/generate_dataset.py`: CLI entry
- `src/synthtsad/components/*`: Stage 1 modules
- `src/synthtsad/causal/*`: Stage 2 modules
- `src/synthtsad/anomaly/*`: Stage 3 modules
- `src/synthtsad/labeling/labeler.py`: Stage 4 labels
- `src/synthtsad/io/writer.py`: dataset writer
- `scripts/setup_env.ps1`: venv bootstrap for Windows

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
.\.venv\Scripts\python.exe .\scripts\generate_dataset.py --config .\configs\default.json --output .\outputs --num-samples 20 --seed 7
```

## Notes

- JSON config works without PyYAML.
- YAML config requires `pyyaml`.
- Sampling is deterministic when `seed` is fixed.\n- `setup_env.ps1` automatically falls back to `include-system-site-packages=true` if `ensurepip` is blocked by local policy.

