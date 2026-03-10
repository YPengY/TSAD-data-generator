# TSAD Studio

Isolated interactive frontend for the synthetic TSAD project.

## Run

```powershell
cd C:\Users\Administrator\Desktop\data\synthetic_tsad
.\.venv\Scripts\python.exe .\apps\tsad_studio\server.py --open-browser
```

Then open:

```text
http://127.0.0.1:8765
```

## What It Does

- edits every config parameter from `configs/default.json`
- random-fills the entire config with a valid sampled setup
- generates a single preview sample fully in memory
- visualizes:
  - Stage1 baseline
  - Stage2 normal output
  - final observed sample
  - point mask
  - DAG
  - realized events and metadata
