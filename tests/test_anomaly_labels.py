from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthtsad.config import load_config
from synthtsad.pipeline import SyntheticGeneratorPipeline


def test_anomaly_labels_present_when_ratio_one(tmp_path: Path) -> None:
    cfg_raw = {
        "num_samples": 1,
        "seed": 11,
        "anomaly_sample_ratio": 1.0,
        "sequence_length": {"min": 80, "max": 80},
        "num_series": {"min": 4, "max": 4},
        "anomaly": {
            "events_per_sample": {"min": 1, "max": 1},
            "p_use_seasonal_injector": 0.0,
            "local_types": ["upward_spike"],
            "p_endogenous": 0.0,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg_raw), encoding="utf-8")

    cfg = load_config(cfg_path)
    out_dir = tmp_path / "out"
    SyntheticGeneratorPipeline(cfg).run(out_dir)

    npz_path = next(out_dir.glob("sample_*.npz"))
    arr = np.load(npz_path)
    assert int(arr["point_mask_any"].sum()) > 0

    json_path = next(out_dir.glob("sample_*.json"))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["events"], "Expected non-empty events list"
