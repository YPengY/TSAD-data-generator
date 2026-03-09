from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthtsad.config import load_config
from synthtsad.pipeline import SyntheticGeneratorPipeline


def test_trend_only_flow(tmp_path: Path) -> None:
    cfg_raw = {
        "num_samples": 1,
        "seed": 2026,
        "anomaly_sample_ratio": 0.0,
        "sequence_length": {"min": 64, "max": 64},
        "num_series": {"min": 3, "max": 3},
        "weights": {
            "trend_type": {
                "increase": 1.0,
                "decrease": 0.0,
                "keep_steady": 0.0,
                "multiple": 0.0,
                "arima": 0.0,
            }
        },
        "debug": {
            "enable_trend": True,
            "enable_seasonality": False,
            "enable_noise": False,
            "enable_causal": False,
            "enable_local_anomaly": False,
            "enable_seasonal_anomaly": False,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg_raw), encoding="utf-8")

    cfg = load_config(cfg_path)
    out_dir = tmp_path / "out"
    SyntheticGeneratorPipeline(cfg).run(out_dir)

    npz_path = next(out_dir.glob("sample_*.npz"))
    arr = np.load(npz_path)

    normal = arr["normal_series"]
    observed = arr["series"]

    # Trend-only path: no anomaly and no causal remapping.
    assert np.allclose(observed, normal)
    assert int(arr["point_mask"].sum()) == 0
    assert int(arr["point_mask_any"].sum()) == 0

    # Forced "increase" trend should produce positive first differences.
    assert np.all(np.diff(normal, axis=0) > 0.0)

    json_path = next(out_dir.glob("sample_*.json"))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    stage1 = payload["metadata"]["stage1_params"]
    assert all(spec["trend"]["trend_type"] == "increase" for spec in stage1)
