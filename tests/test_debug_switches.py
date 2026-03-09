from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthtsad.config import load_config
from synthtsad.pipeline import SyntheticGeneratorPipeline


def test_disable_components_for_debug(tmp_path: Path) -> None:
    cfg_raw = {
        "num_samples": 1,
        "seed": 13,
        "anomaly_sample_ratio": 0.0,
        "sequence_length": {"min": 64, "max": 64},
        "num_features": {"min": 3, "max": 3},
        "debug": {
            "enable_trend": False,
            "enable_seasonality": False,
            "enable_noise": False,
            "enable_causal": False,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg_raw), encoding="utf-8")

    cfg = load_config(cfg_path)
    out_dir = tmp_path / "out"
    SyntheticGeneratorPipeline(cfg).run(out_dir)

    npz_path = next(out_dir.glob("sample_*.npz"))
    arr = np.load(npz_path)

    assert np.allclose(arr["normal_series"], 0.0)
    assert np.allclose(arr["series"], 0.0)
    assert int(arr["point_mask"].sum()) == 0
    assert int(arr["point_mask_any"].sum()) == 0
