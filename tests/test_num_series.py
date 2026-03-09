from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthtsad.config import load_config
from synthtsad.pipeline import SyntheticGeneratorPipeline


def test_num_series_is_independent_and_sampled_first(tmp_path: Path) -> None:
    cfg_raw = {
        "num_samples": 1,
        "seed": 9,
        "sequence_length": {"min": 48, "max": 48},
        "num_series": {"min": 4, "max": 4},
        # Legacy flag should not override explicit num_series.
        "multivariate_flag": False,
        "anomaly_sample_ratio": 0.0,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg_raw), encoding="utf-8")

    cfg = load_config(cfg_path)
    out_dir = tmp_path / "out"
    SyntheticGeneratorPipeline(cfg).run(out_dir)

    npz_path = next(out_dir.glob("sample_*.npz"))
    arr = np.load(npz_path)
    assert arr["series"].shape[1] == 4
