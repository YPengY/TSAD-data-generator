from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthtsad.config import load_config
from synthtsad.pipeline import SyntheticGeneratorPipeline


def _write_test_config(tmp_path: Path) -> Path:
    cfg = {
        "num_samples": 2,
        "sequence_length": {"min": 64, "max": 96},
        "num_series": {"min": 3, "max": 5},
        "seed": 7,
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_pipeline_smoke(tmp_path: Path) -> None:
    cfg_path = _write_test_config(tmp_path)
    cfg = load_config(cfg_path)

    out_dir = tmp_path / "out"
    pipeline = SyntheticGeneratorPipeline(cfg)
    pipeline.run(out_dir)

    npz_files = sorted(out_dir.glob("sample_*.npz"))
    json_files = sorted(out_dir.glob("sample_*.json"))

    assert len(npz_files) == 2
    assert len(json_files) == 2

    arr = np.load(npz_files[0])
    assert "series" in arr
    assert "normal_series" in arr
    assert "point_mask" in arr
    assert arr["series"].shape == arr["normal_series"].shape
