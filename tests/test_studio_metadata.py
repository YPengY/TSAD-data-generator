from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
STUDIO_ROOT = REPO_ROOT / "apps" / "tsad_studio"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(STUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDIO_ROOT))

import studio_core


def test_preview_metadata_uses_stage_scoped_structure() -> None:
    payload = studio_core.import_config_text(
        json.dumps(
            {
                "seed": 11,
                "num_series": {"min": 2, "max": 2},
            }
        )
    )
    preview = studio_core.preview_sample(payload["config"])
    metadata = preview["metadata"]

    assert sorted(metadata.keys()) == ["sample", "stage1", "stage2", "stage3"]
    assert sorted(metadata["sample"].keys()) == ["seed_state"]
    assert sorted(metadata["stage1"].keys()) == ["params"]
    assert sorted(metadata["stage2"].keys()) == ["params"]
    assert sorted(metadata["stage3"].keys()) == ["sampled_events"]
    assert sorted(metadata["stage3"]["sampled_events"].keys()) == ["local", "seasonal"]
