from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
STUDIO_ROOT = REPO_ROOT / "apps" / "tsad_studio"
STATIC_ROOT = STUDIO_ROOT / "static"

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


def test_studio_bootstrap_payload_uses_descriptive_labels_and_hints() -> None:
    payload = studio_core.get_bootstrap_payload()
    ui = payload["ui"]

    assert (
        ui["locales"]["en"]["pathLabels"]["anomaly.local.per_type.upward_spike"] == "Upward Spike"
    )
    assert ui["locales"]["zh"]["pathLabels"]["anomaly.local.per_type.upward_spike"] == "上升尖峰"
    assert (
        ui["locales"]["en"]["pathDescriptions"]["anomaly.local.per_type.upward_spike"]
        == "Per-type settings for Upward Spike."
    )
    assert ui["locales"]["en"]["pathDescriptions"]["stage1.trend.change_points"] == (
        "How many slope changes a piecewise trend may contain."
    )


def test_preview_payload_matches_studio_frontend_contract() -> None:
    payload = studio_core.import_config_text(
        json.dumps(
            {
                "seed": 17,
                "num_series": {"min": 2, "max": 2},
                "sequence_length": {"min": 48, "max": 48},
            }
        )
    )
    preview = studio_core.preview_sample(payload["config"])
    metadata = preview["metadata"]
    labels = preview["labels"]
    summary = labels["summary"]

    assert isinstance(metadata["sample"]["seed_state"], str)
    assert isinstance(metadata["stage1"]["params"], list)
    assert isinstance(metadata["stage3"]["sampled_events"]["local"], list)
    assert isinstance(metadata["stage3"]["sampled_events"]["seasonal"], list)

    stage2_params = metadata["stage2"]["params"]
    assert isinstance(stage2_params, dict)
    if stage2_params.get("disabled") is True:
        assert sorted(stage2_params.keys()) == ["disabled"]
    else:
        assert "max_lag" in stage2_params

    assert sorted(summary.keys()) == [
        "endogenous",
        "local",
        "seasonal",
        "target_components",
        "total",
    ]
    assert preview["summary"]["num_events"] == summary["total"]
    assert preview["summary"]["num_local_events"] == summary["local"]
    assert preview["summary"]["num_seasonal_events"] == summary["seasonal"]
    assert preview["summary"]["num_endogenous_events"] == summary["endogenous"]


def test_studio_static_shell_exposes_structured_metadata_panel() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'id="metadata-stages"' in html
    assert 'data-i18n="metadata.stagesTitle"' in html
    assert 'data-i18n="metadata.stagesSubtitle"' in html
    assert 'data-i18n="import.note"' in html
    assert "Runtime and Studio import both accept only the formal schema." in html


def test_studio_script_consumes_structured_metadata_and_label_summary() -> None:
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'metadataStages: document.getElementById("metadata-stages")' in script
    assert "function renderMetadataStages()" in script
    assert "const metadata = state.preview.metadata ?? {};" in script
    assert "const stage3Sampled = metadata.stage3?.sampled_events ?? {};" in script
    assert "const summary = state.preview.labels.summary ?? {};" in script
    assert '"metadata.stagesTitle": "Structured Metadata"' in script
    assert '"metadata.field.localSampled": "Local sampled"' in script
    assert '"import.note": "Runtime and Studio import both accept only the formal schema.' in script
    assert "title.textContent = formatTokenLabel(event.anomaly_type);" in script
    assert "function describeCurrentValue(path)" in script
    assert "Configure this parameter." not in script
