from __future__ import annotations

import numpy as np

from synthtsad import load_config_from_raw
from synthtsad.anomaly.local import AnomalyEvent
from synthtsad.labeling.labeler import LabelBuilder


def test_root_cause_affected_nodes_use_event_window_only() -> None:
    cfg = load_config_from_raw(
        {
            "num_samples": 1,
            "sequence_length": {"min": 10, "max": 10},
            "num_series": {"min": 3, "max": 3},
            "seed": 7,
        }
    )
    builder = LabelBuilder(cfg)

    x_normal = np.zeros((10, 3), dtype=float)
    x_anom = x_normal.copy()
    x_anom[2:5, 0] = 1.0
    x_anom[3:5, 1] = 0.5
    x_anom[7:9, 2] = 2.0

    events = [
        AnomalyEvent(
            anomaly_type="upward_spike",
            node=0,
            t_start=2,
            t_end=5,
            params={"amplitude": 1.0},
            is_endogenous=True,
            root_cause_node=0,
            affected_nodes=[0],
        ),
        AnomalyEvent(
            anomaly_type="upward_spike",
            node=2,
            t_start=7,
            t_end=9,
            params={"amplitude": 2.0},
            is_endogenous=False,
            root_cause_node=None,
            affected_nodes=[2],
        ),
    ]

    labels = builder.build(
        x_normal=x_normal,
        x_anom=x_anom,
        events=events,
        graph=None,
        causal_state=None,
    )

    assert labels["root_cause"] == [0]
    assert labels["affected_nodes"] == {"0": [0, 1]}
    assert labels["events"][0]["affected_nodes"] == [0, 1]
    assert labels["events"][1]["affected_nodes"] == [2]
    assert labels["summary"] == {
        "total": 2,
        "local": 2,
        "seasonal": 0,
        "endogenous": 1,
        "target_components": {"observed": 2},
    }
    assert labels["point_mask_any"].tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
