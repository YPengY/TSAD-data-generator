from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces import GenerationMetadata, LabelPayload


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


class DatasetWriter:
    """Persist generated samples and metadata.

    NPZ payload:
    - series: anomalous/observed sequence [T, D]
    - normal_series: pre-anomaly reference [T, D]
    - point_mask: per-node anomaly labels [T, D]
    - point_mask_any: sequence-level mask [T]

    JSON payload:
    - event list, causal graph, sampled parameters and summary statistics.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def write_sample(
        self,
        sample_id: int,
        normal_series: np.ndarray,
        observed_series: np.ndarray,
        labels: LabelPayload,
        graph,
        metadata: GenerationMetadata,
    ) -> None:
        npz_path = self.output_dir / f"sample_{sample_id:06d}.npz"
        json_path = self.output_dir / f"sample_{sample_id:06d}.json"

        np.savez_compressed(
            npz_path,
            series=observed_series,
            normal_series=normal_series,
            point_mask=labels["point_mask"],
            point_mask_any=labels["point_mask_any"],
        )

        payload = {
            "summary": {
                "sample_id": int(sample_id),
                "length": int(observed_series.shape[0]),
                "num_series": int(observed_series.shape[1]),
                "is_anomalous_sample": int(labels["is_anomalous_sample"]),
            },
            "labels": {
                "root_cause": labels["root_cause"],
                "affected_nodes": labels["affected_nodes"],
                "summary": labels["summary"],
            },
            "events": labels["events"],
            "graph": {
                "num_nodes": int(graph.num_nodes),
                "adjacency": graph.adjacency.tolist(),
                "topo_order": [int(v) for v in graph.topo_order],
                "parents": [[int(u) for u in nodes] for nodes in graph.parents],
            },
            "metadata": metadata,
        }
        json_path.write_text(
            json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
