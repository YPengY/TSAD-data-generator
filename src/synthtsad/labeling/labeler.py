from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..anomaly.local import AnomalyEvent


class LabelBuilder:
    """Build point/event/root-cause labels for generated samples."""

    def __init__(self, config) -> None:
        self.config = config

    def build(
        self,
        x_normal: np.ndarray,
        x_anom: np.ndarray,
        events: list[AnomalyEvent],
        graph,
        causal_state,
    ) -> dict[str, Any]:
        t, d = x_anom.shape
        point_mask = np.zeros((t, d), dtype=np.uint8)

        root_to_nodes: dict[int, set[int]] = defaultdict(set)
        event_records: list[dict[str, Any]] = []

        for event in events:
            s = max(0, int(event.t_start))
            e = min(t, int(event.t_end))
            if s >= e:
                continue
            node = int(event.node)
            point_mask[s:e, node] = 1

            if event.root_cause_node is not None:
                root = int(event.root_cause_node)
                root_to_nodes[root].update(event.affected_nodes)

            event_records.append(event.to_dict())

        point_mask_any = (np.sum(point_mask, axis=1) > 0).astype(np.uint8)
        root_cause_nodes = sorted(root_to_nodes.keys())
        affected_nodes = {str(k): sorted(v) for k, v in root_to_nodes.items()}

        return {
            "point_mask": point_mask,
            "point_mask_any": point_mask_any,
            "events": event_records,
            "root_cause": root_cause_nodes,
            "affected_nodes": affected_nodes,
            "is_anomalous_sample": int(point_mask_any.any()),
        }
