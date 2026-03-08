from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthtsad.causal.dag import CausalGraphSampler
from synthtsad.config import load_config


def test_dag_is_acyclic(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"seed": 123, "causal": {"edge_density": 0.4}}), encoding="utf-8")
    cfg = load_config(cfg_path)

    rng = np.random.default_rng(123)
    graph = CausalGraphSampler(cfg).sample_graph(num_nodes=12, rng=rng)

    pos = {node: idx for idx, node in enumerate(graph.topo_order)}
    parents, children = np.where(graph.adjacency == 1)

    for p, c in zip(parents.tolist(), children.tolist()):
        assert pos[p] < pos[c], "Edge violates topological order, graph is cyclic"
