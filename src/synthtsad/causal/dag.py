from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GeneratorConfig


@dataclass
class CausalGraph:
    num_nodes: int
    adjacency: np.ndarray  # shape [D, D], adjacency[parent, child] in {0, 1}
    topo_order: list[int]
    parents: list[list[int]]


class CausalGraphSampler:
    """Stage 2.1: DAG sampling with Erdos-Renyi connectivity."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def sample_graph(self, num_nodes: int, rng: np.random.Generator) -> CausalGraph:
        if num_nodes <= 1:
            adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int8)
            return CausalGraph(
                num_nodes=num_nodes, adjacency=adjacency, topo_order=[0], parents=[[]]
            )

        p = float(self.config.causal.edge_density)
        order = list(rng.permutation(num_nodes).astype(int))
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int8)

        for i in range(num_nodes):
            parent = order[i]
            for j in range(i + 1, num_nodes):
                child = order[j]
                if rng.random() < p:
                    adjacency[parent, child] = 1

        parents = [np.where(adjacency[:, i] == 1)[0].astype(int).tolist() for i in range(num_nodes)]
        return CausalGraph(
            num_nodes=num_nodes, adjacency=adjacency, topo_order=order, parents=parents
        )
