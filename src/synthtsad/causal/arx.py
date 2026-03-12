from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GeneratorConfig
from ..interfaces import ARXModelParams, ARXParams
from ..utils import clamp_float
from .dag import CausalGraph


@dataclass
class ARXState:
    z: np.ndarray  # latent causal channel, shape [T, D]
    params: ARXModelParams


class ARXSystem:
    """Stage 2.2-2.4: ODE-inspired ARX discretization and signal mixing."""

    def __init__(self, config: GeneratorConfig, graph: CausalGraph) -> None:
        self.config = config
        self.graph = graph

    def sample_params(self, rng: np.random.Generator) -> ARXParams:
        d = self.graph.num_nodes
        c = self.config.causal

        a = rng.uniform(-c.a_i_bound, c.a_i_bound, size=d)
        alpha = rng.uniform(c.alpha_i_min, c.alpha_i_max, size=d)
        bias = rng.normal(0.0, c.bias_std, size=d)

        lag = np.zeros((d, d), dtype=np.int32)
        gain = np.zeros((d, d), dtype=np.float32)

        in_degree = np.sum(self.graph.adjacency, axis=0)
        for parent in range(d):
            for child in range(d):
                if self.graph.adjacency[parent, child] == 0:
                    continue
                lag[parent, child] = int(rng.integers(0, c.max_lag + 1))
                scale = c.b_ij_std / np.sqrt(max(1, in_degree[child]))
                gain[parent, child] = float(rng.normal(0.0, scale))

        return {
            "a": a.tolist(),
            "alpha": alpha.tolist(),
            "bias": bias.tolist(),
            "lag": lag.tolist(),
            "gain": gain.tolist(),
            "max_lag": int(c.max_lag),
        }

    def simulate_with_params(
        self,
        x_base: np.ndarray,
        n_steps: int,
        params: ARXParams,
    ) -> tuple[np.ndarray, ARXState]:
        if x_base.ndim == 1:
            x_base = np.tile(x_base[:, None], (1, self.graph.num_nodes))

        t_final = min(n_steps, x_base.shape[0])
        x_base = x_base[:t_final].astype(float, copy=False)
        d = self.graph.num_nodes

        a = np.array(params["a"], dtype=float)
        alpha = np.array(params["alpha"], dtype=float)
        bias = np.array(params["bias"], dtype=float)
        lag = np.array(params["lag"], dtype=int)
        gain = np.array(params["gain"], dtype=float)

        x = np.zeros((t_final, d), dtype=float)
        z = np.zeros((t_final, d), dtype=float)

        for t in range(t_final):
            for node in self.graph.topo_order:
                prev = z[t - 1, node] if t > 0 else 0.0
                forcing = 0.0
                for parent in self.graph.parents[node]:
                    lag_steps = int(lag[parent, node])
                    src_t = t - lag_steps
                    if src_t >= 0:
                        forcing += float(gain[parent, node]) * x[src_t, parent]

                z_val = a[node] * prev + forcing + bias[node]
                z_val = clamp_float(float(z_val), -20.0, 20.0)
                z[t, node] = z_val
                x[t, node] = (1.0 - alpha[node]) * x_base[t, node] + alpha[node] * z_val

            x[t] = np.clip(x[t], -30.0, 30.0)

        return x, ARXState(z=z, params=params)

    def simulate_linear_response(
        self,
        x_base: np.ndarray,
        n_steps: int,
        params: ARXParams,
    ) -> tuple[np.ndarray, ARXState]:
        linear_params: ARXParams = {
            "a": list(params["a"]),
            "alpha": list(params["alpha"]),
            "bias": [0.0 for _ in range(self.graph.num_nodes)],
            "lag": [list(row) for row in params["lag"]],
            "gain": [list(row) for row in params["gain"]],
            "max_lag": int(params["max_lag"]),
        }
        return self.simulate_with_params(x_base=x_base, n_steps=n_steps, params=linear_params)

    def simulate_from_baseline(
        self,
        x_base: np.ndarray,
        n_steps: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, ARXState]:
        params = self.sample_params(rng)
        return self.simulate_with_params(x_base=x_base, n_steps=n_steps, params=params)
