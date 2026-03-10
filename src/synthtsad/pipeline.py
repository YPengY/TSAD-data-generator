from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .anomaly.local import LocalAnomalyInjector
from .anomaly.seasonal import SeasonalAnomalyInjector
from .causal.arx import ARXState, ARXSystem
from .causal.dag import CausalGraph, CausalGraphSampler
from .components.noise import render_noise, sample_noise_params
from .components.seasonality import render_seasonality, sample_seasonality_params
from .components.trend import render_trend, sample_trend_params
from .config import GeneratorConfig
from .io.writer import DatasetWriter
from .labeling.labeler import LabelBuilder


class SyntheticGeneratorPipeline:
    """Four-stage synthetic generator aligned with Appendix C.

    This pipeline uses a parameter-first workflow:
    1) sample generation parameters
    2) realize final sequences from the sampled parameters
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def _empty_graph(self, d: int) -> CausalGraph:
        adjacency = np.zeros((d, d), dtype=np.int8)
        parents = [[] for _ in range(d)]
        topo_order = list(range(d))
        return CausalGraph(num_nodes=d, adjacency=adjacency, topo_order=topo_order, parents=parents)

    def _disabled_causal_state(self, n: int, d: int) -> ARXState:
        return ARXState(
            z=np.zeros((n, d), dtype=float),
            params={"disabled": True},
        )

    def _sample_dimensions(self, rng: np.random.Generator) -> tuple[int, int]:
        n = self.config.sequence_length.sample(rng)
        d = self.config.num_series.sample(rng)
        return n, d

    def _sample_stage1_params(self, n: int, d: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        params: list[dict[str, Any]] = []
        for node in range(d):
            params.append(
                {
                    "node": node,
                    "trend": sample_trend_params(n=n, config=self.config, rng=rng),
                    "seasonality": sample_seasonality_params(n=n, config=self.config, rng=rng),
                    "noise": sample_noise_params(n=n, config=self.config, rng=rng),
                }
            )
        return params

    def _realize_stage1(self, t: np.ndarray, stage1_params: list[dict[str, Any]]) -> np.ndarray:
        n = t.size
        d = len(stage1_params)
        x_base = np.zeros((n, d), dtype=float)

        for spec in stage1_params:
            node = int(spec["node"])
            trend = render_trend(t=t, params=spec["trend"]) if self.config.debug.enable_trend else np.zeros(n, dtype=float)
            season = (
                render_seasonality(t=t, params=spec["seasonality"])
                if self.config.debug.enable_seasonality
                else np.zeros(n, dtype=float)
            )
            noise = render_noise(n=n, params=spec["noise"]) if self.config.debug.enable_noise else np.zeros(n, dtype=float)
            x_base[:, node] = trend + season + noise

        return x_base

    def run(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = DatasetWriter(output_dir)

        rng = np.random.default_rng(self.config.seed)

        for sample_id in range(self.config.num_samples):
            n, d = self._sample_dimensions(rng)
            t = np.arange(n, dtype=float)

            # Stage 1 (parameter sampling only)
            stage1_params = self._sample_stage1_params(n=n, d=d, rng=rng)

            # Stage 2 (parameter sampling only)
            if self.config.debug.enable_causal:
                graph = CausalGraphSampler(self.config).sample_graph(num_nodes=d, rng=rng)
                arx = ARXSystem(self.config, graph)
                arx_params = arx.sample_params(rng)
            else:
                graph = self._empty_graph(d)
                arx = ARXSystem(self.config, graph)
                arx_params = {"disabled": True}

            # Stage 3 (parameter/event sampling only)
            local_injector = LocalAnomalyInjector(self.config)
            seasonal_injector = SeasonalAnomalyInjector(self.config)
            sampled_local_events = []
            sampled_seasonal_events = []

            if rng.random() < self.config.anomaly_sample_ratio:
                if self.config.debug.enable_local_anomaly:
                    sampled_local_events = local_injector.sample_events(n=n, d=d, rng=rng, graph=graph)
                if self.config.debug.enable_seasonal_anomaly:
                    sampled_seasonal_events = seasonal_injector.sample_events(n=n, d=d, rng=rng)

            if not self.config.debug.enable_causal:
                for event in sampled_local_events:
                    event.is_endogenous = False
                    event.root_cause_node = None

            # Realization from sampled parameters.
            x_base = self._realize_stage1(t=t, stage1_params=stage1_params)

            # Endogenous local events are injected before causal realization so they
            # propagate naturally through the structural dynamics.
            pre_causal_local_events = [e for e in sampled_local_events if bool(e.is_endogenous)]
            post_causal_local_events = [e for e in sampled_local_events if not bool(e.is_endogenous)]

            x_base_anom = x_base.copy()
            realized_events = []
            if pre_causal_local_events:
                x_base_anom, local_events = local_injector.apply_events(
                    x_normal=x_base_anom,
                    events=pre_causal_local_events,
                )
                realized_events.extend(local_events)

            if self.config.debug.enable_causal:
                x_normal, causal_state = arx.simulate_with_params(x_base=x_base, n_steps=n, params=arx_params)
                x_observed, _ = arx.simulate_with_params(x_base=x_base_anom, n_steps=n, params=arx_params)
            else:
                x_normal = x_base.copy()
                x_observed = x_base_anom.copy()
                causal_state = self._disabled_causal_state(n=n, d=d)

            if post_causal_local_events:
                x_observed, local_events = local_injector.apply_events(
                    x_normal=x_observed,
                    events=post_causal_local_events,
                )
                realized_events.extend(local_events)

            if sampled_seasonal_events:
                x_observed, seasonal_events = seasonal_injector.apply_events(
                    x_input=x_observed,
                    events=sampled_seasonal_events,
                    rng=rng,
                )
                realized_events.extend(seasonal_events)

            # Stage 4 labels
            labels = LabelBuilder(self.config).build(
                x_normal=x_normal,
                x_anom=x_observed,
                events=realized_events,
                graph=graph,
                causal_state=causal_state,
            )

            metadata = {
                "sample_seed_state": str(rng.bit_generator.state["state"]["state"]),
                "stage1_params": stage1_params,
                "stage2_params": arx_params,
                "stage3_sampled_events": {
                    "local": [e.to_dict() for e in sampled_local_events],
                    "seasonal": [e.to_dict() for e in sampled_seasonal_events],
                },
            }
            writer.write_sample(
                sample_id=sample_id,
                normal_series=x_normal,
                observed_series=x_observed,
                labels=labels,
                graph=graph,
                metadata=metadata,
            )
