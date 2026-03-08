from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .anomaly.local import LocalAnomalyInjector
from .anomaly.seasonal import SeasonalAnomalyInjector
from .causal.arx import ARXSystem
from .causal.dag import CausalGraphSampler
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

    def _sample_dimensions(self, rng: np.random.Generator) -> tuple[int, int]:
        n = self.config.sequence_length.sample(rng)
        if self.config.multivariate_flag:
            d = self.config.num_features.sample(rng)
        else:
            d = 1
        d = int(np.clip(d, self.config.causal.num_nodes.min, self.config.causal.num_nodes.max))
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
            trend = render_trend(t=t, params=spec["trend"])
            season = render_seasonality(t=t, params=spec["seasonality"])
            noise = render_noise(n=n, params=spec["noise"])
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
            graph = CausalGraphSampler(self.config).sample_graph(num_nodes=d, rng=rng)
            arx = ARXSystem(self.config, graph)
            arx_params = arx.sample_params(rng)

            # Stage 3 (parameter/event sampling only)
            local_injector = LocalAnomalyInjector(self.config)
            seasonal_injector = SeasonalAnomalyInjector(self.config)
            sampled_local_events = []
            sampled_seasonal_events = []

            if rng.random() < self.config.anomaly_sample_ratio:
                sampled_local_events = local_injector.sample_events(n=n, d=d, rng=rng, graph=graph)
                sampled_seasonal_events = seasonal_injector.sample_events(n=n, d=d, rng=rng)

            # Realization from sampled parameters.
            x_base = self._realize_stage1(t=t, stage1_params=stage1_params)
            x_normal, causal_state = arx.simulate_with_params(x_base=x_base, n_steps=n, params=arx_params)

            x_observed = x_normal.copy()
            realized_events = []
            if sampled_local_events:
                x_observed, local_events = local_injector.apply_events(
                    x_normal=x_observed,
                    events=sampled_local_events,
                    graph=graph,
                    causal_state=causal_state,
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
