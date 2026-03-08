from __future__ import annotations

import numpy as np

from ..config import GeneratorConfig
from .local import AnomalyEvent


class SeasonalAnomalyInjector:
    """Stage 3 seasonal/contextual anomalies from Appendix C.4.2."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def _sample_window(self, n: int, rng: np.random.Generator) -> tuple[int, int]:
        max_len = min(self.config.anomaly.window_length.max, n)
        min_len = min(self.config.anomaly.window_length.min, max_len)
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, n - length + 1))
        return start, start + length

    def _frequency_warp(self, segment: np.ndarray, factor: float) -> np.ndarray:
        m = segment.size
        src = np.arange(m, dtype=float)
        warped = (src * factor) % max(1, m - 1)
        return np.interp(warped, src, segment)

    def sample_events(self, n: int, d: int, rng: np.random.Generator) -> list[AnomalyEvent]:
        if d == 0 or rng.random() > self.config.anomaly.p_use_seasonal_injector:
            return []

        node = int(rng.integers(0, d))
        t_start, t_end = self._sample_window(n, rng)
        kind = str(rng.choice(self.config.anomaly.seasonal_types))

        params: dict[str, float] = {}
        if kind == "amplitude_scaling":
            params["scale"] = float(rng.uniform(0.35, 2.2))
        elif kind == "frequency_change":
            params["factor"] = float(rng.uniform(0.5, 1.9))
        elif kind == "phase_shift":
            params["shift"] = float(rng.integers(1, max(2, (t_end - t_start) // 4 + 1)))
        elif kind == "noise_injection":
            params["noise_scale"] = float(rng.uniform(0.3, 1.2))

        return [
            AnomalyEvent(
                anomaly_type=kind,
                node=node,
                t_start=t_start,
                t_end=t_end,
                params=params,
                is_endogenous=False,
                root_cause_node=None,
                affected_nodes=[node],
            )
        ]

    def apply_events(
        self,
        x_input: np.ndarray,
        events: list[AnomalyEvent],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[AnomalyEvent]]:
        x_out = x_input.copy()
        realized: list[AnomalyEvent] = []

        for event in events:
            t_start, t_end = int(event.t_start), int(event.t_end)
            node = int(event.node)
            seg = x_out[t_start:t_end, node].copy()
            kind = event.anomaly_type

            if kind == "waveform_inversion":
                seg = -seg
            elif kind == "amplitude_scaling":
                seg = float(event.params["scale"]) * seg
            elif kind == "frequency_change":
                seg = self._frequency_warp(seg, float(event.params["factor"]))
            elif kind == "phase_shift":
                seg = np.roll(seg, int(event.params["shift"]))
            else:
                std = float(np.std(seg)) + 1e-4
                noise_scale = float(event.params.get("noise_scale", 0.5))
                seg = seg + rng.normal(0.0, noise_scale * std, size=seg.size)
                kind = "noise_injection"

            x_out[t_start:t_end, node] = seg
            realized.append(
                AnomalyEvent(
                    anomaly_type=kind,
                    node=node,
                    t_start=t_start,
                    t_end=t_end,
                    params=event.params,
                    is_endogenous=False,
                    root_cause_node=None,
                    affected_nodes=[node],
                )
            )

        return x_out, realized

    # Backward-compatible wrapper.
    def inject(
        self,
        x_input: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[AnomalyEvent]]:
        sampled = self.sample_events(n=x_input.shape[0], d=x_input.shape[1], rng=rng)
        return self.apply_events(x_input=x_input, events=sampled, rng=rng)
