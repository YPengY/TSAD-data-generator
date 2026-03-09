from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..config import GeneratorConfig


@dataclass
class AnomalyEvent:
    anomaly_type: str
    node: int
    t_start: int
    t_end: int
    params: dict[str, Any]
    is_endogenous: bool
    root_cause_node: int | None
    affected_nodes: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalAnomalyInjector:
    """Stage 3 local/change anomalies from Appendix C.4.1."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def _sample_window(self, n: int, rng: np.random.Generator) -> tuple[int, int]:
        max_len = min(self.config.anomaly.window_length.max, n)
        min_len = min(self.config.anomaly.window_length.min, max_len)
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, n - length + 1))
        return start, start + length

    def _sample_template_params(
        self,
        kind: str,
        t_start: int,
        t_end: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        amplitude = float(rng.uniform(0.8, 3.0))
        params: dict[str, Any] = {"amplitude": amplitude}

        if kind in {"upward_spike", "downward_spike"}:
            params["center"] = int(rng.integers(t_start, t_end))
            params["half_width"] = max(1, (t_end - t_start) // 6)
        elif kind in {"sudden_increase", "sudden_decrease"}:
            params["t0"] = int(rng.integers(t_start, t_end))
            params["kappa"] = float(rng.uniform(0.1, 0.4))
        elif kind == "shake":
            params["freq"] = float(rng.uniform(0.1, 0.35))
            params["phase"] = float(rng.uniform(0.0, 2 * np.pi))
        elif kind == "plateau":
            params["sign"] = float(rng.choice([-1.0, 1.0]))

        return params

    def _render_template(self, kind: str, n: int, t_start: int, t_end: int, params: dict[str, Any]) -> np.ndarray:
        delta = np.zeros(n, dtype=float)
        idx = np.arange(t_start, t_end)
        rel = np.arange(idx.size)

        amplitude = float(params.get("amplitude", 1.0))
        sign = -1.0 if kind in {"downward_spike", "sudden_decrease"} else 1.0

        if kind in {"upward_spike", "downward_spike"}:
            center = int(params["center"])
            half_width = int(params["half_width"])
            shape = np.maximum(1.0 - np.abs(idx - center) / max(1, half_width), 0.0)
            delta[idx] = sign * amplitude * shape
            return delta

        if kind in {"sudden_increase", "sudden_decrease"}:
            t0 = int(params["t0"])
            kappa = float(params["kappa"])
            logits = kappa * (idx - t0)
            sigm = 1.0 / (1.0 + np.exp(-logits))
            delta[idx] = sign * amplitude * sigm
            return delta

        if kind == "plateau":
            phase = np.pi * rel / max(1, idx.size - 1)
            sgn = float(params.get("sign", 1.0))
            delta[idx] = sgn * amplitude * 0.5 * (1.0 - np.cos(phase))
            return delta

        freq = float(params.get("freq", 0.2))
        phase = float(params.get("phase", 0.0))
        window = np.sin(np.pi * rel / max(1, idx.size - 1)) ** 2
        delta[idx] = amplitude * window * np.sin(2 * np.pi * freq * rel + phase)
        return delta

    def sample_events(
        self,
        n: int,
        d: int,
        rng: np.random.Generator,
        graph=None,
    ) -> list[AnomalyEvent]:
        _ = graph
        events: list[AnomalyEvent] = []
        event_count = self.config.anomaly.events_per_sample.sample(rng)
        local_types = self.config.anomaly.local_types

        for _ in range(event_count):
            kind = str(rng.choice(local_types))
            node = int(rng.integers(0, d))
            t_start, t_end = self._sample_window(n, rng)
            params = self._sample_template_params(kind, t_start, t_end, rng)
            is_endogenous = bool(d > 1 and rng.random() < self.config.anomaly.p_endogenous)

            events.append(
                AnomalyEvent(
                    anomaly_type=kind,
                    node=node,
                    t_start=t_start,
                    t_end=t_end,
                    params=params,
                    is_endogenous=is_endogenous,
                    root_cause_node=node if is_endogenous else None,
                    affected_nodes=[node],
                )
            )

        return events

    def apply_events(
        self,
        x_normal: np.ndarray,
        events: list[AnomalyEvent],
        graph=None,
        causal_state=None,
    ) -> tuple[np.ndarray, list[AnomalyEvent]]:
        _ = graph
        _ = causal_state
        x_anom = x_normal.copy()
        n, _ = x_anom.shape
        realized: list[AnomalyEvent] = []

        for event in events:
            delta = self._render_template(
                kind=event.anomaly_type,
                n=n,
                t_start=event.t_start,
                t_end=event.t_end,
                params=event.params,
            )
            x_anom[:, event.node] += delta
            realized.append(event)

        return x_anom, realized

    # Backward-compatible wrapper.
    def inject(
        self,
        x_normal: np.ndarray,
        rng: np.random.Generator,
        graph=None,
        causal_state=None,
    ) -> tuple[np.ndarray, list[AnomalyEvent]]:
        sampled = self.sample_events(n=x_normal.shape[0], d=x_normal.shape[1], rng=rng, graph=graph)
        return self.apply_events(x_normal=x_normal, events=sampled, graph=graph, causal_state=causal_state)
