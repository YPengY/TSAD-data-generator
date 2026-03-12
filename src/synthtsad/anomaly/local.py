from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..config import GeneratorConfig
from ..utils import IntRange, weighted_choice


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
    family: str = "local"
    target_component: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalAnomalyInjector:
    """Stage 3 local/change anomalies with parameter-first templates."""

    _SINGLE_SPIKE_TYPES = {"upward_spike", "downward_spike"}
    _BURST_SPIKE_TYPES = {"continuous_upward_spikes", "continuous_downward_spikes"}
    _WIDE_SPIKE_TYPES = {"wide_upward_spike", "wide_downward_spike"}
    _SUDDEN_SHIFT_TYPES = {"sudden_increase", "sudden_decrease"}
    _ASYMMETRIC_TYPES = {
        "rapid_rise_slow_decline",
        "slow_rise_rapid_decline",
        "rapid_decline_slow_rise",
        "slow_decline_rapid_rise",
    }
    _SPIKE_LEVEL_TYPES = {
        "decrease_after_upward_spike",
        "increase_after_downward_spike",
        "increase_after_upward_spike",
        "decrease_after_downward_spike",
    }

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def _local_config(self):
        return self.config.anomaly.local

    def _placement_policy(self):
        return self.config.anomaly.placement

    def _range_bounds(
        self,
        value: IntRange | dict[str, Any] | int | float,
        *,
        integer: bool,
    ) -> tuple[int | float, int | float]:
        if isinstance(value, IntRange):
            return value.min, value.max
        if isinstance(value, dict) and "min" in value and "max" in value:
            low = value["min"]
            high = value["max"]
        else:
            low = value
            high = value
        if integer:
            return int(low), int(high)
        return float(low), float(high)

    def _sample_int(
        self,
        value: IntRange | dict[str, Any] | int,
        rng: np.random.Generator,
        *,
        low_clip: int | None = None,
        high_clip: int | None = None,
    ) -> int:
        low, high = self._range_bounds(value, integer=True)
        if low_clip is not None:
            low = max(int(low), int(low_clip))
        if high_clip is not None:
            high = min(int(high), int(high_clip))
        if int(high) < int(low):
            high = low
        return int(rng.integers(int(low), int(high) + 1))

    def _sample_float(
        self,
        value: dict[str, Any] | float | int,
        rng: np.random.Generator,
        *,
        low_clip: float | None = None,
        high_clip: float | None = None,
    ) -> float:
        low, high = self._range_bounds(value, integer=False)
        if low_clip is not None:
            low = max(float(low), float(low_clip))
        if high_clip is not None:
            high = min(float(high), float(high_clip))
        if float(high) < float(low):
            high = low
        if float(high) == float(low):
            return float(low)
        return float(rng.uniform(float(low), float(high)))

    def _sample_window(
        self,
        n: int,
        rng: np.random.Generator,
        spec: dict[str, Any],
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> tuple[int, int]:
        family_window = self._local_config().window_length
        base_window = spec.get("window_length", family_window)
        base_min, base_max = self._range_bounds(base_window, integer=True)
        global_max = min(int(base_max), n)
        global_min = min(max(1, int(base_min)), global_max)
        eff_min = global_min if min_len is None else min(max(1, min_len), global_max)
        eff_max = global_max if max_len is None else min(max(1, max_len), global_max)
        eff_min = min(eff_min, eff_max)
        length = int(rng.integers(eff_min, eff_max + 1))
        start = int(rng.integers(0, n - length + 1))
        return start, start + length

    def _sample_spike_params(
        self,
        spec: dict[str, Any],
        t_start: int,
        t_end: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        width_cap = max(2, (t_end - t_start) // 3 + 1)
        return {
            "amplitude": self._sample_float(spec["amplitude"], rng),
            "center": int(rng.integers(t_start, t_end)),
            "half_width": self._sample_int(spec["half_width"], rng, low_clip=1, high_clip=width_cap),
        }

    def _sample_template_spec(
        self,
        kind: str,
        n: int,
        rng: np.random.Generator,
        spec: dict[str, Any],
    ) -> tuple[int, int, dict[str, Any]]:
        if kind == "outlier":
            t0 = int(rng.integers(0, n))
            amplitude = self._sample_float(spec["amplitude"], rng)
            min_abs = float(spec.get("min_abs_amplitude", 0.0))
            if abs(amplitude) < min_abs:
                amplitude = min_abs if amplitude >= 0.0 else -min_abs
            return t0, t0 + 1, {"amplitude": amplitude, "t0": t0}

        if kind in self._SPIKE_LEVEL_TYPES:
            t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=8)
            spike = self._sample_spike_params(spec=spec, t_start=t_start, t_end=t_end, rng=rng)
            shift_start = int(min(n - 1, max(spike["center"], t_end - 1)))
            shift_magnitude = self._sample_float(spec["shift_magnitude"], rng)
            spike["shift_start"] = shift_start
            spike["shift_magnitude"] = shift_magnitude
            return t_start, n, spike

        if kind in self._SINGLE_SPIKE_TYPES:
            t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=5)
            return t_start, t_end, self._sample_spike_params(spec=spec, t_start=t_start, t_end=t_end, rng=rng)

        if kind in self._BURST_SPIKE_TYPES:
            t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=10)
            window = max(2, t_end - t_start)
            _, configured_max = self._range_bounds(spec["spike_count"], integer=True)
            spike_cap = max(2, min(int(configured_max), max(2, window // 2)))
            sampled_count = self._sample_int(spec["spike_count"], rng, low_clip=2, high_clip=spike_cap)
            stride_cap = max(1, min(self._range_bounds(spec["stride"], integer=True)[1], window // max(2, sampled_count)))
            stride = self._sample_int(spec["stride"], rng, low_clip=1, high_clip=int(stride_cap))
            amplitudes = [self._sample_float(spec["amplitude"], rng) for _ in range(sampled_count)]
            width_cap = max(1, min(int(self._range_bounds(spec["half_width"], integer=True)[1]), window // 2 + 1))
            widths = [
                self._sample_int(spec["half_width"], rng, low_clip=1, high_clip=width_cap)
                for _ in range(sampled_count)
            ]
            return t_start, t_end, {
                "spike_count": sampled_count,
                "stride": stride,
                "amplitudes": amplitudes,
                "widths": widths,
                "t0": t_start,
            }

        if kind in self._WIDE_SPIKE_TYPES:
            t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=8)
            window = max(3, t_end - t_start)
            rise_length = self._sample_int(spec["rise_length"], rng, low_clip=1, high_clip=window - 1)
            fall_length = self._sample_int(spec["fall_length"], rng, low_clip=1, high_clip=window - 1)
            return t_start, t_end, {
                "amplitude": self._sample_float(spec["amplitude"], rng),
                "rise_length": rise_length,
                "fall_length": fall_length,
            }

        if kind in self._SUDDEN_SHIFT_TYPES:
            t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=6)
            return t_start, t_end, {
                "amplitude": self._sample_float(spec["amplitude"], rng),
                "t0": int(rng.integers(t_start, t_end)),
                "kappa": self._sample_float(spec["kappa"], rng, low_clip=1e-6),
            }

        if kind in {"plateau", "convex_plateau", "concave_plateau"}:
            t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=6)
            positive_p = float(spec.get("positive_p", 0.5))
            sign = 1.0 if kind == "convex_plateau" else -1.0 if kind == "concave_plateau" else (1.0 if rng.random() < positive_p else -1.0)
            return t_start, t_end, {
                "amplitude": self._sample_float(spec["amplitude"], rng),
                "sign": sign,
            }

        if kind in self._ASYMMETRIC_TYPES:
            t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=8)
            peak = int(rng.integers(t_start + 1, t_end))
            rapid = self._sample_float(spec["rapid_tau"], rng, low_clip=1e-6)
            slow = self._sample_float(spec["slow_tau"], rng, low_clip=1e-6)
            rise_tau = rapid if "rapid_rise" in kind or "slow_decline_rapid_rise" in kind else slow
            fall_tau = rapid if "rapid_decline" in kind or "slow_rise_rapid_decline" in kind else slow
            sign = -1.0 if kind in {"rapid_decline_slow_rise", "slow_decline_rapid_rise"} else 1.0
            return t_start, t_end, {
                "amplitude": self._sample_float(spec["amplitude"], rng),
                "peak": peak,
                "rise_tau": rise_tau,
                "fall_tau": fall_tau,
                "sign": sign,
            }

        if kind == "shake":
            t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=10)
            return t_start, t_end, {
                "amplitude": self._sample_float(spec["amplitude"], rng),
                "freq": self._sample_float(spec["frequency"], rng),
                "phase": self._sample_float(spec["phase"], rng),
            }

        t_start, t_end = self._sample_window(n=n, rng=rng, spec=spec, min_len=5)
        return t_start, t_end, self._sample_spike_params(spec=spec, t_start=t_start, t_end=t_end, rng=rng)

    def _eligible_type_weights(self) -> dict[str, float]:
        local_cfg = self._local_config()
        weights: dict[str, float] = {}
        for kind, spec in local_cfg.per_type.items():
            weight = float(local_cfg.type_weights.get(kind, 0.0))
            if spec.get("enabled", True) and weight > 0.0:
                weights[kind] = weight
        return weights

    def _eligible_nodes(self, d: int) -> list[int]:
        policy = self._local_config().node_policy
        nodes = list(range(d))
        allowed = policy.allowed_nodes
        if allowed is not None:
            allowed_set = {int(node) for node in allowed}
            nodes = [node for node in nodes if node in allowed_set]
        return nodes

    def _can_place(
        self,
        node: int,
        t_start: int,
        t_end: int,
        placements: dict[int, list[tuple[int, int]]],
        counts: dict[int, int],
    ) -> bool:
        policy = self._placement_policy()
        if counts.get(node, 0) >= int(policy.max_events_per_node):
            return False
        if bool(policy.allow_overlap):
            return True
        min_gap = int(policy.min_gap)
        for start, end in placements.get(node, []):
            if not (t_end + min_gap <= start or t_start >= end + min_gap):
                return False
        return True

    def _render_triangular_spike(
        self,
        idx: np.ndarray,
        center: int,
        half_width: int,
        amplitude: float,
        sign: float,
    ) -> np.ndarray:
        shape = np.maximum(1.0 - np.abs(idx - center) / max(1, half_width), 0.0)
        return sign * amplitude * shape

    def _render_wide_spike(
        self,
        idx: np.ndarray,
        t_start: int,
        t_end: int,
        amplitude: float,
        rise_length: int,
        fall_length: int,
        sign: float,
    ) -> np.ndarray:
        delta = np.zeros(idx.size, dtype=float)
        if idx.size == 0:
            return delta
        rise_end = min(t_end, t_start + max(1, rise_length))
        fall_start = max(rise_end, t_end - max(1, fall_length))

        rise_mask = idx < rise_end
        if np.any(rise_mask):
            delta[rise_mask] = amplitude * (idx[rise_mask] - t_start + 1) / max(1, rise_end - t_start)

        plateau_mask = (idx >= rise_end) & (idx < fall_start)
        if np.any(plateau_mask):
            delta[plateau_mask] = amplitude

        fall_mask = idx >= fall_start
        if np.any(fall_mask):
            delta[fall_mask] = amplitude * np.maximum(
                1.0 - (idx[fall_mask] - fall_start + 1) / max(1, t_end - fall_start),
                0.0,
            )

        return sign * delta

    def _render_asymmetric_transient(
        self,
        idx: np.ndarray,
        t_start: int,
        peak: int,
        amplitude: float,
        rise_tau: float,
        fall_tau: float,
        sign: float,
    ) -> np.ndarray:
        delta = np.zeros(idx.size, dtype=float)
        rise_mask = idx <= peak
        if np.any(rise_mask):
            delta[rise_mask] = amplitude * (1.0 - np.exp(-(idx[rise_mask] - t_start) / max(rise_tau, 1e-6)))
        fall_mask = idx > peak
        if np.any(fall_mask):
            delta[fall_mask] = amplitude * np.exp(-(idx[fall_mask] - peak) / max(fall_tau, 1e-6))
        return sign * delta

    def _render_template(self, kind: str, n: int, t_start: int, t_end: int, params: dict[str, Any]) -> np.ndarray:
        delta = np.zeros(n, dtype=float)
        idx = np.arange(max(0, t_start), min(n, t_end))
        rel = np.arange(idx.size)

        if kind in self._SINGLE_SPIKE_TYPES:
            sign = 1.0 if kind == "upward_spike" else -1.0
            delta[idx] = self._render_triangular_spike(
                idx=idx,
                center=int(params["center"]),
                half_width=int(params["half_width"]),
                amplitude=float(params["amplitude"]),
                sign=sign,
            )
            return delta

        if kind in self._BURST_SPIKE_TYPES:
            sign = 1.0 if kind == "continuous_upward_spikes" else -1.0
            spike_count = int(params["spike_count"])
            stride = int(params["stride"])
            t0 = int(params["t0"])
            amplitudes = [float(v) for v in params["amplitudes"]]
            widths = [int(v) for v in params["widths"]]
            for offset in range(spike_count):
                delta[idx] += self._render_triangular_spike(
                    idx=idx,
                    center=t0 + offset * stride,
                    half_width=widths[offset],
                    amplitude=amplitudes[offset],
                    sign=sign,
                )
            return delta

        if kind in self._WIDE_SPIKE_TYPES:
            sign = 1.0 if kind == "wide_upward_spike" else -1.0
            delta[idx] = self._render_wide_spike(
                idx=idx,
                t_start=t_start,
                t_end=t_end,
                amplitude=float(params["amplitude"]),
                rise_length=int(params["rise_length"]),
                fall_length=int(params["fall_length"]),
                sign=sign,
            )
            return delta

        if kind == "outlier":
            t0 = int(params["t0"])
            if 0 <= t0 < n:
                delta[t0] = float(params["amplitude"])
            return delta

        if kind in self._SUDDEN_SHIFT_TYPES:
            sign = 1.0 if kind == "sudden_increase" else -1.0
            t0 = int(params["t0"])
            kappa = float(params["kappa"])
            logits = kappa * (idx - t0)
            delta[idx] = sign * float(params["amplitude"]) * (1.0 / (1.0 + np.exp(-logits)))
            return delta

        if kind in {"plateau", "convex_plateau", "concave_plateau"}:
            phase = np.pi * rel / max(1, idx.size - 1)
            delta[idx] = float(params["sign"]) * float(params["amplitude"]) * 0.5 * (1.0 - np.cos(phase))
            return delta

        if kind in self._ASYMMETRIC_TYPES:
            delta[idx] = self._render_asymmetric_transient(
                idx=idx,
                t_start=t_start,
                peak=int(params["peak"]),
                amplitude=float(params["amplitude"]),
                rise_tau=float(params["rise_tau"]),
                fall_tau=float(params["fall_tau"]),
                sign=float(params["sign"]),
            )
            return delta

        if kind in self._SPIKE_LEVEL_TYPES:
            spike_sign = 1.0 if "upward" in kind else -1.0
            shift_sign = 1.0 if kind in {"increase_after_downward_spike", "increase_after_upward_spike"} else -1.0
            spike = self._render_triangular_spike(
                idx=np.arange(n),
                center=int(params["center"]),
                half_width=int(params["half_width"]),
                amplitude=float(params["amplitude"]),
                sign=spike_sign,
            )
            shift = np.zeros(n, dtype=float)
            shift_start = int(params["shift_start"])
            if shift_start < n:
                shift[shift_start:] = shift_sign * float(params["shift_magnitude"])
            return spike + shift

        freq = float(params.get("freq", 0.2))
        phase = float(params.get("phase", 0.0))
        window = np.sin(np.pi * rel / max(1, idx.size - 1)) ** 2
        delta[idx] = float(params.get("amplitude", 1.0)) * window * np.sin(2 * np.pi * freq * rel + phase)
        return delta

    def sample_events(
        self,
        n: int,
        d: int,
        rng: np.random.Generator,
        graph=None,
    ) -> list[AnomalyEvent]:
        _ = graph
        if d <= 0:
            return []

        events: list[AnomalyEvent] = []
        local_cfg = self._local_config()
        type_weights = self._eligible_type_weights()
        nodes = self._eligible_nodes(d)
        if not type_weights or not nodes:
            return []

        placements: dict[int, list[tuple[int, int]]] = {node: [] for node in nodes}
        counts: dict[int, int] = {node: 0 for node in nodes}
        event_count = local_cfg.events_per_sample.sample(rng)

        for _ in range(event_count):
            placed = False
            for _attempt in range(48):
                kind = weighted_choice(rng, type_weights)
                node = int(rng.choice(nodes))
                spec = local_cfg.per_type[kind]
                t_start, t_end, params = self._sample_template_spec(kind=kind, n=n, rng=rng, spec=spec)
                if not self._can_place(node=node, t_start=t_start, t_end=t_end, placements=placements, counts=counts):
                    continue
                endogenous_p = float(local_cfg.endogenous_p)
                is_endogenous = bool(d > 1 and rng.random() < endogenous_p)
                placements[node].append((t_start, t_end))
                counts[node] += 1
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
                        family="local",
                        target_component=str(local_cfg.target_component),
                    )
                )
                placed = True
                break
            if not placed:
                continue

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

    def inject(
        self,
        x_normal: np.ndarray,
        rng: np.random.Generator,
        graph=None,
        causal_state=None,
    ) -> tuple[np.ndarray, list[AnomalyEvent]]:
        sampled = self.sample_events(n=x_normal.shape[0], d=x_normal.shape[1], rng=rng, graph=graph)
        return self.apply_events(x_normal=x_normal, events=sampled, graph=graph, causal_state=causal_state)
