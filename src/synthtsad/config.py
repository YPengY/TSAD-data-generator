from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import (
    IntRange,
    ensure_int_range,
    ensure_non_negative_float,
    ensure_non_negative_int,
    ensure_positive_int,
    ensure_probability,
    normalize_weights,
)


@dataclass(frozen=True)
class CausalConfig:
    num_nodes: IntRange
    edge_density: float
    max_lag: int
    a_i_bound: float
    bias_std: float
    b_ij_std: float
    alpha_i_min: float
    alpha_i_max: float


@dataclass(frozen=True)
class Stage1Config:
    trend_change_points: IntRange
    trend_slope_scale: float
    arima_noise_scale: float
    arima_p_max: int
    arima_q_max: int
    arima_d: IntRange
    arima_coef_bound: float
    seasonal_atoms: IntRange
    seasonal_amplitude: tuple[float, float]
    period_low: IntRange
    period_high: IntRange
    wavelet_family_weights: dict[str, float]
    wavelet_scale: tuple[float, float]
    wavelet_shift: tuple[float, float]
    wavelet_contrastive_ratio: float
    wavelet_contrastive_params: list[str]
    volatility_windows: IntRange
    volatility_multiplier: tuple[float, float]
    noise_sigma: dict[str, float]


@dataclass(frozen=True)
class NodePolicyConfig:
    mode: str
    allowed_nodes: tuple[int, ...] | None


@dataclass(frozen=True)
class AnomalyPlacementConfig:
    allow_overlap: bool
    min_gap: int
    max_events_per_node: int


@dataclass(frozen=True)
class LocalAnomalyFamilyConfig:
    events_per_sample: IntRange
    window_length: IntRange
    endogenous_p: float
    target_component: str
    node_policy: NodePolicyConfig
    type_weights: dict[str, float]
    per_type: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SeasonalAnomalyFamilyConfig:
    activation_p: float
    events_per_sample: IntRange
    window_length: IntRange
    endogenous_p: float
    target_component: str
    node_policy: NodePolicyConfig
    type_weights: dict[str, float]
    per_type: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class AnomalyConfig:
    placement: AnomalyPlacementConfig
    local: LocalAnomalyFamilyConfig
    seasonal: SeasonalAnomalyFamilyConfig


@dataclass(frozen=True)
class DebugConfig:
    enable_trend: bool
    enable_seasonality: bool
    enable_noise: bool
    enable_causal: bool
    enable_local_anomaly: bool
    enable_seasonal_anomaly: bool


@dataclass(frozen=True)
class GeneratorConfig:
    raw: dict[str, Any]
    num_samples: int
    sequence_length: IntRange
    anomaly_sample_ratio: float
    num_series: IntRange
    seed: int | None
    weights: dict[str, dict[str, float]]
    stage1: Stage1Config
    causal: CausalConfig
    anomaly: AnomalyConfig
    debug: DebugConfig


DEFAULT_WAVELET_FAMILY_WEIGHTS: dict[str, float] = {
    "morlet": 0.22,
    "ricker": 0.20,
    "haar": 0.14,
    "gaus": 0.16,
    "mexh": 0.14,
    "shan": 0.14,
}

LOCAL_ANOMALY_TYPES: list[str] = [
    "upward_spike",
    "downward_spike",
    "continuous_upward_spikes",
    "continuous_downward_spikes",
    "wide_upward_spike",
    "wide_downward_spike",
    "outlier",
    "sudden_increase",
    "sudden_decrease",
    "convex_plateau",
    "concave_plateau",
    "rapid_rise_slow_decline",
    "slow_rise_rapid_decline",
    "rapid_decline_slow_rise",
    "slow_decline_rapid_rise",
    "decrease_after_upward_spike",
    "increase_after_downward_spike",
    "increase_after_upward_spike",
    "decrease_after_downward_spike",
    "shake",
    "plateau",
]

SEASONAL_ANOMALY_TYPES: list[str] = [
    "waveform_inversion",
    "amplitude_scaling",
    "frequency_change",
    "phase_shift",
    "noise_injection",
    "waveform_change",
    "add_harmonic",
    "remove_harmonic",
    "modify_harmonic_phase",
    "modify_modulation_depth",
    "modify_modulation_frequency",
    "modify_modulation_phase",
    "pulse_shift",
    "pulse_width_modulation",
    "wavelet_family_change",
    "wavelet_scale_change",
    "wavelet_shift_change",
    "wavelet_amplitude_change",
    "add_wavelet",
    "remove_wavelet",
]

ROOT_CONFIG_KEYS: set[str] = {
    "num_samples",
    "sequence_length",
    "anomaly_sample_ratio",
    "num_series",
    "seed",
    "weights",
    "stage1",
    "causal",
    "anomaly",
    "debug",
}
WEIGHTS_KEYS: set[str] = {"seasonality_type", "trend_type", "frequency_regime", "noise_level"}
STAGE1_KEYS: set[str] = {"trend", "seasonality", "noise"}
TREND_KEYS: set[str] = {"change_points", "slope_scale", "arima_noise_scale", "arima"}
TREND_ARIMA_KEYS: set[str] = {"p_max", "q_max", "d", "coef_bound"}
SEASONAL_KEYS: set[str] = {"atoms", "amplitude", "base_period", "wavelet"}
BASE_PERIOD_KEYS: set[str] = {"low", "high"}
WAVELET_KEYS: set[str] = {"families", "scale", "shift", "contrastive"}
CONTRASTIVE_KEYS: set[str] = {"ratio", "params"}
NOISE_KEYS: set[str] = {"sigma", "volatility_windows", "volatility_multiplier"}
CAUSAL_KEYS: set[str] = {"num_nodes", "edge_density", "max_lag", "a_i_bound", "bias_std", "b_ij_std", "alpha_i_min", "alpha_i_max"}
DEBUG_KEYS: set[str] = {
    "enable_trend",
    "enable_seasonality",
    "enable_noise",
    "enable_causal",
    "enable_local_anomaly",
    "enable_seasonal_anomaly",
}
ANOMALY_ROOT_KEYS: set[str] = {"defaults", "local", "seasonal"}
ANOMALY_PLACEMENT_KEYS: set[str] = {"allow_overlap", "min_gap", "max_events_per_node"}
ANOMALY_BUDGET_KEYS: set[str] = {"events_per_sample"}
LOCAL_ANOMALY_KEYS: set[str] = {"budget", "defaults", "type_weights", "per_type"}
SEASONAL_ANOMALY_KEYS: set[str] = {"activation_p", "budget", "defaults", "type_weights", "per_type"}
LOCAL_DEFAULT_KEYS: set[str] = {"window_length", "endogenous_p", "target_component", "node_policy"}
SEASONAL_DEFAULT_KEYS: set[str] = {"window_length", "endogenous_p", "target_component", "node_policy"}
NODE_POLICY_KEYS: set[str] = {"mode", "allowed_nodes"}
LOCAL_TARGET_COMPONENTS: tuple[str, ...] = ("observed",)
SEASONAL_TARGET_COMPONENTS: tuple[str, ...] = ("seasonality",)
LOCAL_NODE_POLICY_MODES: tuple[str, ...] = ("uniform",)
SEASONAL_NODE_POLICY_MODES: tuple[str, ...] = ("seasonal_eligible", "uniform")
LEGACY_ANOMALY_KEYS: set[str] = {
    "events_per_sample",
    "seasonal_events_per_sample",
    "window_length",
    "local_types",
    "seasonal_types",
    "p_endogenous",
    "p_endogenous_seasonal",
    "p_use_seasonal_injector",
}


def _default_local_type_weights() -> dict[str, float]:
    return {
        "upward_spike": 1.0,
        "downward_spike": 1.0,
        "continuous_upward_spikes": 0.8,
        "continuous_downward_spikes": 0.8,
        "wide_upward_spike": 0.6,
        "wide_downward_spike": 0.6,
        "outlier": 0.4,
        "sudden_increase": 0.7,
        "sudden_decrease": 0.7,
        "convex_plateau": 0.5,
        "concave_plateau": 0.5,
        "rapid_rise_slow_decline": 0.5,
        "slow_rise_rapid_decline": 0.5,
        "rapid_decline_slow_rise": 0.5,
        "slow_decline_rapid_rise": 0.5,
        "decrease_after_upward_spike": 0.4,
        "increase_after_downward_spike": 0.4,
        "increase_after_upward_spike": 0.4,
        "decrease_after_downward_spike": 0.4,
        "shake": 0.3,
        "plateau": 0.2,
    }


def _default_local_per_type() -> dict[str, dict[str, Any]]:
    return {
        "upward_spike": {
            "enabled": True,
            "window_length": {"min": 5, "max": 24},
            "amplitude": {"min": 0.8, "max": 3.0},
            "half_width": {"min": 1, "max": 6},
        },
        "downward_spike": {
            "enabled": True,
            "window_length": {"min": 5, "max": 24},
            "amplitude": {"min": 0.8, "max": 3.0},
            "half_width": {"min": 1, "max": 6},
        },
        "continuous_upward_spikes": {
            "enabled": True,
            "window_length": {"min": 10, "max": 48},
            "spike_count": {"min": 2, "max": 5},
            "stride": {"min": 1, "max": 8},
            "amplitude": {"min": 0.6, "max": 2.4},
            "half_width": {"min": 1, "max": 4},
        },
        "continuous_downward_spikes": {
            "enabled": True,
            "window_length": {"min": 10, "max": 48},
            "spike_count": {"min": 2, "max": 5},
            "stride": {"min": 1, "max": 8},
            "amplitude": {"min": 0.6, "max": 2.4},
            "half_width": {"min": 1, "max": 4},
        },
        "wide_upward_spike": {
            "enabled": True,
            "window_length": {"min": 8, "max": 40},
            "amplitude": {"min": 0.8, "max": 3.0},
            "rise_length": {"min": 1, "max": 10},
            "fall_length": {"min": 1, "max": 10},
        },
        "wide_downward_spike": {
            "enabled": True,
            "window_length": {"min": 8, "max": 40},
            "amplitude": {"min": 0.8, "max": 3.0},
            "rise_length": {"min": 1, "max": 10},
            "fall_length": {"min": 1, "max": 10},
        },
        "outlier": {
            "enabled": True,
            "amplitude": {"min": -3.0, "max": 3.0},
            "min_abs_amplitude": 0.3,
        },
        "sudden_increase": {
            "enabled": True,
            "window_length": {"min": 6, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
            "kappa": {"min": 0.1, "max": 0.4},
        },
        "sudden_decrease": {
            "enabled": True,
            "window_length": {"min": 6, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
            "kappa": {"min": 0.1, "max": 0.4},
        },
        "convex_plateau": {
            "enabled": True,
            "window_length": {"min": 6, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
        },
        "concave_plateau": {
            "enabled": True,
            "window_length": {"min": 6, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
        },
        "plateau": {
            "enabled": True,
            "window_length": {"min": 6, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
            "positive_p": 0.5,
        },
        "rapid_rise_slow_decline": {
            "enabled": True,
            "window_length": {"min": 8, "max": 60},
            "amplitude": {"min": 0.8, "max": 3.0},
            "rapid_tau": {"min": 1.0, "max": 3.0},
            "slow_tau": {"min": 4.0, "max": 12.0},
        },
        "slow_rise_rapid_decline": {
            "enabled": True,
            "window_length": {"min": 8, "max": 60},
            "amplitude": {"min": 0.8, "max": 3.0},
            "rapid_tau": {"min": 1.0, "max": 3.0},
            "slow_tau": {"min": 4.0, "max": 12.0},
        },
        "rapid_decline_slow_rise": {
            "enabled": True,
            "window_length": {"min": 8, "max": 60},
            "amplitude": {"min": 0.8, "max": 3.0},
            "rapid_tau": {"min": 1.0, "max": 3.0},
            "slow_tau": {"min": 4.0, "max": 12.0},
        },
        "slow_decline_rapid_rise": {
            "enabled": True,
            "window_length": {"min": 8, "max": 60},
            "amplitude": {"min": 0.8, "max": 3.0},
            "rapid_tau": {"min": 1.0, "max": 3.0},
            "slow_tau": {"min": 4.0, "max": 12.0},
        },
        "decrease_after_upward_spike": {
            "enabled": True,
            "window_length": {"min": 8, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
            "half_width": {"min": 1, "max": 6},
            "shift_magnitude": {"min": 0.5, "max": 2.5},
        },
        "increase_after_downward_spike": {
            "enabled": True,
            "window_length": {"min": 8, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
            "half_width": {"min": 1, "max": 6},
            "shift_magnitude": {"min": 0.5, "max": 2.5},
        },
        "increase_after_upward_spike": {
            "enabled": True,
            "window_length": {"min": 8, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
            "half_width": {"min": 1, "max": 6},
            "shift_magnitude": {"min": 0.5, "max": 2.5},
        },
        "decrease_after_downward_spike": {
            "enabled": True,
            "window_length": {"min": 8, "max": 80},
            "amplitude": {"min": 0.8, "max": 3.0},
            "half_width": {"min": 1, "max": 6},
            "shift_magnitude": {"min": 0.5, "max": 2.5},
        },
        "shake": {
            "enabled": True,
            "window_length": {"min": 10, "max": 60},
            "amplitude": {"min": 0.5, "max": 2.2},
            "frequency": {"min": 0.1, "max": 0.35},
            "phase": {"min": 0.0, "max": 2.0 * 3.141592653589793},
        },
    }


def _default_seasonal_type_weights() -> dict[str, float]:
    return {
        "waveform_inversion": 0.8,
        "amplitude_scaling": 1.0,
        "frequency_change": 0.7,
        "phase_shift": 0.8,
        "noise_injection": 0.5,
        "waveform_change": 0.4,
        "add_harmonic": 0.5,
        "remove_harmonic": 0.4,
        "modify_harmonic_phase": 0.6,
        "modify_modulation_depth": 0.4,
        "modify_modulation_frequency": 0.4,
        "modify_modulation_phase": 0.4,
        "pulse_shift": 0.4,
        "pulse_width_modulation": 0.4,
        "wavelet_family_change": 0.4,
        "wavelet_scale_change": 0.6,
        "wavelet_shift_change": 0.6,
        "wavelet_amplitude_change": 0.6,
        "add_wavelet": 0.3,
        "remove_wavelet": 0.3,
    }


def _default_seasonal_per_type() -> dict[str, dict[str, Any]]:
    periodic = ["sine", "square", "triangle"]
    all_types = ["sine", "square", "triangle", "wavelet"]
    return {
        "waveform_inversion": {
            "enabled": True,
            "applies_to": list(all_types),
            "window_length": {"min": 8, "max": 80},
        },
        "amplitude_scaling": {
            "enabled": True,
            "applies_to": list(all_types),
            "window_length": {"min": 8, "max": 80},
            "scale": {"min": 0.35, "max": 2.2},
        },
        "frequency_change": {
            "enabled": True,
            "applies_to": list(all_types),
            "window_length": {"min": 8, "max": 80},
            "factor": {"min": 0.5, "max": 1.9},
        },
        "phase_shift": {
            "enabled": True,
            "applies_to": list(periodic),
            "window_length": {"min": 8, "max": 80},
            "delta_phase": {"min": 0.4712, "max": 4.7124},
        },
        "noise_injection": {
            "enabled": True,
            "applies_to": list(all_types),
            "window_length": {"min": 8, "max": 80},
            "noise_scale": {"min": 0.2, "max": 1.1},
        },
        "waveform_change": {
            "enabled": True,
            "applies_to": list(periodic),
            "window_length": {"min": 8, "max": 80},
            "target_type_weights": {
                "sine": 0.4,
                "square": 0.3,
                "triangle": 0.3,
            },
        },
        "add_harmonic": {
            "enabled": True,
            "applies_to": list(periodic),
            "window_length": {"min": 8, "max": 80},
            "order": {"min": 2, "max": 4},
            "amplitude_scale": {"min": 0.25, "max": 0.8},
            "phase": {"min": 0.0, "max": 2.0 * 3.141592653589793},
        },
        "remove_harmonic": {
            "enabled": True,
            "applies_to": list(periodic),
            "window_length": {"min": 8, "max": 80},
        },
        "modify_harmonic_phase": {
            "enabled": True,
            "applies_to": list(periodic),
            "window_length": {"min": 8, "max": 80},
            "delta_phase": {"min": 0.4712, "max": 4.7124},
        },
        "modify_modulation_depth": {
            "enabled": True,
            "applies_to": list(periodic),
            "window_length": {"min": 8, "max": 80},
            "depth": {"min": 0.05, "max": 0.95},
        },
        "modify_modulation_frequency": {
            "enabled": True,
            "applies_to": list(periodic),
            "window_length": {"min": 8, "max": 80},
            "factor": {"min": 0.15, "max": 2.0},
        },
        "modify_modulation_phase": {
            "enabled": True,
            "applies_to": list(periodic),
            "window_length": {"min": 8, "max": 80},
            "phase": {"min": 0.0, "max": 2.0 * 3.141592653589793},
        },
        "pulse_shift": {
            "enabled": True,
            "applies_to": ["square", "triangle"],
            "window_length": {"min": 8, "max": 80},
            "delta_cycle": {"min": -0.35, "max": 0.35},
        },
        "pulse_width_modulation": {
            "enabled": True,
            "applies_to": ["square", "triangle"],
            "window_length": {"min": 8, "max": 80},
            "factor": {"min": 0.5, "max": 1.6},
        },
        "wavelet_family_change": {
            "enabled": True,
            "applies_to": ["wavelet"],
            "window_length": {"min": 8, "max": 80},
            "target_family_weights": dict(DEFAULT_WAVELET_FAMILY_WEIGHTS),
        },
        "wavelet_scale_change": {
            "enabled": True,
            "applies_to": ["wavelet"],
            "window_length": {"min": 8, "max": 80},
            "factor": {"min": 0.6, "max": 1.8},
        },
        "wavelet_shift_change": {
            "enabled": True,
            "applies_to": ["wavelet"],
            "window_length": {"min": 8, "max": 80},
            "delta_shift": {"min": -0.35, "max": 0.35},
        },
        "wavelet_amplitude_change": {
            "enabled": True,
            "applies_to": ["wavelet"],
            "window_length": {"min": 8, "max": 80},
            "factor": {"min": 0.5, "max": 1.8},
        },
        "add_wavelet": {
            "enabled": True,
            "applies_to": ["wavelet"],
            "window_length": {"min": 8, "max": 80},
            "family_weights": dict(DEFAULT_WAVELET_FAMILY_WEIGHTS),
            "period": {"min": 6.0, "max": 120.0},
            "amplitude": {"min": 0.2, "max": 2.0},
            "phase": {"min": 0.0, "max": 2.0 * 3.141592653589793},
            "scale": {"min": 0.08, "max": 0.35},
            "shift": {"min": 0.0, "max": 1.0},
        },
        "remove_wavelet": {
            "enabled": True,
            "applies_to": ["wavelet"],
            "window_length": {"min": 8, "max": 80},
        },
    }


def _default_anomaly_config() -> dict[str, Any]:
    return {
        "defaults": {
            "allow_overlap": False,
            "min_gap": 0,
            "max_events_per_node": 2,
        },
        "local": {
            "budget": {"events_per_sample": {"min": 1, "max": 3}},
            "defaults": {
                "window_length": {"min": 8, "max": 80},
                "endogenous_p": 0.5,
                "target_component": "observed",
                "node_policy": {"mode": "uniform"},
            },
            "type_weights": _default_local_type_weights(),
            "per_type": _default_local_per_type(),
        },
        "seasonal": {
            "activation_p": 0.4,
            "budget": {"events_per_sample": {"min": 1, "max": 2}},
            "defaults": {
                "window_length": {"min": 8, "max": 80},
                "endogenous_p": 0.25,
                "target_component": "seasonality",
                "node_policy": {"mode": "seasonal_eligible"},
            },
            "type_weights": _default_seasonal_type_weights(),
            "per_type": _default_seasonal_per_type(),
        },
    }


DEFAULT_CONFIG: dict[str, Any] = {
    "num_samples": 100,
    "sequence_length": {"min": 100, "max": 1000},
    "anomaly_sample_ratio": 0.7,
    "num_series": {"min": 2, "max": 8},
    "seed": 42,
    "weights": {
        "seasonality_type": {
            "none": 0.30,
            "sine": 0.30,
            "square": 0.05,
            "triangle": 0.05,
            "wavelet": 0.30,
        },
        "trend_type": {
            "decrease": 0.20,
            "increase": 0.20,
            "keep_steady": 0.20,
            "multiple": 0.30,
            "arima": 0.10,
        },
        "frequency_regime": {"high": 0.50, "low": 0.50},
        "noise_level": {
            "almost_none": 0.25,
            "low": 0.25,
            "moderate": 0.25,
            "high": 0.25,
        },
    },
    "stage1": {
        "trend": {
            "change_points": {"min": 1, "max": 4},
            "slope_scale": 0.02,
            "arima_noise_scale": 0.05,
            "arima": {
                "p_max": 2,
                "q_max": 2,
                "d": {"min": 1, "max": 2},
                "coef_bound": 0.6,
            },
        },
        "seasonality": {
            "atoms": {"min": 1, "max": 3},
            "amplitude": {"min": 0.2, "max": 2.0},
            "base_period": {
                "low": {"min": 30, "max": 120},
                "high": {"min": 6, "max": 30},
            },
            "wavelet": {
                "families": dict(DEFAULT_WAVELET_FAMILY_WEIGHTS),
                "scale": {"min": 0.08, "max": 0.35},
                "shift": {"min": 0.0, "max": 1.0},
                "contrastive": {
                    "ratio": 0.25,
                    "params": ["family", "scale", "shift"],
                },
            },
        },
        "noise": {
            "sigma": {
                "almost_none": 0.01,
                "low": 0.05,
                "moderate": 0.10,
                "high": 0.20,
            },
            "volatility_windows": {"min": 0, "max": 3},
            "volatility_multiplier": {"min": 0.1, "max": 1.0},
        },
    },
    "causal": {
        "num_nodes": {"min": 1, "max": 20},
        "edge_density": 0.12,
        "max_lag": 12,
        "a_i_bound": 0.8,
        "bias_std": 0.1,
        "b_ij_std": 0.35,
        "alpha_i_min": 0.1,
        "alpha_i_max": 0.9,
    },
    "anomaly": _default_anomaly_config(),
    "debug": {
        "enable_trend": True,
        "enable_seasonality": True,
        "enable_noise": True,
        "enable_causal": True,
        "enable_local_anomaly": True,
        "enable_seasonal_anomaly": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_raw(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyYAML is required for .yaml/.yml config files.") from exc
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config format: {path}")


def _ensure_generic_range(raw: Any, name: str) -> dict[str, int | float]:
    if not isinstance(raw, dict) or set(raw.keys()) != {"min", "max"}:
        raise ValueError(f"{name} must have min/max")
    low = raw["min"]
    high = raw["max"]
    if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f"{name} must use numeric min/max")
    if float(high) < float(low):
        raise ValueError(f"{name}.max must be >= {name}.min")
    return {"min": low, "max": high}


def _ensure_non_negative_weight_map(
    raw: Any,
    name: str,
    allowed_keys: set[str] | None = None,
) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        skey = str(key)
        if allowed_keys is not None and skey not in allowed_keys:
            raise ValueError(f"{name} contains unsupported key: {skey}")
        normalized[skey] = ensure_non_negative_float(value, f"{name}.{skey}")
    return normalized


def _ensure_allowed_keys(raw: Any, name: str, allowed_keys: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    extras = set(raw.keys()) - allowed_keys
    if extras:
        raise ValueError(f"{name} contains unsupported keys: {sorted(extras)}")
    return raw


def _normalize_node_policy(raw: Any, name: str, allowed_modes: set[str]) -> dict[str, Any]:
    raw = _ensure_allowed_keys(raw, name, NODE_POLICY_KEYS)
    mode = str(raw.get("mode", "uniform"))
    if mode not in allowed_modes:
        raise ValueError(f"{name}.mode must be one of {sorted(allowed_modes)}, got {mode}")
    allowed_nodes = raw.get("allowed_nodes")
    if allowed_nodes is not None:
        if not isinstance(allowed_nodes, list):
            raise ValueError(f"{name}.allowed_nodes must be a list of ints or null")
        cleaned_nodes = [ensure_non_negative_int(value, f"{name}.allowed_nodes[{index}]") for index, value in enumerate(allowed_nodes)]
    else:
        cleaned_nodes = None
    return {"mode": mode, "allowed_nodes": cleaned_nodes}


def _normalize_type_specs(
    raw: Any,
    name: str,
    allowed_types: list[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    allowed_set = set(allowed_types)
    extras = set(raw.keys()) - allowed_set
    if extras:
        raise ValueError(f"{name} contains unsupported types: {sorted(extras)}")

    specs: dict[str, dict[str, Any]] = {}
    for kind in allowed_types:
        spec = raw.get(kind)
        if not isinstance(spec, dict):
            raise ValueError(f"{name}.{kind} must be a mapping")
        cleaned: dict[str, Any] = {}
        for key, value in spec.items():
            field_name = f"{name}.{kind}.{key}"
            if key == "enabled":
                cleaned[key] = bool(value)
                continue
            if key == "window_length":
                cleaned[key] = ensure_int_range(value, field_name, min_value=1)
                continue
            if key == "applies_to":
                if not isinstance(value, list) or not value:
                    raise ValueError(f"{field_name} must be a non-empty list")
                cleaned[key] = [str(item) for item in value]
                invalid = [item for item in cleaned[key] if item not in {"sine", "square", "triangle", "wavelet"}]
                if invalid:
                    raise ValueError(f"{field_name} contains unsupported values: {invalid}")
                continue
            if key == "target_type_weights":
                cleaned[key] = _ensure_non_negative_weight_map(
                    value,
                    field_name,
                    {"sine", "square", "triangle"},
                )
                continue
            if key in {"target_family_weights", "family_weights"}:
                cleaned[key] = _ensure_non_negative_weight_map(
                    value,
                    field_name,
                    set(DEFAULT_WAVELET_FAMILY_WEIGHTS),
                )
                continue
            if isinstance(value, dict) and set(value.keys()) == {"min", "max"}:
                cleaned[key] = _ensure_generic_range(value, field_name)
                continue
            if isinstance(value, list):
                cleaned[key] = [item for item in value]
                continue
            if isinstance(value, (int, float, str, bool)):
                cleaned[key] = value
                continue
            raise ValueError(f"Unsupported value for {field_name}")
        cleaned.setdefault("enabled", True)
        specs[kind] = cleaned
    return specs


def _legacy_type_weights(selected: Any, allowed_types: list[str]) -> dict[str, float]:
    selected_set = {str(value) for value in selected} if isinstance(selected, list) else set()
    return {kind: 1.0 if kind in selected_set else 0.0 for kind in allowed_types}


def _copy_legacy_window_length(target: dict[str, Any], window_length: Any) -> None:
    target.setdefault("defaults", {})["window_length"] = window_length
    target.setdefault("per_type", {})
    for spec in target["per_type"].values():
        if "window_length" in spec:
            spec["window_length"] = window_length


def _migrate_legacy_anomaly_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("anomaly must be a mapping")

    normalized = json.loads(json.dumps(_default_anomaly_config()))
    for key in ["defaults", "local", "seasonal"]:
        if key in raw and isinstance(raw[key], dict):
            normalized[key] = _deep_merge(normalized[key], raw[key])
    if "events_per_sample" in raw:
        normalized["local"]["budget"]["events_per_sample"] = raw["events_per_sample"]
    if "seasonal_events_per_sample" in raw:
        normalized["seasonal"]["budget"]["events_per_sample"] = raw["seasonal_events_per_sample"]
    if "window_length" in raw:
        _copy_legacy_window_length(normalized["local"], raw["window_length"])
        _copy_legacy_window_length(normalized["seasonal"], raw["window_length"])
    if "local_types" in raw:
        normalized["local"]["type_weights"] = _legacy_type_weights(raw["local_types"], LOCAL_ANOMALY_TYPES)
    if "seasonal_types" in raw:
        normalized["seasonal"]["type_weights"] = _legacy_type_weights(raw["seasonal_types"], SEASONAL_ANOMALY_TYPES)
    if "p_endogenous" in raw:
        normalized["local"]["defaults"]["endogenous_p"] = raw["p_endogenous"]
    if "p_endogenous_seasonal" in raw:
        normalized["seasonal"]["defaults"]["endogenous_p"] = raw["p_endogenous_seasonal"]
    if "p_use_seasonal_injector" in raw:
        normalized["seasonal"]["activation_p"] = raw["p_use_seasonal_injector"]
    return normalized


def migrate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = json.loads(json.dumps(raw))
    if "num_features" in migrated and "num_series" not in migrated:
        multivariate = bool(migrated.get("multivariate_flag", True))
        migrated["num_series"] = migrated["num_features"] if multivariate else {"min": 1, "max": 1}
    if "num_features" in migrated:
        migrated.pop("num_features", None)
    if "multivariate_flag" in migrated:
        migrated.pop("multivariate_flag", None)
    if "anomaly" in migrated and isinstance(migrated["anomaly"], dict):
        anomaly_raw = migrated["anomaly"]
        if LEGACY_ANOMALY_KEYS & set(anomaly_raw.keys()):
            migrated["anomaly"] = _migrate_legacy_anomaly_schema(anomaly_raw)
    return migrated


def _normalize_anomaly_schema(raw: Any) -> dict[str, Any]:
    raw = _ensure_allowed_keys(raw, "anomaly", ANOMALY_ROOT_KEYS)

    normalized = json.loads(json.dumps(_default_anomaly_config()))
    if "defaults" in raw:
        defaults_raw = _ensure_allowed_keys(raw["defaults"], "anomaly.defaults", ANOMALY_PLACEMENT_KEYS)
        normalized["defaults"] = _deep_merge(normalized["defaults"], defaults_raw)
    if "local" in raw:
        local_raw = _ensure_allowed_keys(raw["local"], "anomaly.local", LOCAL_ANOMALY_KEYS)
        if "budget" in local_raw:
            _ensure_allowed_keys(local_raw["budget"], "anomaly.local.budget", ANOMALY_BUDGET_KEYS)
        if "defaults" in local_raw:
            defaults_raw = _ensure_allowed_keys(local_raw["defaults"], "anomaly.local.defaults", LOCAL_DEFAULT_KEYS)
            if "node_policy" in defaults_raw:
                _ensure_allowed_keys(defaults_raw["node_policy"], "anomaly.local.defaults.node_policy", NODE_POLICY_KEYS)
        normalized["local"] = _deep_merge(normalized["local"], local_raw)
    if "seasonal" in raw:
        seasonal_raw = _ensure_allowed_keys(raw["seasonal"], "anomaly.seasonal", SEASONAL_ANOMALY_KEYS)
        if "budget" in seasonal_raw:
            _ensure_allowed_keys(seasonal_raw["budget"], "anomaly.seasonal.budget", ANOMALY_BUDGET_KEYS)
        if "defaults" in seasonal_raw:
            defaults_raw = _ensure_allowed_keys(seasonal_raw["defaults"], "anomaly.seasonal.defaults", SEASONAL_DEFAULT_KEYS)
            if "node_policy" in defaults_raw:
                _ensure_allowed_keys(defaults_raw["node_policy"], "anomaly.seasonal.defaults.node_policy", NODE_POLICY_KEYS)
        normalized["seasonal"] = _deep_merge(normalized["seasonal"], seasonal_raw)
    return normalized


def _build_config(raw: dict[str, Any]) -> GeneratorConfig:
    _ensure_allowed_keys(raw, "config", ROOT_CONFIG_KEYS)
    _ensure_allowed_keys(raw["weights"], "weights", WEIGHTS_KEYS)
    _ensure_allowed_keys(raw["stage1"], "stage1", STAGE1_KEYS)
    _ensure_allowed_keys(raw["stage1"]["trend"], "stage1.trend", TREND_KEYS)
    _ensure_allowed_keys(raw["stage1"]["trend"]["arima"], "stage1.trend.arima", TREND_ARIMA_KEYS)
    _ensure_allowed_keys(raw["stage1"]["seasonality"], "stage1.seasonality", SEASONAL_KEYS)
    _ensure_allowed_keys(raw["stage1"]["seasonality"]["base_period"], "stage1.seasonality.base_period", BASE_PERIOD_KEYS)
    _ensure_allowed_keys(raw["stage1"]["seasonality"]["wavelet"], "stage1.seasonality.wavelet", WAVELET_KEYS)
    _ensure_allowed_keys(
        raw["stage1"]["seasonality"]["wavelet"]["contrastive"],
        "stage1.seasonality.wavelet.contrastive",
        CONTRASTIVE_KEYS,
    )
    _ensure_allowed_keys(raw["stage1"]["noise"], "stage1.noise", NOISE_KEYS)
    _ensure_allowed_keys(raw["causal"], "causal", CAUSAL_KEYS)
    _ensure_allowed_keys(raw["debug"], "debug", DEBUG_KEYS)
    weights = {
        key: normalize_weights(raw["weights"][key])
        for key in ["seasonality_type", "trend_type", "frequency_regime", "noise_level"]
    }

    stage1_raw = raw["stage1"]
    trend_raw = stage1_raw["trend"]
    season_raw = stage1_raw["seasonality"]
    wavelet_raw = season_raw["wavelet"]
    contrastive_raw = wavelet_raw["contrastive"]
    noise_raw = stage1_raw["noise"]

    wavelet_scale = (float(wavelet_raw["scale"]["min"]), float(wavelet_raw["scale"]["max"]))
    if wavelet_scale[0] <= 0.0 or wavelet_scale[1] < wavelet_scale[0]:
        raise ValueError(f"Invalid stage1.seasonality.wavelet.scale range: {wavelet_scale}")

    wavelet_shift = (float(wavelet_raw["shift"]["min"]), float(wavelet_raw["shift"]["max"]))
    if wavelet_shift[1] < wavelet_shift[0]:
        raise ValueError(f"Invalid stage1.seasonality.wavelet.shift range: {wavelet_shift}")

    wavelet_contrastive_ratio = float(contrastive_raw["ratio"])
    if wavelet_contrastive_ratio < 0.0 or wavelet_contrastive_ratio > 1.0:
        raise ValueError(
            "stage1.seasonality.wavelet.contrastive.ratio must be in [0, 1], "
            f"got {wavelet_contrastive_ratio}"
        )

    wavelet_contrastive_params = [str(v) for v in contrastive_raw["params"]]
    allowed_contrastive_params = {"family", "scale", "shift"}
    invalid_params = [v for v in wavelet_contrastive_params if v not in allowed_contrastive_params]
    if invalid_params:
        raise ValueError(
            "stage1.seasonality.wavelet.contrastive.params contains unsupported values: "
            f"{invalid_params}"
        )
    if not wavelet_contrastive_params:
        raise ValueError("stage1.seasonality.wavelet.contrastive.params must not be empty")

    seasonal_amplitude = (
        float(season_raw["amplitude"]["min"]),
        float(season_raw["amplitude"]["max"]),
    )
    if seasonal_amplitude[0] < 0.0 or seasonal_amplitude[1] < seasonal_amplitude[0]:
        raise ValueError(
            "Invalid stage1.seasonality.amplitude range: "
            f"{seasonal_amplitude}"
        )

    volatility_multiplier = (
        float(noise_raw["volatility_multiplier"]["min"]),
        float(noise_raw["volatility_multiplier"]["max"]),
    )
    if volatility_multiplier[0] < 0.0 or volatility_multiplier[1] < volatility_multiplier[0]:
        raise ValueError(
            "Invalid stage1.noise.volatility_multiplier range: "
            f"{volatility_multiplier}"
        )

    noise_sigma = {str(k): float(v) for k, v in noise_raw["sigma"].items()}
    if not noise_sigma:
        raise ValueError("stage1.noise.sigma must not be empty")
    invalid_noise_sigma = {k: v for k, v in noise_sigma.items() if v < 0.0}
    if invalid_noise_sigma:
        raise ValueError(
            "stage1.noise.sigma values must be >= 0, got "
            f"{invalid_noise_sigma}"
        )

    stage1 = Stage1Config(
        trend_change_points=ensure_int_range(trend_raw["change_points"], "stage1.trend.change_points"),
        trend_slope_scale=ensure_non_negative_float(trend_raw["slope_scale"], "stage1.trend.slope_scale"),
        arima_noise_scale=ensure_non_negative_float(
            trend_raw["arima_noise_scale"],
            "stage1.trend.arima_noise_scale",
        ),
        arima_p_max=ensure_non_negative_int(trend_raw["arima"]["p_max"], "stage1.trend.arima.p_max"),
        arima_q_max=ensure_non_negative_int(trend_raw["arima"]["q_max"], "stage1.trend.arima.q_max"),
        arima_d=ensure_int_range(trend_raw["arima"]["d"], "stage1.trend.arima.d"),
        arima_coef_bound=ensure_non_negative_float(
            trend_raw["arima"]["coef_bound"],
            "stage1.trend.arima.coef_bound",
        ),
        seasonal_atoms=ensure_int_range(season_raw["atoms"], "stage1.seasonality.atoms"),
        seasonal_amplitude=seasonal_amplitude,
        period_low=ensure_int_range(season_raw["base_period"]["low"], "stage1.seasonality.base_period.low"),
        period_high=ensure_int_range(season_raw["base_period"]["high"], "stage1.seasonality.base_period.high"),
        wavelet_family_weights=normalize_weights({str(k): float(v) for k, v in wavelet_raw["families"].items()}),
        wavelet_scale=wavelet_scale,
        wavelet_shift=wavelet_shift,
        wavelet_contrastive_ratio=wavelet_contrastive_ratio,
        wavelet_contrastive_params=wavelet_contrastive_params,
        volatility_windows=ensure_int_range(
            noise_raw["volatility_windows"],
            "stage1.noise.volatility_windows",
            min_value=0,
        ),
        volatility_multiplier=volatility_multiplier,
        noise_sigma=noise_sigma,
    )

    causal_raw = raw["causal"]
    alpha_i_min = ensure_probability(causal_raw["alpha_i_min"], "causal.alpha_i_min")
    alpha_i_max = ensure_probability(causal_raw["alpha_i_max"], "causal.alpha_i_max")
    if alpha_i_max < alpha_i_min:
        raise ValueError(
            "causal.alpha_i_max must be >= causal.alpha_i_min, "
            f"got ({alpha_i_min}, {alpha_i_max})"
        )
    causal = CausalConfig(
        num_nodes=ensure_int_range(causal_raw["num_nodes"], "causal.num_nodes"),
        edge_density=ensure_probability(causal_raw["edge_density"], "causal.edge_density"),
        max_lag=ensure_non_negative_int(causal_raw["max_lag"], "causal.max_lag"),
        a_i_bound=ensure_non_negative_float(causal_raw["a_i_bound"], "causal.a_i_bound"),
        bias_std=ensure_non_negative_float(causal_raw["bias_std"], "causal.bias_std"),
        b_ij_std=ensure_non_negative_float(causal_raw["b_ij_std"], "causal.b_ij_std"),
        alpha_i_min=alpha_i_min,
        alpha_i_max=alpha_i_max,
    )

    anomaly_raw = _normalize_anomaly_schema(raw["anomaly"])
    anomaly_defaults_raw = anomaly_raw["defaults"]
    local_raw = anomaly_raw["local"]
    seasonal_raw = anomaly_raw["seasonal"]

    placement = AnomalyPlacementConfig(
        allow_overlap=bool(anomaly_defaults_raw["allow_overlap"]),
        min_gap=ensure_non_negative_int(anomaly_defaults_raw["min_gap"], "anomaly.defaults.min_gap"),
        max_events_per_node=ensure_positive_int(
            anomaly_defaults_raw["max_events_per_node"],
            "anomaly.defaults.max_events_per_node",
        ),
    )

    local_target_component = str(local_raw["defaults"]["target_component"])
    if local_target_component not in LOCAL_TARGET_COMPONENTS:
        raise ValueError(
            "anomaly.local.defaults.target_component must be one of "
            f"{list(LOCAL_TARGET_COMPONENTS)}, got {local_target_component}"
        )
    local_node_policy_raw = _normalize_node_policy(
        local_raw["defaults"].get("node_policy", {"mode": LOCAL_NODE_POLICY_MODES[0]}),
        "anomaly.local.defaults.node_policy",
        set(LOCAL_NODE_POLICY_MODES),
    )
    local = LocalAnomalyFamilyConfig(
        events_per_sample=ensure_int_range(
            local_raw["budget"]["events_per_sample"],
            "anomaly.local.budget.events_per_sample",
        ),
        window_length=ensure_int_range(
            local_raw["defaults"]["window_length"],
            "anomaly.local.defaults.window_length",
            min_value=1,
        ),
        endogenous_p=ensure_probability(
            local_raw["defaults"]["endogenous_p"],
            "anomaly.local.defaults.endogenous_p",
        ),
        target_component=local_target_component,
        node_policy=NodePolicyConfig(
            mode=str(local_node_policy_raw["mode"]),
            allowed_nodes=tuple(local_node_policy_raw["allowed_nodes"]) if local_node_policy_raw["allowed_nodes"] is not None else None,
        ),
        type_weights=_ensure_non_negative_weight_map(
            local_raw["type_weights"],
            "anomaly.local.type_weights",
            set(LOCAL_ANOMALY_TYPES),
        ),
        per_type=_normalize_type_specs(
            local_raw["per_type"],
            "anomaly.local.per_type",
            LOCAL_ANOMALY_TYPES,
        ),
    )

    seasonal_target_component = str(seasonal_raw["defaults"]["target_component"])
    if seasonal_target_component not in SEASONAL_TARGET_COMPONENTS:
        raise ValueError(
            "anomaly.seasonal.defaults.target_component must be one of "
            f"{list(SEASONAL_TARGET_COMPONENTS)}, got {seasonal_target_component}"
        )
    seasonal_node_policy_raw = _normalize_node_policy(
        seasonal_raw["defaults"].get("node_policy", {"mode": SEASONAL_NODE_POLICY_MODES[0]}),
        "anomaly.seasonal.defaults.node_policy",
        set(SEASONAL_NODE_POLICY_MODES),
    )
    seasonal = SeasonalAnomalyFamilyConfig(
        activation_p=ensure_probability(
            seasonal_raw["activation_p"],
            "anomaly.seasonal.activation_p",
        ),
        events_per_sample=ensure_int_range(
            seasonal_raw["budget"]["events_per_sample"],
            "anomaly.seasonal.budget.events_per_sample",
        ),
        window_length=ensure_int_range(
            seasonal_raw["defaults"]["window_length"],
            "anomaly.seasonal.defaults.window_length",
            min_value=1,
        ),
        endogenous_p=ensure_probability(
            seasonal_raw["defaults"]["endogenous_p"],
            "anomaly.seasonal.defaults.endogenous_p",
        ),
        target_component=seasonal_target_component,
        node_policy=NodePolicyConfig(
            mode=str(seasonal_node_policy_raw["mode"]),
            allowed_nodes=tuple(seasonal_node_policy_raw["allowed_nodes"]) if seasonal_node_policy_raw["allowed_nodes"] is not None else None,
        ),
        type_weights=_ensure_non_negative_weight_map(
            seasonal_raw["type_weights"],
            "anomaly.seasonal.type_weights",
            set(SEASONAL_ANOMALY_TYPES),
        ),
        per_type=_normalize_type_specs(
            seasonal_raw["per_type"],
            "anomaly.seasonal.per_type",
            SEASONAL_ANOMALY_TYPES,
        ),
    )

    anomaly = AnomalyConfig(
        placement=placement,
        local=local,
        seasonal=seasonal,
    )

    debug_raw = raw["debug"]
    debug = DebugConfig(
        enable_trend=bool(debug_raw["enable_trend"]),
        enable_seasonality=bool(debug_raw["enable_seasonality"]),
        enable_noise=bool(debug_raw["enable_noise"]),
        enable_causal=bool(debug_raw["enable_causal"]),
        enable_local_anomaly=bool(debug_raw["enable_local_anomaly"]),
        enable_seasonal_anomaly=bool(debug_raw["enable_seasonal_anomaly"]),
    )

    num_series = ensure_int_range(raw["num_series"], "num_series")

    if num_series.min < causal.num_nodes.min or num_series.max > causal.num_nodes.max:
        raise ValueError(
            "num_series range must stay within causal.num_nodes range, "
            f"got num_series={num_series} and causal.num_nodes={causal.num_nodes}"
        )

    return GeneratorConfig(
        raw=raw,
        num_samples=ensure_positive_int(raw["num_samples"], "num_samples"),
        sequence_length=ensure_int_range(raw["sequence_length"], "sequence_length"),
        anomaly_sample_ratio=ensure_probability(raw["anomaly_sample_ratio"], "anomaly_sample_ratio"),
        num_series=num_series,
        seed=int(raw["seed"]) if raw.get("seed") is not None else None,
        weights=weights,
        stage1=stage1,
        causal=causal,
        anomaly=anomaly,
        debug=debug,
    )


def load_config(path: Path) -> GeneratorConfig:
    user_raw = _load_raw(path)
    return load_config_from_raw(user_raw)


def load_config_from_raw(raw: dict[str, Any]) -> GeneratorConfig:
    incoming = json.loads(json.dumps(raw))
    merged = _deep_merge(DEFAULT_CONFIG, incoming)
    merged = json.loads(json.dumps(merged))
    return _build_config(merged)
