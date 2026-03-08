from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import IntRange, ensure_int_range, normalize_weights


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
    seasonal_atoms: IntRange
    seasonal_amplitude: tuple[float, float]
    period_low: IntRange
    period_high: IntRange
    volatility_windows: IntRange
    volatility_multiplier: tuple[float, float]
    noise_sigma: dict[str, float]


@dataclass(frozen=True)
class AnomalyConfig:
    events_per_sample: IntRange
    window_length: IntRange
    local_types: list[str]
    seasonal_types: list[str]
    p_endogenous: float
    p_use_seasonal_injector: float


@dataclass(frozen=True)
class GeneratorConfig:
    raw: dict[str, Any]
    num_samples: int
    sequence_length: IntRange
    anomaly_sample_ratio: float
    multivariate_flag: bool
    num_features: IntRange
    seed: int | None
    weights: dict[str, dict[str, float]]
    stage1: Stage1Config
    causal: CausalConfig
    anomaly: AnomalyConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "num_samples": 100,
    "sequence_length": {"min": 100, "max": 1000},
    "anomaly_sample_ratio": 0.7,
    "multivariate_flag": True,
    "num_features": {"min": 2, "max": 8},
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
        },
        "seasonality": {
            "atoms": {"min": 1, "max": 3},
            "amplitude": {"min": 0.2, "max": 2.0},
            "base_period": {
                "low": {"min": 30, "max": 120},
                "high": {"min": 6, "max": 30},
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
    "anomaly": {
        "events_per_sample": {"min": 1, "max": 3},
        "window_length": {"min": 8, "max": 80},
        "local_types": [
            "upward_spike",
            "downward_spike",
            "sudden_increase",
            "sudden_decrease",
            "shake",
            "plateau",
        ],
        "seasonal_types": [
            "waveform_inversion",
            "amplitude_scaling",
            "frequency_change",
            "phase_shift",
            "noise_injection",
        ],
        "p_endogenous": 0.5,
        "p_use_seasonal_injector": 0.4,
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


def _build_config(raw: dict[str, Any]) -> GeneratorConfig:
    weights = {
        key: normalize_weights(raw["weights"][key])
        for key in ["seasonality_type", "trend_type", "frequency_regime", "noise_level"]
    }

    stage1_raw = raw["stage1"]
    trend_raw = stage1_raw["trend"]
    season_raw = stage1_raw["seasonality"]
    noise_raw = stage1_raw["noise"]

    stage1 = Stage1Config(
        trend_change_points=ensure_int_range(trend_raw["change_points"], "stage1.trend.change_points"),
        trend_slope_scale=float(trend_raw["slope_scale"]),
        arima_noise_scale=float(trend_raw["arima_noise_scale"]),
        seasonal_atoms=ensure_int_range(season_raw["atoms"], "stage1.seasonality.atoms"),
        seasonal_amplitude=(float(season_raw["amplitude"]["min"]), float(season_raw["amplitude"]["max"])),
        period_low=ensure_int_range(season_raw["base_period"]["low"], "stage1.seasonality.base_period.low"),
        period_high=ensure_int_range(season_raw["base_period"]["high"], "stage1.seasonality.base_period.high"),
        volatility_windows=ensure_int_range(
            noise_raw["volatility_windows"],
            "stage1.noise.volatility_windows",
            min_value=0,
        ),
        volatility_multiplier=(
            float(noise_raw["volatility_multiplier"]["min"]),
            float(noise_raw["volatility_multiplier"]["max"]),
        ),
        noise_sigma={k: float(v) for k, v in noise_raw["sigma"].items()},
    )

    causal_raw = raw["causal"]
    causal = CausalConfig(
        num_nodes=ensure_int_range(causal_raw["num_nodes"], "causal.num_nodes"),
        edge_density=float(causal_raw["edge_density"]),
        max_lag=int(causal_raw["max_lag"]),
        a_i_bound=float(causal_raw["a_i_bound"]),
        bias_std=float(causal_raw["bias_std"]),
        b_ij_std=float(causal_raw["b_ij_std"]),
        alpha_i_min=float(causal_raw["alpha_i_min"]),
        alpha_i_max=float(causal_raw["alpha_i_max"]),
    )

    anomaly_raw = raw["anomaly"]
    anomaly = AnomalyConfig(
        events_per_sample=ensure_int_range(anomaly_raw["events_per_sample"], "anomaly.events_per_sample"),
        window_length=ensure_int_range(anomaly_raw["window_length"], "anomaly.window_length"),
        local_types=[str(v) for v in anomaly_raw["local_types"]],
        seasonal_types=[str(v) for v in anomaly_raw["seasonal_types"]],
        p_endogenous=float(anomaly_raw["p_endogenous"]),
        p_use_seasonal_injector=float(anomaly_raw["p_use_seasonal_injector"]),
    )

    return GeneratorConfig(
        raw=raw,
        num_samples=int(raw["num_samples"]),
        sequence_length=ensure_int_range(raw["sequence_length"], "sequence_length"),
        anomaly_sample_ratio=float(raw["anomaly_sample_ratio"]),
        multivariate_flag=bool(raw["multivariate_flag"]),
        num_features=ensure_int_range(raw["num_features"], "num_features"),
        seed=int(raw["seed"]) if raw.get("seed") is not None else None,
        weights=weights,
        stage1=stage1,
        causal=causal,
        anomaly=anomaly,
    )


def load_config(path: Path) -> GeneratorConfig:
    user_raw = _load_raw(path)
    return load_config_from_raw(user_raw)


def load_config_from_raw(raw: dict[str, Any]) -> GeneratorConfig:
    merged = _deep_merge(DEFAULT_CONFIG, raw)
    merged = json.loads(json.dumps(merged))
    return _build_config(merged)
