from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from synthtsad.anomaly.local import LocalAnomalyInjector
from synthtsad.anomaly.seasonal import SeasonalAnomalyInjector
from synthtsad.causal.arx import ARXSystem
from synthtsad.causal.dag import CausalGraphSampler
from synthtsad.components.noise import render_noise
from synthtsad.components.seasonality import render_seasonality
from synthtsad.components.trend import render_trend
from synthtsad.config import DEFAULT_CONFIG, load_config_from_raw
from synthtsad.labeling.labeler import LabelBuilder
from synthtsad.pipeline import SyntheticGeneratorPipeline


CONFIG_PATH = REPO_ROOT / "configs" / "default.json"

SECTION_LABELS = {
    "root": "Project Config",
    "num_samples": "Number of Samples",
    "sequence_length": "Sequence Length",
    "anomaly_sample_ratio": "Anomalous Sample Ratio",
    "num_series": "Number of Series",
    "seed": "Global Seed",
    "weights": "Sampling Weights",
    "stage1": "Stage 1 Baseline",
    "stage1.trend": "Trend",
    "stage1.seasonality": "Seasonality",
    "stage1.noise": "Noise",
    "causal": "Stage 2 Causal",
    "anomaly": "Stage 3 Anomalies",
    "debug": "Debug Toggles",
}

FIELD_LABELS = {
    "min": "Min",
    "max": "Max",
    "num_samples": "Sample Count",
    "sequence_length": "Sequence Length",
    "anomaly_sample_ratio": "Anomalous Sample Ratio",
    "num_series": "Series Count",
    "seed": "Seed",
    "change_points": "Change Points",
    "slope_scale": "Slope Scale",
    "arima_noise_scale": "ARIMA Noise Scale",
    "arima": "ARIMA Settings",
    "p_max": "AR Order Max",
    "q_max": "MA Order Max",
    "d": "Difference Order",
    "coef_bound": "Coefficient Bound",
    "atoms": "Seasonal Atoms",
    "amplitude": "Amplitude Range",
    "base_period": "Base Period",
    "low": "Low Frequency Period",
    "high": "High Frequency Period",
    "wavelet": "Wavelet Settings",
    "families": "Wavelet Family Weights",
    "scale": "Scale Range",
    "shift": "Shift Range",
    "contrastive": "Contrastive Pairing",
    "ratio": "Contrastive Ratio",
    "params": "Contrastive Params",
    "sigma": "Noise Sigma",
    "volatility_windows": "Volatility Windows",
    "volatility_multiplier": "Volatility Multiplier",
    "num_nodes": "Causal Node Range",
    "edge_density": "Edge Density",
    "max_lag": "Max Lag",
    "a_i_bound": "Self Dynamic Bound",
    "bias_std": "Bias Std",
    "b_ij_std": "Edge Gain Std",
    "alpha_i_min": "Alpha Min",
    "alpha_i_max": "Alpha Max",
    "events_per_sample": "Events Per Sample",
    "window_length": "Event Window Length",
    "local_types": "Local Anomaly Types",
    "seasonal_types": "Seasonal Anomaly Types",
    "p_endogenous": "Endogenous Local Probability",
    "p_use_seasonal_injector": "Seasonal Injector Probability",
    "enable_trend": "Enable Trend",
    "enable_seasonality": "Enable Seasonality",
    "enable_noise": "Enable Noise",
    "enable_causal": "Enable Causal",
    "enable_local_anomaly": "Enable Local Anomaly",
    "enable_seasonal_anomaly": "Enable Seasonal Anomaly",
}

PATH_LABEL_OVERRIDES = {
    "weights.seasonality_type": "Seasonality Type Weights",
    "weights.seasonality_type.none": "None Weight",
    "weights.seasonality_type.sine": "Sine Weight",
    "weights.seasonality_type.square": "Square Weight",
    "weights.seasonality_type.triangle": "Triangle Weight",
    "weights.seasonality_type.wavelet": "Wavelet Weight",
    "weights.trend_type": "Trend Type Weights",
    "weights.trend_type.decrease": "Decrease Weight",
    "weights.trend_type.increase": "Increase Weight",
    "weights.trend_type.keep_steady": "Keep Steady Weight",
    "weights.trend_type.multiple": "Multiple Weight",
    "weights.trend_type.arima": "ARIMA Weight",
    "weights.frequency_regime": "Frequency Regime Weights",
    "weights.frequency_regime.low": "Low Regime Weight",
    "weights.frequency_regime.high": "High Regime Weight",
    "weights.noise_level": "Noise Level Weights",
    "weights.noise_level.almost_none": "Almost None Weight",
    "weights.noise_level.low": "Low Noise Weight",
    "weights.noise_level.moderate": "Moderate Noise Weight",
    "weights.noise_level.high": "High Noise Weight",
    "stage1.noise.sigma.almost_none": "Almost None Sigma",
    "stage1.noise.sigma.low": "Low Noise Sigma",
    "stage1.noise.sigma.moderate": "Moderate Noise Sigma",
    "stage1.noise.sigma.high": "High Noise Sigma",
}

MULTI_SELECT_OPTIONS = {
    "anomaly.local_types": list(DEFAULT_CONFIG["anomaly"]["local_types"]),
    "anomaly.seasonal_types": list(DEFAULT_CONFIG["anomaly"]["seasonal_types"]),
    "stage1.seasonality.wavelet.contrastive.params": list(
        DEFAULT_CONFIG["stage1"]["seasonality"]["wavelet"]["contrastive"]["params"]
    ),
}

RANGE_BOUNDS: dict[str, tuple[float, float, str]] = {
    "sequence_length": (32, 2048, "int"),
    "num_series": (1, 12, "int"),
    "stage1.trend.change_points": (1, 8, "int"),
    "stage1.trend.arima.d": (1, 3, "int"),
    "stage1.seasonality.atoms": (1, 5, "int"),
    "stage1.seasonality.amplitude": (0.1, 3.0, "float"),
    "stage1.seasonality.base_period.low": (24, 256, "int"),
    "stage1.seasonality.base_period.high": (4, 64, "int"),
    "stage1.seasonality.wavelet.scale": (0.04, 0.5, "float"),
    "stage1.seasonality.wavelet.shift": (0.0, 1.0, "float"),
    "stage1.noise.volatility_windows": (0, 6, "int"),
    "stage1.noise.volatility_multiplier": (0.05, 1.5, "float"),
    "causal.num_nodes": (1, 20, "int"),
    "anomaly.events_per_sample": (1, 5, "int"),
    "anomaly.window_length": (4, 160, "int"),
}

SCALAR_BOUNDS: dict[str, tuple[float, float, str]] = {
    "num_samples": (1, 256, "int"),
    "seed": (1, 999_999, "int"),
    "anomaly_sample_ratio": (0.0, 1.0, "float"),
    "stage1.trend.slope_scale": (0.001, 0.08, "float"),
    "stage1.trend.arima_noise_scale": (0.005, 0.2, "float"),
    "stage1.trend.arima.p_max": (0, 4, "int"),
    "stage1.trend.arima.q_max": (0, 4, "int"),
    "stage1.trend.arima.coef_bound": (0.1, 1.2, "float"),
    "stage1.seasonality.wavelet.contrastive.ratio": (0.0, 1.0, "float"),
    "causal.edge_density": (0.0, 0.8, "float"),
    "causal.max_lag": (0, 24, "int"),
    "causal.a_i_bound": (0.1, 1.2, "float"),
    "causal.bias_std": (0.0, 0.5, "float"),
    "causal.b_ij_std": (0.05, 0.8, "float"),
    "causal.alpha_i_min": (0.05, 0.8, "float"),
    "causal.alpha_i_max": (0.2, 0.98, "float"),
    "anomaly.p_endogenous": (0.0, 1.0, "float"),
    "anomaly.p_use_seasonal_injector": (0.0, 1.0, "float"),
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _deep_copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _pretty_label(path: str) -> str:
    if path in PATH_LABEL_OVERRIDES:
        return PATH_LABEL_OVERRIDES[path]
    if path in SECTION_LABELS:
        return SECTION_LABELS[path]
    leaf = path.split(".")[-1]
    if leaf in FIELD_LABELS:
        return FIELD_LABELS[leaf]
    return leaf.replace("_", " ").title()


def _build_numeric_bounds() -> dict[str, dict[str, int | float | str]]:
    bounds: dict[str, dict[str, int | float | str]] = {}
    for path, (low, high, kind) in SCALAR_BOUNDS.items():
        bounds[path] = {"min": low, "max": high, "kind": kind}
    for path, (low, high, kind) in RANGE_BOUNDS.items():
        bounds[f"{path}.min"] = {"min": low, "max": high, "kind": kind}
        bounds[f"{path}.max"] = {"min": low, "max": high, "kind": kind}
    return bounds


def _collect_paths(node: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if prefix:
        paths.append(prefix)
    if isinstance(node, dict):
        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_collect_paths(value, child_prefix))
    return paths


def load_default_config_raw() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    return _deep_copy_jsonable(DEFAULT_CONFIG)


def get_bootstrap_payload() -> dict[str, Any]:
    defaults = load_default_config_raw()
    path_labels = {path: _pretty_label(path) for path in _collect_paths(defaults)}
    path_labels["root"] = SECTION_LABELS["root"]
    return {
        "defaults": defaults,
        "ui": {
            "sectionLabels": SECTION_LABELS,
            "fieldLabels": FIELD_LABELS,
            "multiSelectOptions": MULTI_SELECT_OPTIONS,
            "numericBounds": _build_numeric_bounds(),
            "pathLabels": path_labels,
        },
    }


def randomize_config(seed: int | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    for _ in range(128):
        candidate = _randomize_from_defaults(rng)
        try:
            load_config_from_raw(candidate)
            return candidate
        except ValueError:
            continue
    raise ValueError("Unable to create a valid randomized config after 128 attempts.")


def _randomize_from_defaults(rng: np.random.Generator) -> dict[str, Any]:
    raw = _deep_copy_jsonable(load_default_config_raw())
    raw["num_samples"] = _sample_scalar("num_samples", rng)
    raw["seed"] = _sample_scalar("seed", rng)
    raw["anomaly_sample_ratio"] = _sample_scalar("anomaly_sample_ratio", rng)
    raw["sequence_length"] = _sample_range("sequence_length", rng)
    raw["num_series"] = _sample_range("num_series", rng)

    raw["weights"] = {key: _randomize_weight_dict(weights, rng) for key, weights in raw["weights"].items()}

    trend = raw["stage1"]["trend"]
    trend["change_points"] = _sample_range("stage1.trend.change_points", rng)
    trend["slope_scale"] = _sample_scalar("stage1.trend.slope_scale", rng)
    trend["arima_noise_scale"] = _sample_scalar("stage1.trend.arima_noise_scale", rng)
    trend["arima"]["p_max"] = _sample_scalar("stage1.trend.arima.p_max", rng)
    trend["arima"]["q_max"] = _sample_scalar("stage1.trend.arima.q_max", rng)
    trend["arima"]["d"] = _sample_range("stage1.trend.arima.d", rng)
    trend["arima"]["coef_bound"] = _sample_scalar("stage1.trend.arima.coef_bound", rng)

    season = raw["stage1"]["seasonality"]
    season["atoms"] = _sample_range("stage1.seasonality.atoms", rng)
    season["amplitude"] = _sample_range("stage1.seasonality.amplitude", rng)
    season["base_period"]["high"] = _sample_range("stage1.seasonality.base_period.high", rng)
    low_period = _sample_range("stage1.seasonality.base_period.low", rng)
    high_max = int(season["base_period"]["high"]["max"])
    low_period["min"] = max(int(low_period["min"]), high_max + 1)
    low_period["max"] = max(int(low_period["max"]), int(low_period["min"]))
    season["base_period"]["low"] = low_period
    season["wavelet"]["families"] = _randomize_weight_dict(season["wavelet"]["families"], rng)
    season["wavelet"]["scale"] = _sample_range("stage1.seasonality.wavelet.scale", rng)
    season["wavelet"]["shift"] = _sample_range("stage1.seasonality.wavelet.shift", rng)
    season["wavelet"]["contrastive"]["ratio"] = _sample_scalar(
        "stage1.seasonality.wavelet.contrastive.ratio",
        rng,
    )
    season["wavelet"]["contrastive"]["params"] = _random_subset(
        MULTI_SELECT_OPTIONS["stage1.seasonality.wavelet.contrastive.params"],
        rng,
    )

    noise = raw["stage1"]["noise"]
    noise["sigma"] = _randomize_scalar_dict(noise["sigma"], 0.001, 0.4, rng)
    noise["volatility_windows"] = _sample_range("stage1.noise.volatility_windows", rng)
    noise["volatility_multiplier"] = _sample_range("stage1.noise.volatility_multiplier", rng)

    causal = raw["causal"]
    causal["num_nodes"] = _sample_range("causal.num_nodes", rng)
    num_series = raw["num_series"]
    causal["num_nodes"]["min"] = min(int(causal["num_nodes"]["min"]), int(num_series["min"]))
    causal["num_nodes"]["max"] = max(int(causal["num_nodes"]["max"]), int(num_series["max"]))
    causal["edge_density"] = _sample_scalar("causal.edge_density", rng)
    causal["max_lag"] = _sample_scalar("causal.max_lag", rng)
    causal["a_i_bound"] = _sample_scalar("causal.a_i_bound", rng)
    causal["bias_std"] = _sample_scalar("causal.bias_std", rng)
    causal["b_ij_std"] = _sample_scalar("causal.b_ij_std", rng)
    alpha_min = _sample_scalar("causal.alpha_i_min", rng)
    alpha_max = _sample_scalar("causal.alpha_i_max", rng)
    causal["alpha_i_min"] = min(alpha_min, alpha_max)
    causal["alpha_i_max"] = max(alpha_min, alpha_max)

    anomaly = raw["anomaly"]
    anomaly["events_per_sample"] = _sample_range("anomaly.events_per_sample", rng)
    anomaly["window_length"] = _sample_range("anomaly.window_length", rng)
    anomaly["local_types"] = _random_subset(MULTI_SELECT_OPTIONS["anomaly.local_types"], rng)
    anomaly["seasonal_types"] = _random_subset(MULTI_SELECT_OPTIONS["anomaly.seasonal_types"], rng)
    anomaly["p_endogenous"] = _sample_scalar("anomaly.p_endogenous", rng)
    anomaly["p_use_seasonal_injector"] = _sample_scalar("anomaly.p_use_seasonal_injector", rng)

    debug = raw["debug"]
    for key in list(debug.keys()):
        debug[key] = bool(rng.random() < 0.82)
    if not any([debug["enable_trend"], debug["enable_seasonality"], debug["enable_noise"]]):
        debug["enable_trend"] = True
    if not debug["enable_local_anomaly"] and not debug["enable_seasonal_anomaly"]:
        debug["enable_local_anomaly"] = True

    return raw


def _sample_scalar(path: str, rng: np.random.Generator) -> int | float:
    low, high, kind = SCALAR_BOUNDS[path]
    if kind == "int":
        return int(rng.integers(int(low), int(high) + 1))
    return round(float(rng.uniform(low, high)), 4)


def _sample_range(path: str, rng: np.random.Generator) -> dict[str, int | float]:
    low, high, kind = RANGE_BOUNDS[path]
    if kind == "int":
        start = int(rng.integers(int(low), int(high) + 1))
        end = int(rng.integers(start, int(high) + 1))
        return {"min": start, "max": end}
    start = round(float(rng.uniform(low, high)), 4)
    end = round(float(rng.uniform(start, high)), 4)
    return {"min": start, "max": end}


def _randomize_weight_dict(weights: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    sampled = {key: float(rng.uniform(0.05, 1.0)) for key in weights}
    total = sum(sampled.values())
    return {key: round(value / total, 4) for key, value in sampled.items()}


def _randomize_scalar_dict(values: dict[str, float], low: float, high: float, rng: np.random.Generator) -> dict[str, float]:
    return {key: round(float(rng.uniform(low, high)), 4) for key in values}


def _random_subset(options: list[str], rng: np.random.Generator) -> list[str]:
    count = int(rng.integers(1, len(options) + 1))
    chosen = list(rng.choice(options, size=count, replace=False))
    return sorted(str(value) for value in chosen)


def preview_sample(raw_config: dict[str, Any]) -> dict[str, Any]:
    preview_raw = _deep_copy_jsonable(raw_config)
    preview_raw["num_samples"] = 1
    cfg = load_config_from_raw(preview_raw)

    pipeline = SyntheticGeneratorPipeline(cfg)
    rng = np.random.default_rng(cfg.seed)
    n, d = pipeline._sample_dimensions(rng)
    t = np.arange(n, dtype=float)

    stage1_params = pipeline._sample_stage1_params(n=n, d=d, rng=rng)
    x_stage1 = _realize_stage1_preview(cfg, t=t, stage1_params=stage1_params)

    if cfg.debug.enable_causal:
        graph = CausalGraphSampler(cfg).sample_graph(num_nodes=d, rng=rng)
        arx = ARXSystem(cfg, graph)
        arx_params = arx.sample_params(rng)
    else:
        graph = pipeline._empty_graph(d)
        arx = ARXSystem(cfg, graph)
        arx_params = {"disabled": True}

    local_injector = LocalAnomalyInjector(cfg)
    seasonal_injector = SeasonalAnomalyInjector(cfg)
    sampled_local_events = []
    sampled_seasonal_events = []

    if rng.random() < cfg.anomaly_sample_ratio:
        if cfg.debug.enable_local_anomaly:
            sampled_local_events = local_injector.sample_events(n=n, d=d, rng=rng, graph=graph)
        if cfg.debug.enable_seasonal_anomaly:
            sampled_seasonal_events = seasonal_injector.sample_events(n=n, d=d, rng=rng)

    if not cfg.debug.enable_causal:
        for event in sampled_local_events:
            event.is_endogenous = False
            event.root_cause_node = None

    pre_causal_local_events = [event for event in sampled_local_events if bool(event.is_endogenous)]
    post_causal_local_events = [event for event in sampled_local_events if not bool(event.is_endogenous)]

    x_stage1_anom = x_stage1.copy()
    realized_events = []
    if pre_causal_local_events:
        x_stage1_anom, local_events = local_injector.apply_events(x_normal=x_stage1_anom, events=pre_causal_local_events)
        realized_events.extend(local_events)

    if cfg.debug.enable_causal:
        x_stage2_normal, causal_state = arx.simulate_with_params(x_base=x_stage1, n_steps=n, params=arx_params)
        x_observed, _ = arx.simulate_with_params(x_base=x_stage1_anom, n_steps=n, params=arx_params)
    else:
        x_stage2_normal = x_stage1.copy()
        x_observed = x_stage1_anom.copy()
        causal_state = pipeline._disabled_causal_state(n=n, d=d)

    if post_causal_local_events:
        x_observed, local_events = local_injector.apply_events(x_normal=x_observed, events=post_causal_local_events)
        realized_events.extend(local_events)

    if sampled_seasonal_events:
        x_observed, seasonal_events = seasonal_injector.apply_events(x_input=x_observed, events=sampled_seasonal_events, rng=rng)
        realized_events.extend(seasonal_events)

    labels = LabelBuilder(cfg).build(
        x_normal=x_stage2_normal,
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
            "local": [event.to_dict() for event in sampled_local_events],
            "seasonal": [event.to_dict() for event in sampled_seasonal_events],
        },
    }
    payload = {
        "summary": {
            "length": int(n),
            "num_features": int(d),
            "is_anomalous_sample": int(labels["is_anomalous_sample"]),
            "num_events": int(len(realized_events)),
        },
        "series": {
            "stage1_baseline": x_stage1,
            "stage2_normal": x_stage2_normal,
            "observed": x_observed,
        },
        "labels": {
            "point_mask": labels["point_mask"],
            "point_mask_any": labels["point_mask_any"],
            "root_cause": labels["root_cause"],
            "affected_nodes": labels["affected_nodes"],
            "events": labels["events"],
        },
        "graph": {
            "num_nodes": int(graph.num_nodes),
            "adjacency": graph.adjacency,
            "topo_order": graph.topo_order,
            "parents": graph.parents,
        },
        "metadata": metadata,
    }
    return _to_jsonable(payload)


def _realize_stage1_preview(cfg, t: np.ndarray, stage1_params: list[dict[str, Any]]) -> np.ndarray:
    n = t.size
    d = len(stage1_params)
    x_base = np.zeros((n, d), dtype=float)
    for spec in stage1_params:
        node = int(spec["node"])
        trend = render_trend(t=t, params=spec["trend"]) if cfg.debug.enable_trend else np.zeros(n, dtype=float)
        season = (
            render_seasonality(t=t, params=spec["seasonality"])
            if cfg.debug.enable_seasonality
            else np.zeros(n, dtype=float)
        )
        noise = render_noise(n=n, params=spec["noise"]) if cfg.debug.enable_noise else np.zeros(n, dtype=float)
        x_base[:, node] = trend + season + noise
    return x_base
