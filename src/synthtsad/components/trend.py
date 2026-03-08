from __future__ import annotations

import numpy as np

from ..config import GeneratorConfig
from ..utils import weighted_choice


TrendParams = dict[str, float | int | str | list[float] | list[int]]


def _piecewise_linear(t: np.ndarray, k0: float, k1: float, cps: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    values = k0 + k1 * t.astype(float)
    for cp, delta in zip(cps, deltas):
        values += delta * np.maximum(t - cp, 0)
    return values


def _render_arima_like(n: int, phi: float, noise_scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diff = np.zeros(n, dtype=float)
    eps = rng.normal(0.0, noise_scale, size=n)
    for i in range(1, n):
        diff[i] = phi * diff[i - 1] + eps[i]
    return np.cumsum(diff)


def sample_trend_params(n: int, config: GeneratorConfig, rng: np.random.Generator) -> TrendParams:
    trend_type = weighted_choice(rng, config.weights["trend_type"])
    slope_scale = config.stage1.trend_slope_scale

    if trend_type == "increase":
        return {
            "trend_type": trend_type,
            "k0": float(rng.normal(0.0, 0.2)),
            "k1": float(rng.uniform(0.25 * slope_scale, 1.2 * slope_scale)),
        }

    if trend_type == "decrease":
        return {
            "trend_type": trend_type,
            "k0": float(rng.normal(0.0, 0.2)),
            "k1": float(-rng.uniform(0.25 * slope_scale, 1.2 * slope_scale)),
        }

    if trend_type == "keep_steady":
        return {
            "trend_type": trend_type,
            "k0": float(rng.normal(0.0, 0.5)),
            "k1": 0.0,
        }

    if trend_type == "multiple":
        cp_count = config.stage1.trend_change_points.sample(rng)
        cp_count = min(cp_count, max(1, n // 16))
        cps = np.sort(rng.choice(np.arange(1, n - 1), size=cp_count, replace=False)).astype(int)
        deltas = rng.normal(0.0, 0.75 * slope_scale, size=cp_count).astype(float)
        return {
            "trend_type": trend_type,
            "k0": float(rng.normal(0.0, 0.2)),
            "k1": float(rng.normal(0.0, slope_scale)),
            "change_points": cps.tolist(),
            "slope_deltas": deltas.tolist(),
        }

    return {
        "trend_type": "arima",
        "phi": float(rng.uniform(-0.6, 0.6)),
        "noise_scale": float(config.stage1.arima_noise_scale),
        "stochastic_seed": int(rng.integers(0, 2**31 - 1)),
    }


def render_trend(t: np.ndarray, params: TrendParams) -> np.ndarray:
    trend_type = str(params["trend_type"])

    if trend_type in {"increase", "decrease", "keep_steady"}:
        k0 = float(params["k0"])
        k1 = float(params["k1"])
        return k0 + k1 * t

    if trend_type == "multiple":
        cps = np.array(params["change_points"], dtype=float)
        deltas = np.array(params["slope_deltas"], dtype=float)
        return _piecewise_linear(t, float(params["k0"]), float(params["k1"]), cps, deltas)

    return _render_arima_like(
        n=t.size,
        phi=float(params["phi"]),
        noise_scale=float(params["noise_scale"]),
        seed=int(params["stochastic_seed"]),
    )


def sample_trend(
    t: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, TrendParams]:
    params = sample_trend_params(n=t.size, config=config, rng=rng)
    return render_trend(t=t, params=params), params
