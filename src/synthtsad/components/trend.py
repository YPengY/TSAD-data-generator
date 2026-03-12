from __future__ import annotations

import numpy as np
from typing import TypeGuard

from ..config import GeneratorConfig
from ..interfaces import ArimaTrendParams, LinearTrendParams, MultipleTrendParams, TrendParams
from ..utils import weighted_choice


def _is_linear_trend_params(params: TrendParams) -> TypeGuard[LinearTrendParams]:
    return params["trend_type"] in {"increase", "decrease", "keep_steady"}


def _is_multiple_trend_params(params: TrendParams) -> TypeGuard[MultipleTrendParams]:
    return params["trend_type"] == "multiple"


def _is_arima_trend_params(params: TrendParams) -> TypeGuard[ArimaTrendParams]:
    return params["trend_type"] == "arima"


def _piecewise_linear(t: np.ndarray, k0: float, k1: float, cps: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    values = k0 + k1 * t.astype(float)
    for cp, delta in zip(cps, deltas):
        values += delta * np.maximum(t - cp, 0)
    return values


def _roots_outside_unit_circle(poly_coeffs: np.ndarray) -> bool:
    if poly_coeffs.size <= 1:
        return True
    roots = np.roots(poly_coeffs)
    if roots.size == 0:
        return True
    return bool(np.all(np.abs(roots) > 1.01))


def _sample_stable_ar_coeffs(rng: np.random.Generator, order: int, bound: float) -> list[float]:
    if order <= 0:
        return []
    for _ in range(128):
        coeffs = rng.uniform(-bound, bound, size=order).astype(float)
        # AR polynomial: phi(B) = 1 - sum(phi_i B^i)
        poly = np.concatenate(([1.0], -coeffs))
        if _roots_outside_unit_circle(poly):
            return coeffs.tolist()
    return (rng.uniform(-0.2, 0.2, size=order).astype(float)).tolist()


def _sample_invertible_ma_coeffs(rng: np.random.Generator, order: int, bound: float) -> list[float]:
    if order <= 0:
        return []
    for _ in range(128):
        coeffs = rng.uniform(-bound, bound, size=order).astype(float)
        # MA polynomial: theta(B) = 1 + sum(theta_j B^j)
        poly = np.concatenate(([1.0], coeffs))
        if _roots_outside_unit_circle(poly):
            return coeffs.tolist()
    return (rng.uniform(-0.2, 0.2, size=order).astype(float)).tolist()


def _simulate_differenced_arma(
    n: int,
    phi: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    p = int(phi.size)
    q = int(theta.size)
    eps = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)

    for t in range(n):
        ar_term = 0.0
        for i in range(1, p + 1):
            src = t - i
            if src >= 0:
                ar_term += float(phi[i - 1]) * y[src]

        ma_term = float(eps[t])
        for j in range(1, q + 1):
            src = t - j
            if src >= 0:
                ma_term += float(theta[j - 1]) * float(eps[src])

        y[t] = ar_term + ma_term

    return y


def _integrate_differences(diff_series: np.ndarray, d_order: int, base_level: float) -> np.ndarray:
    trend = diff_series.copy()
    for _ in range(max(1, d_order)):
        trend = np.cumsum(trend)
    return trend + base_level


def sample_trend_params(n: int, config: GeneratorConfig, rng: np.random.Generator) -> TrendParams:
    trend_type = weighted_choice(rng, config.weights["trend_type"])
    slope_scale = config.stage1.trend_slope_scale

    if trend_type == "increase":
        return {
            "trend_type": "increase",
            "k0": float(rng.normal(0.0, 0.2)),
            "k1": float(rng.uniform(0.25 * slope_scale, 1.2 * slope_scale)),
        }

    if trend_type == "decrease":
        return {
            "trend_type": "decrease",
            "k0": float(rng.normal(0.0, 0.2)),
            "k1": float(-rng.uniform(0.25 * slope_scale, 1.2 * slope_scale)),
        }

    if trend_type == "keep_steady":
        return {
            "trend_type": "keep_steady",
            "k0": float(rng.normal(0.0, 0.5)),
            "k1": 0.0,
        }

    if trend_type == "multiple":
        cp_count = config.stage1.trend_change_points.sample(rng)
        cp_count = min(cp_count, max(1, n // 16))
        cps = np.sort(rng.choice(np.arange(1, n - 1), size=cp_count, replace=False)).astype(int)
        deltas = rng.normal(0.0, 0.75 * slope_scale, size=cp_count).astype(float)
        return {
            "trend_type": "multiple",
            "k0": float(rng.normal(0.0, 0.2)),
            "k1": float(rng.normal(0.0, slope_scale)),
            "change_points": cps.tolist(),
            "slope_deltas": deltas.tolist(),
        }

    p_max = max(0, int(config.stage1.arima_p_max))
    q_max = max(0, int(config.stage1.arima_q_max))
    d_order = int(config.stage1.arima_d.sample(rng))
    coef_bound = float(abs(config.stage1.arima_coef_bound))
    p = int(rng.integers(0, p_max + 1))
    q = int(rng.integers(0, q_max + 1))
    phi = _sample_stable_ar_coeffs(rng=rng, order=p, bound=coef_bound)
    theta = _sample_invertible_ma_coeffs(rng=rng, order=q, bound=coef_bound)

    return {
        "trend_type": "arima",
        "p": p,
        "d": d_order,
        "q": q,
        "phi": phi,
        "theta": theta,
        "sigma": float(config.stage1.arima_noise_scale),
        "base_level": float(rng.normal(0.0, 0.2)),
        "stochastic_seed": int(rng.integers(0, 2**31 - 1)),
    }


def render_trend(t: np.ndarray, params: TrendParams) -> np.ndarray:
    if _is_linear_trend_params(params):
        k0 = float(params["k0"])
        k1 = float(params["k1"])
        return k0 + k1 * t

    if _is_multiple_trend_params(params):
        cps = np.array(params["change_points"], dtype=float)
        deltas = np.array(params["slope_deltas"], dtype=float)
        return _piecewise_linear(t, float(params["k0"]), float(params["k1"]), cps, deltas)

    if not _is_arima_trend_params(params):
        raise ValueError(f"Unsupported trend params: {params}")

    seed = int(params["stochastic_seed"])
    rng = np.random.default_rng(seed)
    phi = np.array(params["phi"], dtype=float)
    theta = np.array(params["theta"], dtype=float)
    d_order = int(params["d"])
    sigma = float(params["sigma"])
    base_level = float(params["base_level"])

    diff_series = _simulate_differenced_arma(
        n=t.size,
        phi=phi,
        theta=theta,
        sigma=sigma,
        rng=rng,
    )
    return _integrate_differences(diff_series=diff_series, d_order=d_order, base_level=base_level)


def sample_trend(
    t: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, TrendParams]:
    params = sample_trend_params(n=t.size, config=config, rng=rng)
    return render_trend(t=t, params=params), params
