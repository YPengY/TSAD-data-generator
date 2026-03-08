from __future__ import annotations

import numpy as np

from ..config import GeneratorConfig
from ..utils import weighted_choice


WAVELET_FAMILIES = ["morlet", "ricker", "haar"]
SeasonalParams = dict[str, str | int | float | list[dict[str, float | str]]]


def _triangle_wave(phase: np.ndarray) -> np.ndarray:
    return (2.0 / np.pi) * np.arcsin(np.sin(phase))


def _wavelet_atom(t: np.ndarray, period: float, phase: float, family: str) -> np.ndarray:
    u = ((t / period) + phase / (2 * np.pi)) % 1.0
    centered = u - 0.5
    if family == "ricker":
        sigma = 0.18
        z = centered / sigma
        return (1 - z**2) * np.exp(-0.5 * z**2)
    if family == "haar":
        return np.where(centered < 0, -1.0, 1.0)
    sigma = 0.14
    return np.cos(10.0 * np.pi * centered) * np.exp(-(centered**2) / (2.0 * sigma**2))


def _sample_period(config: GeneratorConfig, rng: np.random.Generator) -> int:
    regime = weighted_choice(rng, config.weights["frequency_regime"])
    period_range = config.stage1.period_high if regime == "high" else config.stage1.period_low
    return period_range.sample(rng)


def sample_seasonality_params(n: int, config: GeneratorConfig, rng: np.random.Generator) -> SeasonalParams:
    _ = n
    season_type = weighted_choice(rng, config.weights["seasonality_type"])
    if season_type == "none":
        return {"seasonality_type": "none", "atoms": []}

    k_atoms = config.stage1.seasonal_atoms.sample(rng)
    amp_min, amp_max = config.stage1.seasonal_amplitude
    atoms: list[dict[str, float | str]] = []

    for _ in range(k_atoms):
        period = float(_sample_period(config, rng))
        freq = 1.0 / period
        amplitude = float(rng.uniform(amp_min, amp_max))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        atom: dict[str, float | str] = {
            "type": season_type,
            "period": period,
            "frequency": freq,
            "amplitude": amplitude,
            "phase": phase,
        }
        if season_type == "wavelet":
            atom["family"] = str(rng.choice(WAVELET_FAMILIES))
        atoms.append(atom)

    return {"seasonality_type": season_type, "atoms": atoms}


def render_seasonality(t: np.ndarray, params: SeasonalParams) -> np.ndarray:
    season_type = str(params["seasonality_type"])
    n = t.size
    if season_type == "none":
        return np.zeros(n, dtype=float)

    signal = np.zeros(n, dtype=float)
    atoms = list(params["atoms"])  # type: ignore[arg-type]

    for atom in atoms:
        freq = float(atom["frequency"])
        amplitude = float(atom["amplitude"])
        phase = float(atom["phase"])

        if season_type == "sine":
            base = np.sin(2 * np.pi * freq * t + phase)
        elif season_type == "square":
            base = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif season_type == "triangle":
            base = _triangle_wave(2 * np.pi * freq * t + phase)
        else:
            period = float(atom["period"])
            family = str(atom.get("family", "morlet"))
            base = _wavelet_atom(t=t, period=period, phase=phase, family=family)

        signal += amplitude * base

    return signal


def sample_seasonality(
    t: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, SeasonalParams]:
    params = sample_seasonality_params(n=t.size, config=config, rng=rng)
    return render_seasonality(t=t, params=params), params
