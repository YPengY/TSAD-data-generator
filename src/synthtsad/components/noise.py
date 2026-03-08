from __future__ import annotations

import numpy as np

from ..config import GeneratorConfig
from ..utils import weighted_choice


NoiseParams = dict[str, str | float | int | list[dict[str, int | float]]]


def sample_noise_params(n: int, config: GeneratorConfig, rng: np.random.Generator) -> NoiseParams:
    noise_level = weighted_choice(rng, config.weights["noise_level"])
    sigma0 = float(config.stage1.noise_sigma[noise_level])

    windows: list[dict[str, int | float]] = []
    burst_count = config.stage1.volatility_windows.sample(rng)
    vmin, vmax = config.stage1.volatility_multiplier

    for _ in range(burst_count):
        if n < 4:
            break
        start = int(rng.integers(0, max(1, n - 2)))
        end = int(rng.integers(start + 1, n + 1))
        v = float(rng.uniform(vmin, vmax))
        windows.append({"start": start, "end": end, "v": v})

    return {
        "noise_level": noise_level,
        "sigma0": sigma0,
        "volatility_windows": windows,
        "stochastic_seed": int(rng.integers(0, 2**31 - 1)),
    }


def render_noise(n: int, params: NoiseParams) -> np.ndarray:
    sigma_t = np.full(n, float(params["sigma0"]), dtype=float)
    for win in params["volatility_windows"]:  # type: ignore[index]
        start = int(win["start"])
        end = int(win["end"])
        v = float(win["v"])
        sigma_t[start:end] *= 1.0 + v

    rng = np.random.default_rng(int(params["stochastic_seed"]))
    return rng.normal(0.0, sigma_t)


def sample_noise(
    n: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, NoiseParams]:
    params = sample_noise_params(n=n, config=config, rng=rng)
    return render_noise(n=n, params=params), params
