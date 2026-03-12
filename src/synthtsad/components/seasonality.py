from __future__ import annotations

import numpy as np

from ..config import GeneratorConfig
from ..interfaces import (
    PeriodicAtom,
    SeasonalityAtom,
    SeasonalityType,
    SeasonalParams,
    WaveletAtom,
)
from ..utils import weighted_choice


def _triangle_wave(phase: np.ndarray) -> np.ndarray:
    return (2.0 / np.pi) * np.arcsin(np.sin(phase))


def _square_wave(cycle: np.ndarray, duty_cycle: float) -> np.ndarray:
    return np.where(cycle < duty_cycle, 1.0, -1.0)


def _asymmetric_triangle_wave(cycle: np.ndarray, duty_cycle: float) -> np.ndarray:
    duty = float(np.clip(duty_cycle, 1e-3, 1.0 - 1e-3))
    rising = cycle < duty
    values = np.empty_like(cycle, dtype=float)
    values[rising] = -1.0 + 2.0 * cycle[rising] / duty
    values[~rising] = 1.0 - 2.0 * (cycle[~rising] - duty) / max(1.0 - duty, 1e-3)
    return values


def _normalize_waveform(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-8:
        return x
    return x / peak


def _wavelet_morlet(z: np.ndarray, theta: dict[str, float]) -> np.ndarray:
    omega = float(theta.get("omega", 7.5))
    return np.cos(omega * z) * np.exp(-0.5 * z**2)


def _wavelet_ricker(z: np.ndarray, theta: dict[str, float]) -> np.ndarray:
    _ = theta
    return (1.0 - z**2) * np.exp(-0.5 * z**2)


def _wavelet_mexh(z: np.ndarray, theta: dict[str, float]) -> np.ndarray:
    _ = theta
    return (2.0 - z**2) * np.exp(-0.5 * z**2)


def _wavelet_haar(z: np.ndarray, theta: dict[str, float]) -> np.ndarray:
    _ = theta
    support = np.abs(z) <= 1.0
    return np.where(support, np.where(z < 0.0, -1.0, 1.0), 0.0)


def _wavelet_gaus(z: np.ndarray, theta: dict[str, float]) -> np.ndarray:
    order = int(theta.get("order", 1))
    if order <= 1:
        return -z * np.exp(-0.5 * z**2)
    return (z**2 - 1.0) * np.exp(-0.5 * z**2)


def _wavelet_shan(z: np.ndarray, theta: dict[str, float]) -> np.ndarray:
    bandwidth = float(theta.get("bandwidth", 5.0))
    center = float(theta.get("center", 1.5))
    return np.sinc(bandwidth * z) * np.cos(2.0 * np.pi * center * z)


WAVELET_REGISTRY = {
    "morlet": _wavelet_morlet,
    "ricker": _wavelet_ricker,
    "mexh": _wavelet_mexh,
    "haar": _wavelet_haar,
    "gaus": _wavelet_gaus,
    "shan": _wavelet_shan,
}


def _sample_wavelet_theta(family: str, rng: np.random.Generator) -> dict[str, float]:
    if family == "morlet":
        return {"omega": float(rng.uniform(6.0, 12.0))}
    if family == "gaus":
        return {"order": float(rng.integers(1, 3))}
    if family == "shan":
        return {
            "bandwidth": float(rng.uniform(3.0, 10.0)),
            "center": float(rng.uniform(0.6, 2.4)),
        }
    return {}


def _wavelet_atom(
    t: np.ndarray,
    period: float,
    phase: float,
    family: str,
    scale: float,
    shift: float,
    theta: dict[str, float],
) -> np.ndarray:
    u = ((t / period) + phase / (2 * np.pi) - shift) % 1.0
    centered = u - 0.5
    z = centered / max(scale, 1e-4)
    kernel = WAVELET_REGISTRY.get(family, _wavelet_morlet)
    return _normalize_waveform(kernel(z, theta))


def _sample_period(config: GeneratorConfig, rng: np.random.Generator) -> int:
    regime = weighted_choice(rng, config.weights["frequency_regime"])
    period_range = config.stage1.period_high if regime == "high" else config.stage1.period_low
    return period_range.sample(rng)


def _base_atom(
    atom_type: str,
    amplitude: float,
    period: float,
    phase: float,
) -> PeriodicAtom:
    atom: PeriodicAtom = {
        "type": atom_type,
        "period": period,
        "frequency": 1.0 / period,
        "amplitude": amplitude,
        "phase": phase,
        "modulation_depth": 0.0,
        "modulation_frequency": 0.0,
        "modulation_phase": 0.0,
    }
    if atom_type in {"square", "triangle"}:
        atom["duty_cycle"] = 0.5
        atom["cycle_shift"] = 0.0
    return atom


def _make_wavelet_atom(
    amplitude: float,
    period: float,
    phase: float,
    family: str,
    scale: float,
    shift: float,
    theta: dict[str, float],
) -> WaveletAtom:
    return {
        "type": "wavelet",
        "period": period,
        "frequency": 1.0 / period,
        "amplitude": amplitude,
        "phase": phase,
        "modulation_depth": 0.0,
        "modulation_frequency": 0.0,
        "modulation_phase": 0.0,
        "family": family,
        "scale": scale,
        "shift": shift,
        "theta": theta,
    }


def _sample_wavelet_atom_base(config: GeneratorConfig, rng: np.random.Generator) -> WaveletAtom:
    amp_min, amp_max = config.stage1.seasonal_amplitude
    period = float(_sample_period(config, rng))
    family = weighted_choice(rng, config.stage1.wavelet_family_weights)
    return _make_wavelet_atom(
        amplitude=float(rng.uniform(amp_min, amp_max)),
        period=period,
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        family=family,
        scale=float(rng.uniform(*config.stage1.wavelet_scale)),
        shift=float(rng.uniform(*config.stage1.wavelet_shift)),
        theta=_sample_wavelet_theta(family, rng),
    )


def _copy_wavelet_atom(atom: WaveletAtom) -> WaveletAtom:
    copied: WaveletAtom = {
        "type": atom["type"],
        "period": float(atom["period"]),
        "frequency": float(atom["frequency"]),
        "amplitude": float(atom["amplitude"]),
        "phase": float(atom["phase"]),
        "modulation_depth": float(atom["modulation_depth"]),
        "modulation_frequency": float(atom["modulation_frequency"]),
        "modulation_phase": float(atom["modulation_phase"]),
        "family": str(atom["family"]),
        "scale": float(atom["scale"]),
        "shift": float(atom["shift"]),
        "theta": dict(atom["theta"]),
    }
    if "contrastive_group" in atom:
        copied["contrastive_group"] = int(atom["contrastive_group"])
    if "contrastive_role" in atom:
        copied["contrastive_role"] = atom["contrastive_role"]
    if "contrastive_changed" in atom:
        copied["contrastive_changed"] = str(atom["contrastive_changed"])
    return copied


def _sample_seasonality_type(config: GeneratorConfig, rng: np.random.Generator) -> SeasonalityType:
    season_type = weighted_choice(rng, config.weights["seasonality_type"])
    if season_type == "none":
        return "none"
    if season_type == "sine":
        return "sine"
    if season_type == "square":
        return "square"
    if season_type == "triangle":
        return "triangle"
    if season_type == "wavelet":
        return "wavelet"
    raise ValueError(f"Unsupported seasonality type: {season_type}")


def _sample_contrastive_variant(
    anchor: WaveletAtom,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[WaveletAtom, str]:
    variant = _copy_wavelet_atom(anchor)
    changed = str(rng.choice(config.stage1.wavelet_contrastive_params))

    if changed == "family":
        source_family = anchor["family"]
        candidates = [f for f in config.stage1.wavelet_family_weights if f != source_family]
        if not candidates:
            changed = "scale"
        else:
            target_family = str(rng.choice(candidates))
            variant["family"] = target_family
            variant["theta"] = _sample_wavelet_theta(target_family, rng)

    if changed == "scale":
        low, high = config.stage1.wavelet_scale
        span = max(high - low, 1e-6)
        current = float(anchor["scale"])
        delta = float(rng.uniform(0.15 * span, 0.50 * span))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        value = float(np.clip(current + sign * delta, low, high))
        if abs(value - current) < 1e-6:
            value = high if current <= (low + high) * 0.5 else low
        variant["scale"] = value

    if changed == "shift":
        low, high = config.stage1.wavelet_shift
        span = max(high - low, 1e-6)
        current = float(anchor["shift"])
        delta = float(rng.uniform(0.15 * span, 0.45 * span))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        value = current + sign * delta
        while value < low:
            value += span
        while value > high:
            value -= span
        variant["shift"] = float(value)

    return variant, changed


def sample_seasonality_params(
    n: int, config: GeneratorConfig, rng: np.random.Generator
) -> SeasonalParams:
    _ = n
    season_type = _sample_seasonality_type(config=config, rng=rng)
    if season_type == "none":
        return {"seasonality_type": "none", "atoms": []}

    k_atoms = config.stage1.seasonal_atoms.sample(rng)
    amp_min, amp_max = config.stage1.seasonal_amplitude
    atoms: list[SeasonalityAtom] = []

    if season_type != "wavelet":
        for _ in range(k_atoms):
            period = float(_sample_period(config, rng))
            atoms.append(
                _base_atom(
                    atom_type=season_type,
                    amplitude=float(rng.uniform(amp_min, amp_max)),
                    period=period,
                    phase=float(rng.uniform(0.0, 2.0 * np.pi)),
                )
            )
        return {"seasonality_type": season_type, "atoms": atoms}

    remaining = k_atoms
    group_id = 0
    while remaining > 0:
        anchor = _sample_wavelet_atom_base(config=config, rng=rng)
        should_pair = remaining >= 2 and rng.random() < config.stage1.wavelet_contrastive_ratio
        if should_pair:
            variant, changed = _sample_contrastive_variant(anchor=anchor, config=config, rng=rng)
            anchor["contrastive_group"] = group_id
            anchor["contrastive_role"] = "anchor"
            anchor["contrastive_changed"] = "none"
            variant["contrastive_group"] = group_id
            variant["contrastive_role"] = "paired"
            variant["contrastive_changed"] = changed
            atoms.extend([anchor, variant])
            remaining -= 2
            group_id += 1
        else:
            atoms.append(anchor)
            remaining -= 1

    return {"seasonality_type": season_type, "atoms": atoms}


def render_seasonality(t: np.ndarray, params: SeasonalParams) -> np.ndarray:
    season_type = str(params["seasonality_type"])
    n = t.size
    if season_type == "none":
        return np.zeros(n, dtype=float)

    signal = np.zeros(n, dtype=float)
    atoms = list(params["atoms"])

    for atom in atoms:
        atom_type = str(atom.get("type", season_type))
        freq = float(atom["frequency"])
        amplitude = float(atom["amplitude"])
        phase = float(atom["phase"])
        mod_depth = float(atom.get("modulation_depth", 0.0))
        mod_freq = float(atom.get("modulation_frequency", 0.0))
        mod_phase = float(atom.get("modulation_phase", 0.0))
        modulation = 1.0 + mod_depth * np.sin(2.0 * np.pi * mod_freq * t + mod_phase)
        effective_amplitude = amplitude * modulation

        if atom_type == "sine":
            base = np.sin(2.0 * np.pi * freq * t + phase)
        elif atom_type == "square":
            cycle_shift = float(atom.get("cycle_shift", 0.0))
            duty_cycle = float(atom.get("duty_cycle", 0.5))
            cycle = ((freq * t) + phase / (2.0 * np.pi) - cycle_shift) % 1.0
            base = _square_wave(cycle=cycle, duty_cycle=duty_cycle)
        elif atom_type == "triangle":
            cycle_shift = float(atom.get("cycle_shift", 0.0))
            duty_cycle = float(atom.get("duty_cycle", 0.5))
            cycle = ((freq * t) + phase / (2.0 * np.pi) - cycle_shift) % 1.0
            base = _asymmetric_triangle_wave(cycle=cycle, duty_cycle=duty_cycle)
        else:
            period = float(atom["period"])
            family = str(atom.get("family", "morlet"))
            scale = float(atom.get("scale", 0.18))
            shift = float(atom.get("shift", 0.0))
            theta = atom.get("theta", {})
            theta_dict = dict(theta) if isinstance(theta, dict) else {}
            base = _wavelet_atom(
                t=t,
                period=period,
                phase=phase,
                family=family,
                scale=scale,
                shift=shift,
                theta={str(k): float(v) for k, v in theta_dict.items()},
            )

        signal += effective_amplitude * base

    return signal


def sample_seasonality(
    t: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, SeasonalParams]:
    params = sample_seasonality_params(n=t.size, config=config, rng=rng)
    return render_seasonality(t=t, params=params), params
