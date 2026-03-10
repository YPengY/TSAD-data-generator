from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class IntRange:
    min: int
    max: int

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.min, self.max + 1))


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("Weights mapping must not be empty.")
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    return {k: float(v) / total for k, v in weights.items()}


def weighted_choice(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
    normalized = normalize_weights(weights)
    keys = list(normalized.keys())
    probs = np.array([normalized[k] for k in keys], dtype=float)
    idx = int(rng.choice(len(keys), p=probs))
    return keys[idx]


def ensure_int_range(raw: Mapping[str, int], name: str, min_value: int = 1) -> IntRange:
    if "min" not in raw or "max" not in raw:
        raise ValueError(f"{name} must have min/max")
    r = IntRange(int(raw["min"]), int(raw["max"]))
    if r.min < min_value or r.max < r.min:
        raise ValueError(f"Invalid range for {name}: {r}")
    return r


def ensure_non_negative_int(value: int, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {number}")
    return number


def ensure_positive_int(value: int, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be > 0, got {number}")
    return number


def ensure_non_negative_float(value: float, name: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0, got {number}")
    return number


def ensure_probability(value: float, name: str) -> float:
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {number}")
    return number


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
