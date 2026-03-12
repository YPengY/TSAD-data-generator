from __future__ import annotations

from copy import deepcopy
from typing import Literal, TypeAlias, cast

import numpy as np

from ..components.seasonality import render_seasonality
from ..config import GeneratorConfig
from ..interfaces import (
    AddHarmonicSeasonalSpec,
    AddWaveletEventParams,
    AddWaveletSeasonalSpec,
    ARXParams,
    AtomicSeasonalityParams,
    AtomicSeasonalityType,
    DeltaCycleEventParams,
    DeltaCycleSeasonalSpec,
    DeltaPhaseEventParams,
    DeltaPhaseSeasonalSpec,
    DeltaShiftSeasonalSpec,
    EmptyEventParams,
    FactorEventParams,
    FactorSeasonalSpec,
    HarmonicEventParams,
    IndexedDeltaPhaseEventParams,
    IndexedDeltaShiftEventParams,
    IndexedDepthEventParams,
    IndexedEventParams,
    IndexedFactorEventParams,
    IndexedFrequencyEventParams,
    IndexedPhaseEventParams,
    IndexedTargetFamilyEventParams,
    ModulationDepthSeasonalSpec,
    ModulationPhaseSeasonalSpec,
    NoiseInjectionEventParams,
    NoiseInjectionSeasonalSpec,
    PeriodicAtom,
    RangeConfig,
    ScaleEventParams,
    ScaleSeasonalSpec,
    SeasonalEventParams,
    SeasonalityAtom,
    SeasonalParams,
    SeasonalTypeSpec,
    Stage1NodeParams,
    TargetTypeEventParams,
    WaveformChangeSeasonalSpec,
    WaveletAtom,
    WaveletFamilySeasonalSpec,
)
from ..utils import IntRange, weighted_choice
from .local import AnomalyEvent

SampledSeasonalEvent: TypeAlias = SeasonalEventParams


class SeasonalAnomalyHandler:
    def sample(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        season_params: SeasonalParams,
        rng: np.random.Generator,
        spec: SeasonalTypeSpec,
    ) -> SampledSeasonalEvent:
        raise NotImplementedError

    def delta(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        t: np.ndarray,
        season_params: SeasonalParams,
        event: AnomalyEvent,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError


_SEASONAL_HANDLER_REGISTRY: dict[str, SeasonalAnomalyHandler] = {}


def register_seasonal_handler(*kinds: str):
    def decorator(handler_cls):
        handler = handler_cls()
        for kind in kinds:
            _SEASONAL_HANDLER_REGISTRY[kind] = handler
        return handler_cls

    return decorator


@register_seasonal_handler(
    "waveform_inversion",
    "amplitude_scaling",
    "frequency_change",
    "noise_injection",
)
class SignalTransformHandler(SeasonalAnomalyHandler):
    def sample(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        season_params: SeasonalParams,
        rng: np.random.Generator,
        spec: SeasonalTypeSpec,
    ) -> SampledSeasonalEvent:
        _ = season_params
        if kind == "waveform_inversion":
            empty_params: EmptyEventParams = {}
            return empty_params
        if kind == "amplitude_scaling":
            scale_spec = cast(ScaleSeasonalSpec, spec)
            scale_params: ScaleEventParams = {
                "scale": injector._sample_float(scale_spec["scale"], rng)
            }
            return scale_params
        if kind == "frequency_change":
            factor_spec = cast(FactorSeasonalSpec, spec)
            factor_params: FactorEventParams = {
                "factor": injector._sample_float(factor_spec["factor"], rng)
            }
            return factor_params
        noise_spec = cast(NoiseInjectionSeasonalSpec, spec)
        noise_params: NoiseInjectionEventParams = {
            "noise_scale": injector._sample_float(noise_spec["noise_scale"], rng)
        }
        return noise_params

    def delta(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        t: np.ndarray,
        season_params: SeasonalParams,
        event: AnomalyEvent,
        rng: np.random.Generator,
    ) -> np.ndarray:
        signal_params = cast(SeasonalEventParams, event.params)
        return injector._signal_window_delta(
            t=t,
            season_params=season_params,
            event=event,
            transform=lambda segment: self._transform_segment(
                injector=injector,
                kind=kind,
                segment=segment,
                params=signal_params,
                rng=rng,
            ),
        )

    def _transform_segment(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        segment: np.ndarray,
        params: SeasonalEventParams,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if kind == "waveform_inversion":
            return -segment
        if kind == "amplitude_scaling":
            scale_params = cast(ScaleEventParams, params)
            return float(scale_params["scale"]) * segment
        if kind == "frequency_change":
            factor_params = cast(FactorEventParams, params)
            return injector._frequency_warp(segment=segment, factor=float(factor_params["factor"]))
        noise_params = cast(NoiseInjectionEventParams, params)
        std = float(np.std(segment)) + 1e-4
        noise_scale = float(noise_params["noise_scale"])
        return segment + rng.normal(0.0, noise_scale * std, size=segment.size)


@register_seasonal_handler(
    "waveform_change",
    "phase_shift",
    "add_harmonic",
    "remove_harmonic",
    "modify_harmonic_phase",
    "modify_modulation_depth",
    "modify_modulation_frequency",
    "modify_modulation_phase",
)
class PeriodicParamTransformHandler(SeasonalAnomalyHandler):
    def sample(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        season_params: SeasonalParams,
        rng: np.random.Generator,
        spec: SeasonalTypeSpec,
    ) -> SampledSeasonalEvent:
        atoms = list(season_params["atoms"])
        season_type = str(season_params["seasonality_type"])

        if kind == "waveform_change":
            waveform_spec = cast(WaveformChangeSeasonalSpec, spec)
            choices = [value for value in ["sine", "square", "triangle"] if value != season_type]
            weights = {
                target: float(waveform_spec["target_type_weights"].get(target, 0.0))
                for target in choices
                if float(waveform_spec["target_type_weights"].get(target, 0.0)) > 0.0
            }
            target = weighted_choice(rng, weights) if weights else str(rng.choice(choices))
            target_type_params: TargetTypeEventParams = {
                "target_type": cast(Literal["sine", "square", "triangle"], target)
            }
            return target_type_params
        if kind == "phase_shift":
            phase_spec = cast(DeltaPhaseSeasonalSpec, spec)
            phase_shift_params: DeltaPhaseEventParams = {
                "delta_phase": injector._sample_float(phase_spec["delta_phase"], rng)
            }
            return phase_shift_params
        if kind == "add_harmonic":
            harmonic_spec = cast(AddHarmonicSeasonalSpec, spec)
            base_period = float(np.median([float(atom["period"]) for atom in atoms]))
            order = injector._sample_int(harmonic_spec["order"], rng, low_clip=2)
            harmonic_period = max(2.0, base_period / order)
            mean_amplitude = float(np.mean([float(atom["amplitude"]) for atom in atoms]))
            harmonic_params: HarmonicEventParams = {
                "amplitude": injector._sample_float(harmonic_spec["amplitude_scale"], rng)
                * max(mean_amplitude, 0.2),
                "period": harmonic_period,
                "phase": injector._sample_float(harmonic_spec["phase"], rng),
            }
            return harmonic_params
        if kind == "remove_harmonic":
            indexed_params: IndexedEventParams = {"index": int(rng.integers(0, len(atoms)))}
            return indexed_params
        if kind == "modify_harmonic_phase":
            phase_spec = cast(DeltaPhaseSeasonalSpec, spec)
            indexed_phase_params: IndexedDeltaPhaseEventParams = {
                "index": int(rng.integers(0, len(atoms))),
                "delta_phase": injector._sample_float(phase_spec["delta_phase"], rng),
            }
            return indexed_phase_params
        if kind == "modify_modulation_depth":
            depth_spec = cast(ModulationDepthSeasonalSpec, spec)
            indexed_depth_params: IndexedDepthEventParams = {
                "index": int(rng.integers(0, len(atoms))),
                "depth": injector._sample_float(
                    depth_spec["depth"], rng, low_clip=0.0, high_clip=1.0
                ),
            }
            return indexed_depth_params
        if kind == "modify_modulation_frequency":
            factor_spec = cast(FactorSeasonalSpec, spec)
            index = int(rng.integers(0, len(atoms)))
            atom_freq = float(atoms[index]["frequency"])
            indexed_frequency_params: IndexedFrequencyEventParams = {
                "index": index,
                "frequency": injector._sample_float(factor_spec["factor"], rng, low_clip=0.0)
                * atom_freq,
            }
            return indexed_frequency_params
        phase_spec = cast(ModulationPhaseSeasonalSpec, spec)
        indexed_modulation_phase_params: IndexedPhaseEventParams = {
            "index": int(rng.integers(0, len(atoms))),
            "phase": injector._sample_float(phase_spec["phase"], rng),
        }
        return indexed_modulation_phase_params

    def delta(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        t: np.ndarray,
        season_params: SeasonalParams,
        event: AnomalyEvent,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _ = rng
        params = cast(SeasonalEventParams, event.params)
        updated = cast(AtomicSeasonalityParams, injector._copy_season_params(season_params))
        atoms = cast(list[SeasonalityAtom], updated["atoms"])

        if kind == "waveform_change":
            type_params = cast(TargetTypeEventParams, params)
            target_type = cast(AtomicSeasonalityType, type_params["target_type"])
            updated["seasonality_type"] = target_type
            for atom in atoms:
                atom["type"] = target_type
                if target_type in {"square", "triangle"}:
                    periodic_atom = cast(PeriodicAtom, atom)
                    periodic_atom["duty_cycle"] = float(periodic_atom.get("duty_cycle", 0.5))
                    periodic_atom["cycle_shift"] = float(periodic_atom.get("cycle_shift", 0.0))
                else:
                    atom.pop("duty_cycle", None)
                    atom.pop("cycle_shift", None)
        elif kind == "phase_shift":
            phase_params = cast(DeltaPhaseEventParams, params)
            delta_phase = float(phase_params["delta_phase"])
            for atom in atoms:
                atom["phase"] = float(atom.get("phase", 0.0)) + delta_phase
        elif kind == "add_harmonic":
            harmonic_params = cast(HarmonicEventParams, params)
            atom_type = cast(AtomicSeasonalityType, updated["seasonality_type"])
            atoms.append(
                injector._make_periodic_atom(
                    atom_type=atom_type,
                    period=float(harmonic_params["period"]),
                    amplitude=float(harmonic_params["amplitude"]),
                    phase=float(harmonic_params["phase"]),
                )
            )
        elif kind == "remove_harmonic":
            indexed_params = cast(IndexedEventParams, params)
            if atoms:
                atoms.pop(int(indexed_params["index"]) % len(atoms))
        elif kind == "modify_harmonic_phase":
            phase_params = cast(IndexedDeltaPhaseEventParams, params)
            if atoms:
                atom = atoms[int(phase_params["index"]) % len(atoms)]
                atom["phase"] = float(atom.get("phase", 0.0)) + float(phase_params["delta_phase"])
        elif kind == "modify_modulation_depth":
            depth_params = cast(IndexedDepthEventParams, params)
            if atoms:
                atom = atoms[int(depth_params["index"]) % len(atoms)]
                atom["modulation_depth"] = float(np.clip(float(depth_params["depth"]), 0.0, 1.0))
        elif kind == "modify_modulation_frequency":
            frequency_params = cast(IndexedFrequencyEventParams, params)
            if atoms:
                atom = atoms[int(frequency_params["index"]) % len(atoms)]
                atom["modulation_frequency"] = max(0.0, float(frequency_params["frequency"]))
        else:
            phase_params = cast(IndexedPhaseEventParams, params)
            if atoms:
                atom = atoms[int(phase_params["index"]) % len(atoms)]
                atom["modulation_phase"] = float(phase_params["phase"])

        return injector._param_window_delta(
            t=t,
            season_params=season_params,
            event=event,
            updated_params=updated,
        )


@register_seasonal_handler("pulse_shift", "pulse_width_modulation")
class PulseTransformHandler(SeasonalAnomalyHandler):
    def sample(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        season_params: SeasonalParams,
        rng: np.random.Generator,
        spec: SeasonalTypeSpec,
    ) -> SampledSeasonalEvent:
        _ = season_params
        if kind == "pulse_shift":
            cycle_spec = cast(DeltaCycleSeasonalSpec, spec)
            cycle_params: DeltaCycleEventParams = {
                "delta_cycle": injector._sample_float(cycle_spec["delta_cycle"], rng)
            }
            return cycle_params
        factor_spec = cast(FactorSeasonalSpec, spec)
        factor_params: FactorEventParams = {
            "factor": injector._sample_float(factor_spec["factor"], rng)
        }
        return factor_params

    def delta(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        t: np.ndarray,
        season_params: SeasonalParams,
        event: AnomalyEvent,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _ = rng
        params = cast(SeasonalEventParams, event.params)
        updated = cast(AtomicSeasonalityParams, injector._copy_season_params(season_params))
        atoms = cast(list[SeasonalityAtom], updated["atoms"])

        if kind == "pulse_shift":
            cycle_params = cast(DeltaCycleEventParams, params)
            delta_cycle = float(cycle_params["delta_cycle"])
            for atom in atoms:
                periodic_atom = cast(PeriodicAtom, atom)
                periodic_atom["cycle_shift"] = (
                    float(periodic_atom.get("cycle_shift", 0.0)) + delta_cycle
                ) % 1.0
        else:
            factor_params = cast(FactorEventParams, params)
            factor = float(factor_params["factor"])
            for atom in atoms:
                periodic_atom = cast(PeriodicAtom, atom)
                duty = float(periodic_atom.get("duty_cycle", 0.5))
                periodic_atom["duty_cycle"] = float(np.clip(duty * factor, 0.1, 0.9))

        return injector._param_window_delta(
            t=t,
            season_params=season_params,
            event=event,
            updated_params=updated,
        )


@register_seasonal_handler(
    "wavelet_family_change",
    "wavelet_scale_change",
    "wavelet_shift_change",
    "wavelet_amplitude_change",
    "add_wavelet",
    "remove_wavelet",
)
class WaveletTransformHandler(SeasonalAnomalyHandler):
    def sample(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        season_params: SeasonalParams,
        rng: np.random.Generator,
        spec: SeasonalTypeSpec,
    ) -> SampledSeasonalEvent:
        atoms = list(season_params["atoms"])

        if kind == "wavelet_family_change":
            family_spec = cast(WaveletFamilySeasonalSpec, spec)
            index = int(rng.integers(0, len(atoms)))
            source = str(atoms[index].get("family", "morlet"))
            weights = {
                family: float(family_spec["target_family_weights"].get(family, 0.0))
                for family in injector.config.stage1.wavelet_family_weights
                if family != source
                and float(family_spec["target_family_weights"].get(family, 0.0)) > 0.0
            }
            choices = list(weights.keys()) or [
                family
                for family in injector.config.stage1.wavelet_family_weights
                if family != source
            ]
            target_family = weighted_choice(rng, weights) if weights else str(rng.choice(choices))
            family_change_params: IndexedTargetFamilyEventParams = {
                "index": index,
                "target_family": target_family,
            }
            return family_change_params
        if kind == "wavelet_scale_change":
            factor_spec = cast(FactorSeasonalSpec, spec)
            scale_change_params: IndexedFactorEventParams = {
                "index": int(rng.integers(0, len(atoms))),
                "factor": injector._sample_float(factor_spec["factor"], rng),
            }
            return scale_change_params
        if kind == "wavelet_shift_change":
            shift_spec = cast(DeltaShiftSeasonalSpec, spec)
            shift_change_params: IndexedDeltaShiftEventParams = {
                "index": int(rng.integers(0, len(atoms))),
                "delta_shift": injector._sample_float(shift_spec["delta_shift"], rng),
            }
            return shift_change_params
        if kind == "wavelet_amplitude_change":
            factor_spec = cast(FactorSeasonalSpec, spec)
            amplitude_change_params: IndexedFactorEventParams = {
                "index": int(rng.integers(0, len(atoms))),
                "factor": injector._sample_float(factor_spec["factor"], rng),
            }
            return amplitude_change_params
        if kind == "add_wavelet":
            wavelet_spec = cast(AddWaveletSeasonalSpec, spec)
            family_weights = {
                family: float(wavelet_spec["family_weights"].get(family, 0.0))
                for family in injector.config.stage1.wavelet_family_weights
                if float(wavelet_spec["family_weights"].get(family, 0.0)) > 0.0
            }
            add_wavelet_params: AddWaveletEventParams = {
                "family": weighted_choice(rng, family_weights)
                if family_weights
                else str(rng.choice(list(injector.config.stage1.wavelet_family_weights.keys()))),
                "period": injector._sample_float(wavelet_spec["period"], rng, low_clip=2.0),
                "amplitude": injector._sample_float(wavelet_spec["amplitude"], rng, low_clip=0.0),
                "phase": injector._sample_float(wavelet_spec["phase"], rng),
                "scale": injector._sample_float(wavelet_spec["scale"], rng, low_clip=1e-6),
                "shift": injector._sample_float(wavelet_spec["shift"], rng),
            }
            return add_wavelet_params
        remove_wavelet_params: IndexedEventParams = {"index": int(rng.integers(0, len(atoms)))}
        return remove_wavelet_params

    def delta(
        self,
        injector: SeasonalAnomalyInjector,
        kind: str,
        t: np.ndarray,
        season_params: SeasonalParams,
        event: AnomalyEvent,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _ = rng
        params = cast(SeasonalEventParams, event.params)
        updated = cast(AtomicSeasonalityParams, injector._copy_season_params(season_params))
        atoms = cast(list[SeasonalityAtom], updated["atoms"])

        if kind == "wavelet_family_change" and atoms:
            family_params = cast(IndexedTargetFamilyEventParams, params)
            atom = cast(WaveletAtom, atoms[int(family_params["index"]) % len(atoms)])
            atom["family"] = str(family_params["target_family"])
            atom["theta"] = {}
        elif kind == "wavelet_scale_change" and atoms:
            factor_params = cast(IndexedFactorEventParams, params)
            atom = cast(WaveletAtom, atoms[int(factor_params["index"]) % len(atoms)])
            atom["scale"] = float(
                np.clip(
                    float(atom.get("scale", 0.18)) * float(factor_params["factor"]),
                    *injector.config.stage1.wavelet_scale,
                )
            )
        elif kind == "wavelet_shift_change" and atoms:
            shift_params = cast(IndexedDeltaShiftEventParams, params)
            atom = cast(WaveletAtom, atoms[int(shift_params["index"]) % len(atoms)])
            atom["shift"] = float(
                (float(atom.get("shift", 0.0)) + float(shift_params["delta_shift"])) % 1.0
            )
        elif kind == "wavelet_amplitude_change" and atoms:
            factor_params = cast(IndexedFactorEventParams, params)
            atom = cast(WaveletAtom, atoms[int(factor_params["index"]) % len(atoms)])
            atom["amplitude"] = float(atom.get("amplitude", 1.0)) * float(factor_params["factor"])
        elif kind == "add_wavelet":
            wavelet_params = cast(AddWaveletEventParams, params)
            atoms.append(
                injector._make_wavelet_atom(
                    period=float(wavelet_params["period"]),
                    amplitude=float(wavelet_params["amplitude"]),
                    phase=float(wavelet_params["phase"]),
                    family=str(wavelet_params["family"]),
                    scale=float(wavelet_params["scale"]),
                    shift=float(wavelet_params["shift"]),
                    theta={},
                )
            )
        elif kind == "remove_wavelet" and atoms:
            indexed_params = cast(IndexedEventParams, params)
            atoms.pop(int(indexed_params["index"]) % len(atoms))

        return injector._param_window_delta(
            t=t,
            season_params=season_params,
            event=event,
            updated_params=updated,
        )


class SeasonalAnomalyInjector:
    """Stage 3 seasonal/contextual anomalies over seasonal components."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._handlers = _SEASONAL_HANDLER_REGISTRY

    def _seasonal_config(self):
        return self.config.anomaly.seasonal

    def _placement_policy(self):
        return self.config.anomaly.placement

    def _range_bounds(
        self,
        value: IntRange | RangeConfig | int | float,
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
        value: IntRange | RangeConfig | int,
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
        value: RangeConfig | int | float,
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
        self, n: int, rng: np.random.Generator, spec: SeasonalTypeSpec
    ) -> tuple[int, int]:
        family_window = self._seasonal_config().window_length
        base_window = spec.get("window_length", family_window)
        min_len, max_len = self._range_bounds(base_window, integer=True)
        max_len = min(int(max_len), n)
        min_len = min(max(1, int(min_len)), max_len)
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, n - length + 1))
        return start, start + length

    def _frequency_warp(self, segment: np.ndarray, factor: float) -> np.ndarray:
        m = segment.size
        if m <= 1:
            return segment.copy()
        src = np.arange(m, dtype=float)
        warped = (src * factor) % max(1, m - 1)
        return np.interp(warped, src, segment)

    @staticmethod
    def _copy_season_params(season_params: SeasonalParams) -> SeasonalParams:
        return cast(SeasonalParams, deepcopy(season_params))

    @staticmethod
    def _make_periodic_atom(
        atom_type: AtomicSeasonalityType,
        period: float,
        amplitude: float,
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

    @staticmethod
    def _make_wavelet_atom(
        period: float,
        amplitude: float,
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
            "family": family,
            "scale": scale,
            "shift": shift,
            "theta": theta,
            "modulation_depth": 0.0,
            "modulation_frequency": 0.0,
            "modulation_phase": 0.0,
        }

    @staticmethod
    def _event_window(size: int, event: AnomalyEvent) -> tuple[int, int]:
        start = max(0, int(event.t_start))
        end = min(size, int(event.t_end))
        return start, end

    def _signal_window_delta(
        self,
        t: np.ndarray,
        season_params: SeasonalParams,
        event: AnomalyEvent,
        transform,
    ) -> np.ndarray:
        baseline = render_seasonality(t=t, params=season_params)
        start, end = self._event_window(t.size, event)
        if start >= end:
            return np.zeros(t.size, dtype=float)
        delta = np.zeros(t.size, dtype=float)
        transformed_segment = transform(baseline[start:end])
        delta[start:end] = transformed_segment - baseline[start:end]
        return delta

    def _param_window_delta(
        self,
        t: np.ndarray,
        season_params: SeasonalParams,
        event: AnomalyEvent,
        updated_params: SeasonalParams,
    ) -> np.ndarray:
        baseline = render_seasonality(t=t, params=season_params)
        start, end = self._event_window(t.size, event)
        if start >= end:
            return np.zeros(t.size, dtype=float)
        transformed = render_seasonality(t=t, params=updated_params)
        delta = np.zeros(t.size, dtype=float)
        delta[start:end] = transformed[start:end] - baseline[start:end]
        return delta

    def _handler_for_kind(self, kind: str) -> SeasonalAnomalyHandler:
        try:
            return self._handlers[kind]
        except KeyError as exc:
            raise ValueError(f"No seasonal anomaly handler registered for kind: {kind}") from exc

    def _candidate_kinds(self, season_params: SeasonalParams) -> list[str]:
        season_type = str(season_params["seasonality_type"])
        atoms = list(season_params["atoms"])
        if season_type == "none" or not atoms:
            return []

        seasonal_cfg = self._seasonal_config()
        type_weights = seasonal_cfg.type_weights
        candidates: list[str] = []
        for kind, spec in seasonal_cfg.per_type.items():
            if (
                spec.get("enabled", True)
                and float(type_weights.get(kind, 0.0)) > 0.0
                and kind not in self._handlers
            ):
                raise ValueError(f"No seasonal anomaly handler registered for enabled kind: {kind}")
            if not spec.get("enabled", True) or float(type_weights.get(kind, 0.0)) <= 0.0:
                continue
            applies_to = [str(value) for value in spec.get("applies_to", [])]
            if applies_to and season_type not in applies_to:
                continue
            candidates.append(kind)
        return candidates

    def _eligible_nodes(
        self,
        d: int,
        stage1_params: list[Stage1NodeParams],
    ) -> list[int]:
        policy = self._seasonal_config().node_policy
        allowed = policy.allowed_nodes
        allowed_set = {int(node) for node in allowed} if allowed is not None else None
        nodes = list(range(d))
        if allowed_set is not None:
            nodes = [node for node in nodes if node in allowed_set]
        if policy.mode == "seasonal_eligible":
            nodes = [
                node
                for node in nodes
                if stage1_params[node]["seasonality"].get("seasonality_type", "none") != "none"
                and stage1_params[node]["seasonality"].get("atoms")
            ]
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

    def sample_events(
        self,
        n: int,
        d: int,
        rng: np.random.Generator,
        stage1_params: list[Stage1NodeParams] | None = None,
    ) -> list[AnomalyEvent]:
        seasonal_cfg = self._seasonal_config()
        if d == 0 or stage1_params is None or rng.random() > float(seasonal_cfg.activation_p):
            return []

        nodes = self._eligible_nodes(d=d, stage1_params=stage1_params)
        if not nodes:
            return []

        placements: dict[int, list[tuple[int, int]]] = {node: [] for node in nodes}
        counts: dict[int, int] = {node: 0 for node in nodes}
        count = seasonal_cfg.events_per_sample.sample(rng)
        events: list[AnomalyEvent] = []

        for _ in range(count):
            placed = False
            for _attempt in range(48):
                node = int(rng.choice(nodes))
                kinds = self._candidate_kinds(stage1_params[node]["seasonality"])
                if not kinds:
                    continue
                kind_weights = {
                    kind: float(seasonal_cfg.type_weights.get(kind, 0.0))
                    for kind in kinds
                    if float(seasonal_cfg.type_weights.get(kind, 0.0)) > 0.0
                }
                if not kind_weights:
                    continue
                kind = weighted_choice(rng, kind_weights)
                spec = seasonal_cfg.per_type[kind]
                t_start, t_end = self._sample_window(n, rng, spec=spec)
                if not self._can_place(
                    node=node, t_start=t_start, t_end=t_end, placements=placements, counts=counts
                ):
                    continue
                handler = self._handler_for_kind(kind)
                params = handler.sample(
                    injector=self,
                    kind=kind,
                    season_params=stage1_params[node]["seasonality"],
                    rng=rng,
                    spec=spec,
                )
                endogenous_p = float(seasonal_cfg.endogenous_p)
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
                        family="seasonal",
                        target_component=str(seasonal_cfg.target_component),
                    )
                )
                placed = True
                break
            if not placed:
                continue

        return events

    def apply_events(
        self,
        x_input: np.ndarray,
        events: list[AnomalyEvent],
        rng: np.random.Generator,
        t: np.ndarray,
        stage1_params: list[Stage1NodeParams],
        arx=None,
        arx_params: ARXParams | None = None,
    ) -> tuple[np.ndarray, list[AnomalyEvent]]:
        x_out = x_input.copy()
        realized: list[AnomalyEvent] = []
        n, d = x_out.shape

        for event in events:
            if event.family != "seasonal":
                raise ValueError(
                    f"SeasonalAnomalyInjector received non-seasonal event: {event.family}"
                )
            node = int(event.node)
            if node < 0 or node >= d:
                continue

            season_params = deepcopy(stage1_params[node]["seasonality"])
            handler = self._handler_for_kind(event.anomaly_type)
            delta = handler.delta(
                injector=self,
                kind=event.anomaly_type,
                t=t,
                season_params=season_params,
                event=event,
                rng=rng,
            )
            if not np.any(np.abs(delta) > 1e-8):
                realized.append(event)
                continue

            delta_matrix = np.zeros((n, d), dtype=float)
            delta_matrix[:, node] = delta
            affected_nodes = [node]

            if arx is not None and arx_params is not None:
                response, _ = arx.simulate_linear_response(
                    x_base=delta_matrix, n_steps=n, params=arx_params
                )
                if event.is_endogenous:
                    x_out += response
                    affected_nodes = (
                        np.where(np.any(np.abs(response) > 1e-8, axis=0))[0].astype(int).tolist()
                    ) or [node]
                else:
                    x_out[:, node] += response[:, node]
            else:
                x_out[:, node] += delta

            realized.append(
                AnomalyEvent(
                    anomaly_type=event.anomaly_type,
                    node=node,
                    t_start=event.t_start,
                    t_end=event.t_end,
                    params=event.params,
                    is_endogenous=event.is_endogenous,
                    root_cause_node=event.root_cause_node,
                    affected_nodes=affected_nodes,
                    family=event.family,
                    target_component=event.target_component,
                )
            )

        return x_out, realized

    def inject(
        self,
        x_input: np.ndarray,
        rng: np.random.Generator,
        t: np.ndarray,
        stage1_params: list[Stage1NodeParams],
        arx=None,
        arx_params: ARXParams | None = None,
    ) -> tuple[np.ndarray, list[AnomalyEvent]]:
        sampled = self.sample_events(
            n=x_input.shape[0], d=x_input.shape[1], rng=rng, stage1_params=stage1_params
        )
        return self.apply_events(
            x_input=x_input,
            events=sampled,
            rng=rng,
            t=t,
            stage1_params=stage1_params,
            arx=arx,
            arx_params=arx_params,
        )
