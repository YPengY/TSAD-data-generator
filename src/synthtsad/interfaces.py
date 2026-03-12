from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

import numpy as np


class _LinearTrendParamsBase(TypedDict):
    k0: float
    k1: float


class IncreaseTrendParams(_LinearTrendParamsBase):
    trend_type: Literal["increase"]


class DecreaseTrendParams(_LinearTrendParamsBase):
    trend_type: Literal["decrease"]


class KeepSteadyTrendParams(_LinearTrendParamsBase):
    trend_type: Literal["keep_steady"]


class MultipleTrendParams(_LinearTrendParamsBase):
    trend_type: Literal["multiple"]
    change_points: list[int]
    slope_deltas: list[float]


class ArimaTrendParams(TypedDict):
    trend_type: Literal["arima"]
    p: int
    d: int
    q: int
    phi: list[float]
    theta: list[float]
    sigma: float
    base_level: float
    stochastic_seed: int


LinearTrendParams: TypeAlias = IncreaseTrendParams | DecreaseTrendParams | KeepSteadyTrendParams
TrendParams: TypeAlias = LinearTrendParams | MultipleTrendParams | ArimaTrendParams


class NoiseWindow(TypedDict):
    start: int
    end: int
    v: float


class NoiseParams(TypedDict):
    noise_level: str
    sigma0: float
    volatility_windows: list[NoiseWindow]
    stochastic_seed: int


class SeasonalityAtomBase(TypedDict):
    type: str
    period: float
    frequency: float
    amplitude: float
    phase: float
    modulation_depth: float
    modulation_frequency: float
    modulation_phase: float


class PeriodicAtom(SeasonalityAtomBase, total=False):
    duty_cycle: float
    cycle_shift: float


class WaveletAtomBase(SeasonalityAtomBase):
    family: str
    scale: float
    shift: float
    theta: dict[str, float]


class WaveletAtom(WaveletAtomBase, total=False):
    contrastive_group: int
    contrastive_role: Literal["anchor", "paired"]
    contrastive_changed: str


SeasonalityAtom: TypeAlias = PeriodicAtom | WaveletAtom


AtomicSeasonalityType: TypeAlias = Literal["sine", "square", "triangle", "wavelet"]
SeasonalityType: TypeAlias = Literal["none"] | AtomicSeasonalityType


class NoneSeasonalityParams(TypedDict):
    seasonality_type: Literal["none"]
    atoms: list[SeasonalityAtom]


class AtomicSeasonalityParams(TypedDict):
    seasonality_type: AtomicSeasonalityType
    atoms: list[SeasonalityAtom]


SeasonalParams: TypeAlias = NoneSeasonalityParams | AtomicSeasonalityParams


class Stage1NodeParams(TypedDict):
    node: int
    trend: TrendParams
    seasonality: SeasonalParams
    noise: NoiseParams


class ARXParams(TypedDict):
    a: list[float]
    alpha: list[float]
    bias: list[float]
    lag: list[list[int]]
    gain: list[list[float]]
    max_lag: int


class DisabledARXParams(TypedDict):
    disabled: Literal[True]


ARXModelParams: TypeAlias = ARXParams | DisabledARXParams


class RangeConfig(TypedDict):
    min: float | int
    max: float | int


class LocalSpecBase(TypedDict):
    enabled: bool


class WindowedLocalSpec(LocalSpecBase):
    window_length: RangeConfig


class SpikeLocalSpec(WindowedLocalSpec):
    amplitude: RangeConfig
    half_width: RangeConfig


class BurstSpikeLocalSpec(WindowedLocalSpec):
    spike_count: RangeConfig
    stride: RangeConfig
    amplitude: RangeConfig
    half_width: RangeConfig


class WideSpikeLocalSpec(WindowedLocalSpec):
    amplitude: RangeConfig
    rise_length: RangeConfig
    fall_length: RangeConfig


class OutlierLocalSpec(LocalSpecBase):
    amplitude: RangeConfig
    min_abs_amplitude: float


class SuddenShiftLocalSpec(WindowedLocalSpec):
    amplitude: RangeConfig
    kappa: RangeConfig


class PlateauLocalSpec(WindowedLocalSpec, total=False):
    positive_p: float
    amplitude: RangeConfig


class AsymmetricLocalSpec(WindowedLocalSpec):
    amplitude: RangeConfig
    rapid_tau: RangeConfig
    slow_tau: RangeConfig


class SpikeLevelLocalSpec(SpikeLocalSpec):
    shift_magnitude: RangeConfig


class ShakeLocalSpec(WindowedLocalSpec):
    amplitude: RangeConfig
    frequency: RangeConfig
    phase: RangeConfig


LocalTypeSpec: TypeAlias = (
    SpikeLocalSpec
    | BurstSpikeLocalSpec
    | WideSpikeLocalSpec
    | OutlierLocalSpec
    | SuddenShiftLocalSpec
    | PlateauLocalSpec
    | AsymmetricLocalSpec
    | SpikeLevelLocalSpec
    | ShakeLocalSpec
)


class SeasonalSpecBase(TypedDict):
    enabled: bool
    applies_to: list[str]
    window_length: RangeConfig


class ScaleSeasonalSpec(SeasonalSpecBase):
    scale: RangeConfig


class FactorSeasonalSpec(SeasonalSpecBase):
    factor: RangeConfig


class NoiseInjectionSeasonalSpec(SeasonalSpecBase):
    noise_scale: RangeConfig


class WaveformChangeSeasonalSpec(SeasonalSpecBase):
    target_type_weights: dict[str, float]


class DeltaPhaseSeasonalSpec(SeasonalSpecBase):
    delta_phase: RangeConfig


class AddHarmonicSeasonalSpec(SeasonalSpecBase):
    order: RangeConfig
    amplitude_scale: RangeConfig
    phase: RangeConfig


class ModulationDepthSeasonalSpec(SeasonalSpecBase):
    depth: RangeConfig


class ModulationPhaseSeasonalSpec(SeasonalSpecBase):
    phase: RangeConfig


class DeltaCycleSeasonalSpec(SeasonalSpecBase):
    delta_cycle: RangeConfig


class DeltaShiftSeasonalSpec(SeasonalSpecBase):
    delta_shift: RangeConfig


class WaveletFamilySeasonalSpec(SeasonalSpecBase):
    target_family_weights: dict[str, float]


class AddWaveletSeasonalSpec(SeasonalSpecBase):
    family_weights: dict[str, float]
    period: RangeConfig
    amplitude: RangeConfig
    phase: RangeConfig
    scale: RangeConfig
    shift: RangeConfig


SeasonalTypeSpec: TypeAlias = (
    SeasonalSpecBase
    | ScaleSeasonalSpec
    | FactorSeasonalSpec
    | NoiseInjectionSeasonalSpec
    | WaveformChangeSeasonalSpec
    | DeltaPhaseSeasonalSpec
    | AddHarmonicSeasonalSpec
    | ModulationDepthSeasonalSpec
    | ModulationPhaseSeasonalSpec
    | DeltaCycleSeasonalSpec
    | DeltaShiftSeasonalSpec
    | WaveletFamilySeasonalSpec
    | AddWaveletSeasonalSpec
)


class OutlierEventParams(TypedDict):
    amplitude: float
    t0: int


class SpikeEventParams(TypedDict):
    amplitude: float
    center: int
    half_width: int


class SpikeLevelEventParams(SpikeEventParams):
    shift_start: int
    shift_magnitude: float


class BurstSpikeEventParams(TypedDict):
    spike_count: int
    stride: int
    amplitudes: list[float]
    widths: list[int]
    t0: int


class WideSpikeEventParams(TypedDict):
    amplitude: float
    rise_length: int
    fall_length: int


class SuddenShiftEventParams(TypedDict):
    amplitude: float
    t0: int
    kappa: float


class PlateauEventParams(TypedDict):
    amplitude: float
    sign: float


class AsymmetricEventParams(TypedDict):
    amplitude: float
    peak: int
    rise_tau: float
    fall_tau: float
    sign: float


class ShakeEventParams(TypedDict):
    amplitude: float
    freq: float
    phase: float


LocalEventParams: TypeAlias = (
    OutlierEventParams
    | SpikeEventParams
    | SpikeLevelEventParams
    | BurstSpikeEventParams
    | WideSpikeEventParams
    | SuddenShiftEventParams
    | PlateauEventParams
    | AsymmetricEventParams
    | ShakeEventParams
)


class EmptyEventParams(TypedDict):
    pass


class ScaleEventParams(TypedDict):
    scale: float


class FactorEventParams(TypedDict):
    factor: float


class NoiseInjectionEventParams(TypedDict):
    noise_scale: float


class TargetTypeEventParams(TypedDict):
    target_type: Literal["sine", "square", "triangle"]


class DeltaPhaseEventParams(TypedDict):
    delta_phase: float


class HarmonicEventParams(TypedDict):
    amplitude: float
    period: float
    phase: float


class IndexedEventParams(TypedDict):
    index: int


class IndexedDeltaPhaseEventParams(IndexedEventParams):
    delta_phase: float


class IndexedDepthEventParams(IndexedEventParams):
    depth: float


class IndexedFrequencyEventParams(IndexedEventParams):
    frequency: float


class IndexedPhaseEventParams(IndexedEventParams):
    phase: float


class DeltaCycleEventParams(TypedDict):
    delta_cycle: float


class IndexedTargetFamilyEventParams(IndexedEventParams):
    target_family: str


class IndexedFactorEventParams(IndexedEventParams):
    factor: float


class IndexedDeltaShiftEventParams(IndexedEventParams):
    delta_shift: float


class AddWaveletEventParams(TypedDict):
    family: str
    period: float
    amplitude: float
    phase: float
    scale: float
    shift: float


SeasonalEventParams: TypeAlias = (
    EmptyEventParams
    | ScaleEventParams
    | FactorEventParams
    | NoiseInjectionEventParams
    | TargetTypeEventParams
    | DeltaPhaseEventParams
    | HarmonicEventParams
    | IndexedEventParams
    | IndexedDeltaPhaseEventParams
    | IndexedDepthEventParams
    | IndexedFrequencyEventParams
    | IndexedPhaseEventParams
    | DeltaCycleEventParams
    | IndexedTargetFamilyEventParams
    | IndexedFactorEventParams
    | IndexedDeltaShiftEventParams
    | AddWaveletEventParams
)


EventParams: TypeAlias = LocalEventParams | SeasonalEventParams
EventFamily: TypeAlias = Literal["local", "seasonal"]


class Stage3EventRecord(TypedDict):
    anomaly_type: str
    node: int
    t_start: int
    t_end: int
    params: EventParams
    is_endogenous: bool
    root_cause_node: int | None
    affected_nodes: list[int]
    family: EventFamily
    target_component: str


class Stage3SampledEvents(TypedDict):
    local: list[Stage3EventRecord]
    seasonal: list[Stage3EventRecord]


class SampleRunMetadata(TypedDict):
    seed_state: str


class Stage1Metadata(TypedDict):
    params: list[Stage1NodeParams]


class Stage2Metadata(TypedDict):
    params: ARXModelParams


class Stage3Metadata(TypedDict):
    sampled_events: Stage3SampledEvents


class GenerationMetadata(TypedDict):
    sample: SampleRunMetadata
    stage1: Stage1Metadata
    stage2: Stage2Metadata
    stage3: Stage3Metadata


class EventSummary(TypedDict):
    total: int
    local: int
    seasonal: int
    endogenous: int
    target_components: dict[str, int]


class LabelPayload(TypedDict):
    point_mask: np.ndarray
    point_mask_any: np.ndarray
    events: list[Stage3EventRecord]
    root_cause: list[int]
    affected_nodes: dict[str, list[int]]
    is_anomalous_sample: int
    summary: EventSummary
