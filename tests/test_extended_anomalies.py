from __future__ import annotations

import numpy as np

from synthtsad import load_config_from_raw
from synthtsad.anomaly.local import AnomalyEvent, LocalAnomalyInjector
from synthtsad.anomaly.seasonal import SeasonalAnomalyInjector
from synthtsad.causal.arx import ARXSystem
from synthtsad.causal.dag import CausalGraph
from synthtsad.pipeline import SyntheticGeneratorPipeline


def _make_cfg() -> object:
    return load_config_from_raw(
        {
            "num_samples": 1,
            "sequence_length": {"min": 64, "max": 64},
            "num_series": {"min": 2, "max": 2},
            "seed": 7,
        }
    )


def test_extended_local_templates_render_expected_shapes() -> None:
    cfg = _make_cfg()
    injector = LocalAnomalyInjector(cfg)
    x = np.zeros((48, 1), dtype=float)

    outlier = AnomalyEvent(
        anomaly_type="outlier",
        node=0,
        t_start=12,
        t_end=13,
        params={"amplitude": 2.5, "t0": 12},
        is_endogenous=False,
        root_cause_node=None,
        affected_nodes=[0],
    )
    x_out, _ = injector.apply_events(x_normal=x, events=[outlier])
    assert np.count_nonzero(np.abs(x_out[:, 0]) > 1e-8) == 1

    burst = AnomalyEvent(
        anomaly_type="continuous_upward_spikes",
        node=0,
        t_start=5,
        t_end=20,
        params={
            "spike_count": 3,
            "stride": 4,
            "amplitudes": [1.0, 1.5, 1.2],
            "widths": [1, 1, 2],
            "t0": 6,
        },
        is_endogenous=False,
        root_cause_node=None,
        affected_nodes=[0],
    )
    x_burst, _ = injector.apply_events(x_normal=x, events=[burst])
    assert np.count_nonzero(np.abs(x_burst[:, 0]) > 1e-8) >= 3
    assert float(np.max(x_burst[:, 0])) > 1.0

    interaction = AnomalyEvent(
        anomaly_type="decrease_after_upward_spike",
        node=0,
        t_start=8,
        t_end=48,
        params={
            "amplitude": 1.8,
            "center": 12,
            "half_width": 2,
            "shift_start": 18,
            "shift_magnitude": 0.9,
        },
        is_endogenous=False,
        root_cause_node=None,
        affected_nodes=[0],
    )
    x_interaction, _ = injector.apply_events(x_normal=x, events=[interaction])
    assert float(np.max(x_interaction[:18, 0])) > 0.0
    assert np.all(x_interaction[18:, 0] <= 1e-8)
    assert float(np.min(x_interaction[18:, 0])) < -0.5


def test_spike_level_sampling_keeps_local_event_window_with_persistent_shift() -> None:
    cfg = _make_cfg()
    injector = LocalAnomalyInjector(cfg)
    rng = np.random.default_rng(0)
    kind = "decrease_after_upward_spike"
    spec = cfg.anomaly.local.per_type[kind]

    t_start, t_end, params = injector._sample_template_spec(kind=kind, n=64, rng=rng, spec=spec)
    delta = injector._render_template(kind=kind, n=64, t_start=t_start, t_end=t_end, params=params)

    assert t_end < 64
    assert np.any(np.abs(delta[t_start:t_end]) > 1e-8)
    assert np.any(np.abs(delta[t_end:]) > 1e-8)


def test_local_handlers_cover_enabled_weighted_types() -> None:
    cfg = _make_cfg()
    injector = LocalAnomalyInjector(cfg)
    active_kinds = {
        kind
        for kind, spec in cfg.anomaly.local.per_type.items()
        if spec.get("enabled", True) and float(cfg.anomaly.local.type_weights.get(kind, 0.0)) > 0.0
    }

    for kind in active_kinds:
        assert injector._handler_for_kind(kind) is not None


def test_seasonal_handlers_cover_enabled_weighted_types() -> None:
    cfg = _make_cfg()
    injector = SeasonalAnomalyInjector(cfg)
    active_kinds = {
        kind
        for kind, spec in cfg.anomaly.seasonal.per_type.items()
        if spec.get("enabled", True)
        and float(cfg.anomaly.seasonal.type_weights.get(kind, 0.0)) > 0.0
    }

    for kind in active_kinds:
        assert injector._handler_for_kind(kind) is not None


def test_seasonal_param_transform_is_windowed_without_causal() -> None:
    cfg = _make_cfg()
    injector = SeasonalAnomalyInjector(cfg)
    t = np.arange(64, dtype=float)
    stage1_params = [
        {
            "node": 0,
            "seasonality": {
                "seasonality_type": "sine",
                "atoms": [
                    {
                        "type": "sine",
                        "period": 12.0,
                        "frequency": 1.0 / 12.0,
                        "amplitude": 1.0,
                        "phase": 0.0,
                        "modulation_depth": 0.0,
                        "modulation_frequency": 0.0,
                        "modulation_phase": 0.0,
                    }
                ],
            },
        }
    ]
    event = AnomalyEvent(
        anomaly_type="add_harmonic",
        node=0,
        t_start=10,
        t_end=28,
        params={"amplitude": 0.5, "period": 6.0, "phase": np.pi / 4},
        is_endogenous=False,
        root_cause_node=None,
        affected_nodes=[0],
        family="seasonal",
        target_component="seasonality",
    )

    x_out, realized = injector.apply_events(
        x_input=np.zeros((64, 1), dtype=float),
        events=[event],
        rng=np.random.default_rng(0),
        t=t,
        stage1_params=stage1_params,
        arx=None,
        arx_params=None,
    )

    assert np.allclose(x_out[:10, 0], 0.0)
    assert np.any(np.abs(x_out[10:28, 0]) > 1e-8)
    assert np.allclose(x_out[28:, 0], 0.0)
    assert realized[0].target_component == "seasonality"


def test_endogenous_seasonal_event_propagates_through_causal_response() -> None:
    cfg = _make_cfg()
    graph = CausalGraph(
        num_nodes=2,
        adjacency=np.array([[0, 1], [0, 0]], dtype=np.int8),
        topo_order=[0, 1],
        parents=[[], [0]],
    )
    arx = ARXSystem(cfg, graph)
    params = {
        "a": [0.0, 0.0],
        "alpha": [0.0, 1.0],
        "bias": [0.0, 0.0],
        "lag": [[0, 0], [0, 0]],
        "gain": [[0.0, 1.0], [0.0, 0.0]],
        "max_lag": 0,
    }
    stage1_params = [
        {
            "node": 0,
            "seasonality": {
                "seasonality_type": "sine",
                "atoms": [
                    {
                        "type": "sine",
                        "period": 8.0,
                        "frequency": 1.0 / 8.0,
                        "amplitude": 1.0,
                        "phase": 0.0,
                        "modulation_depth": 0.0,
                        "modulation_frequency": 0.0,
                        "modulation_phase": 0.0,
                    }
                ],
            },
        },
        {
            "node": 1,
            "seasonality": {
                "seasonality_type": "none",
                "atoms": [],
            },
        },
    ]
    event = AnomalyEvent(
        anomaly_type="waveform_inversion",
        node=0,
        t_start=8,
        t_end=24,
        params={},
        is_endogenous=True,
        root_cause_node=0,
        affected_nodes=[0],
        family="seasonal",
        target_component="seasonality",
    )
    injector = SeasonalAnomalyInjector(cfg)
    x_out, realized = injector.apply_events(
        x_input=np.zeros((64, 2), dtype=float),
        events=[event],
        rng=np.random.default_rng(0),
        t=np.arange(64, dtype=float),
        stage1_params=stage1_params,
        arx=arx,
        arx_params=params,
    )

    assert np.any(np.abs(x_out[:, 0]) > 1e-8)
    assert np.any(np.abs(x_out[:, 1]) > 1e-8)
    assert 1 in realized[0].affected_nodes


def test_endogenous_local_event_propagates_affected_nodes_through_causal_response() -> None:
    cfg = _make_cfg()
    graph = CausalGraph(
        num_nodes=2,
        adjacency=np.array([[0, 1], [0, 0]], dtype=np.int8),
        topo_order=[0, 1],
        parents=[[], [0]],
    )
    arx = ARXSystem(cfg, graph)
    params = {
        "a": [0.0, 0.0],
        "alpha": [0.0, 1.0],
        "bias": [0.0, 0.0],
        "lag": [[0, 0], [0, 0]],
        "gain": [[0.0, 1.0], [0.0, 0.0]],
        "max_lag": 0,
    }
    event = AnomalyEvent(
        anomaly_type="upward_spike",
        node=0,
        t_start=8,
        t_end=24,
        params={"amplitude": 1.2, "center": 12, "half_width": 2},
        is_endogenous=True,
        root_cause_node=0,
        affected_nodes=[0],
    )
    pipeline = SyntheticGeneratorPipeline(cfg)
    injector = LocalAnomalyInjector(cfg)

    pipeline._annotate_endogenous_local_events(
        n=64,
        d=2,
        local_injector=injector,
        events=[event],
        arx=arx,
        arx_params=params,
    )

    assert event.affected_nodes == [0, 1]


def test_arx_simulation_does_not_clip_latent_or_observed_values() -> None:
    cfg = _make_cfg()
    graph = CausalGraph(
        num_nodes=1,
        adjacency=np.array([[0]], dtype=np.int8),
        topo_order=[0],
        parents=[[]],
    )
    arx = ARXSystem(cfg, graph)
    params = {
        "a": [0.0],
        "alpha": [1.0],
        "bias": [50.0],
        "lag": [[0]],
        "gain": [[0.0]],
        "max_lag": 0,
    }

    x_out, state = arx.simulate_with_params(
        x_base=np.zeros((8, 1), dtype=float),
        n_steps=8,
        params=params,
    )

    assert float(np.max(state.z[:, 0])) == 50.0
    assert float(np.max(x_out[:, 0])) == 50.0


def test_arx_linear_response_is_additive() -> None:
    cfg = _make_cfg()
    graph = CausalGraph(
        num_nodes=2,
        adjacency=np.array([[0, 1], [0, 0]], dtype=np.int8),
        topo_order=[0, 1],
        parents=[[], [0]],
    )
    arx = ARXSystem(cfg, graph)
    params = {
        "a": [0.0, 0.0],
        "alpha": [0.0, 1.0],
        "bias": [0.0, 0.0],
        "lag": [[0, 0], [0, 0]],
        "gain": [[0.0, 2.0], [0.0, 0.0]],
        "max_lag": 0,
    }

    delta1 = np.zeros((8, 2), dtype=float)
    delta2 = np.zeros((8, 2), dtype=float)
    delta1[:, 0] = 15.0
    delta2[:, 0] = 25.0

    response1, _ = arx.simulate_linear_response(x_base=delta1, n_steps=8, params=params)
    response2, _ = arx.simulate_linear_response(x_base=delta2, n_steps=8, params=params)
    combined, _ = arx.simulate_linear_response(
        x_base=delta1 + delta2,
        n_steps=8,
        params=params,
    )

    assert np.allclose(combined, response1 + response2)
