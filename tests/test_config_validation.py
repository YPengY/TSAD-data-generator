from __future__ import annotations

import pytest

from synthtsad import load_config_from_raw
from synthtsad.config import migrate_legacy_config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"anomaly_sample_ratio": 1.2}, "anomaly_sample_ratio"),
        ({"causal": {"edge_density": -0.1}}, "causal.edge_density"),
        ({"causal": {"alpha_i_min": 0.8, "alpha_i_max": 0.4}}, "causal.alpha_i_max"),
        ({"anomaly": {"local": {"defaults": {"endogenous_p": -0.2}}}}, "anomaly.local.defaults.endogenous_p"),
        ({"anomaly": {"local": {"defaults": {"unexpected": 1}}}}, "anomaly.local.defaults contains unsupported keys"),
        ({"anomaly": {"seasonal": {"activation_p": 1.2}}}, "anomaly.seasonal.activation_p"),
        ({"num_features": {"min": 2, "max": 2}}, "config contains unsupported keys"),
        ({"num_series": {"min": 2, "max": 25}}, "num_series range"),
    ],
)
def test_invalid_configs_raise_value_error(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_config_from_raw(override)


def test_legacy_anomaly_config_is_rejected_by_runtime_loader() -> None:
    with pytest.raises(ValueError, match="anomaly contains unsupported keys"):
        load_config_from_raw(
            {
                "anomaly": {
                    "events_per_sample": {"min": 2, "max": 2},
                    "seasonal_events_per_sample": {"min": 1, "max": 1},
                    "window_length": {"min": 12, "max": 24},
                    "local_types": ["upward_spike", "shake"],
                    "seasonal_types": ["phase_shift", "waveform_inversion"],
                    "p_endogenous": 0.7,
                    "p_endogenous_seasonal": 0.15,
                    "p_use_seasonal_injector": 0.6,
                }
            }
        )


def test_legacy_config_can_be_migrated_explicitly() -> None:
    migrated = migrate_legacy_config(
        {
            "num_features": {"min": 3, "max": 3},
            "multivariate_flag": True,
            "anomaly": {
                "events_per_sample": {"min": 2, "max": 2},
                "seasonal_events_per_sample": {"min": 1, "max": 1},
                "window_length": {"min": 12, "max": 24},
                "local_types": ["upward_spike", "shake"],
                "seasonal_types": ["phase_shift", "waveform_inversion"],
                "p_endogenous": 0.7,
                "p_endogenous_seasonal": 0.15,
                "p_use_seasonal_injector": 0.6,
            },
        }
    )

    cfg = load_config_from_raw(migrated)
    assert cfg.raw["num_series"] == {"min": 3, "max": 3}
    assert cfg.raw["anomaly"]["local"]["budget"]["events_per_sample"] == {"min": 2, "max": 2}
    assert cfg.raw["anomaly"]["seasonal"]["budget"]["events_per_sample"] == {"min": 1, "max": 1}
    assert cfg.raw["anomaly"]["local"]["defaults"]["endogenous_p"] == 0.7
    assert cfg.raw["anomaly"]["seasonal"]["defaults"]["endogenous_p"] == 0.15
    assert cfg.raw["anomaly"]["seasonal"]["activation_p"] == 0.6
    assert cfg.raw["anomaly"]["local"]["type_weights"]["upward_spike"] == 1.0
    assert cfg.raw["anomaly"]["local"]["type_weights"]["downward_spike"] == 0.0
