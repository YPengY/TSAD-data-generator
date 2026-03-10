from __future__ import annotations

import pytest

from synthtsad import load_config_from_raw


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"anomaly_sample_ratio": 1.2}, "anomaly_sample_ratio"),
        ({"causal": {"edge_density": -0.1}}, "causal.edge_density"),
        ({"causal": {"alpha_i_min": 0.8, "alpha_i_max": 0.4}}, "causal.alpha_i_max"),
        ({"anomaly": {"p_endogenous": -0.2}}, "anomaly.p_endogenous"),
        ({"num_series": {"min": 2, "max": 25}}, "num_series range"),
    ],
)
def test_invalid_configs_raise_value_error(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_config_from_raw(override)
