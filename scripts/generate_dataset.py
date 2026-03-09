from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthtsad.config import load_config, load_config_from_raw
from synthtsad.pipeline import SyntheticGeneratorPipeline


def _cfg_from_overrides(
    path: Path,
    num_samples: int | None,
    seed: int | None,
    num_series: int | None,
    disable_trend: bool,
    disable_seasonality: bool,
    disable_noise: bool,
    disable_causal: bool,
    disable_local_anomaly: bool,
    disable_seasonal_anomaly: bool,
):
    cfg = load_config(path)
    has_toggle = any(
        [
            disable_trend,
            disable_seasonality,
            disable_noise,
            disable_causal,
            disable_local_anomaly,
            disable_seasonal_anomaly,
        ]
    )
    if num_samples is None and seed is None and num_series is None and not has_toggle:
        return cfg

    raw = dict(cfg.raw)
    if num_samples is not None:
        raw["num_samples"] = int(num_samples)
    if seed is not None:
        raw["seed"] = int(seed)
    if num_series is not None:
        raw["num_series"] = {"min": int(num_series), "max": int(num_series)}

    debug = dict(raw.get("debug", {}))
    if disable_trend:
        debug["enable_trend"] = False
    if disable_seasonality:
        debug["enable_seasonality"] = False
    if disable_noise:
        debug["enable_noise"] = False
    if disable_causal:
        debug["enable_causal"] = False
    if disable_local_anomaly:
        debug["enable_local_anomaly"] = False
    if disable_seasonal_anomaly:
        debug["enable_seasonal_anomaly"] = False
    if debug:
        raw["debug"] = debug

    return load_config_from_raw(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic TSAD corpus")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--num-samples", type=int, default=None, help="Override config.num_samples")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed")
    parser.add_argument("--num-series", type=int, default=None, help="Override sample series count (fixed min=max)")
    parser.add_argument("--disable-trend", action="store_true", help="Disable trend component")
    parser.add_argument("--disable-seasonality", action="store_true", help="Disable seasonality component")
    parser.add_argument("--disable-noise", action="store_true", help="Disable noise component")
    parser.add_argument("--disable-causal", action="store_true", help="Disable causal ARX stage")
    parser.add_argument("--disable-local-anomaly", action="store_true", help="Disable local anomaly injector")
    parser.add_argument("--disable-seasonal-anomaly", action="store_true", help="Disable seasonal anomaly injector")
    parser.add_argument("--print-config", action="store_true", help="Print final merged config and exit")
    args = parser.parse_args()

    cfg = _cfg_from_overrides(
        path=args.config,
        num_samples=args.num_samples,
        seed=args.seed,
        num_series=args.num_series,
        disable_trend=args.disable_trend,
        disable_seasonality=args.disable_seasonality,
        disable_noise=args.disable_noise,
        disable_causal=args.disable_causal,
        disable_local_anomaly=args.disable_local_anomaly,
        disable_seasonal_anomaly=args.disable_seasonal_anomaly,
    )
    if args.print_config:
        print(json.dumps(cfg.raw, ensure_ascii=False, indent=2))
        return

    pipeline = SyntheticGeneratorPipeline(cfg)
    pipeline.run(args.output)


if __name__ == "__main__":
    main()
