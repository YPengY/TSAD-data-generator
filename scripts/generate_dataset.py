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


def _cfg_from_overrides(path: Path, num_samples: int | None, seed: int | None):
    cfg = load_config(path)
    if num_samples is None and seed is None:
        return cfg

    raw = dict(cfg.raw)
    if num_samples is not None:
        raw["num_samples"] = int(num_samples)
    if seed is not None:
        raw["seed"] = int(seed)
    return load_config_from_raw(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic TSAD corpus")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--num-samples", type=int, default=None, help="Override config.num_samples")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed")
    parser.add_argument("--print-config", action="store_true", help="Print final merged config and exit")
    args = parser.parse_args()

    cfg = _cfg_from_overrides(args.config, args.num_samples, args.seed)
    if args.print_config:
        print(json.dumps(cfg.raw, ensure_ascii=False, indent=2))
        return

    pipeline = SyntheticGeneratorPipeline(cfg)
    pipeline.run(args.output)


if __name__ == "__main__":
    main()
