from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthtsad.components.seasonality import render_seasonality, sample_seasonality_params
from synthtsad.components.trend import render_trend, sample_trend_params
from synthtsad.config import load_config


def _load_trend_params(args: argparse.Namespace, n: int, rng: np.random.Generator) -> dict[str, Any]:
    if args.trend_params_json:
        return json.loads(args.trend_params_json)
    if args.trend_params_file:
        return json.loads(Path(args.trend_params_file).read_text(encoding="utf-8"))
    if args.sample_random:
        cfg = load_config(Path(args.config))
        return sample_trend_params(n=n, config=cfg, rng=rng)

    # Default deterministic fallback
    return {
        "trend_type": "increase",
        "k0": float(args.k0),
        "k1": float(args.k1),
    }


def _load_seasonality_params(args: argparse.Namespace, n: int, rng: np.random.Generator) -> dict[str, Any]:
    if args.seasonality_params_json:
        return json.loads(args.seasonality_params_json)
    if args.seasonality_params_file:
        return json.loads(Path(args.seasonality_params_file).read_text(encoding="utf-8"))
    if args.sample_random:
        cfg = load_config(Path(args.config))
        return sample_seasonality_params(n=n, config=cfg, rng=rng)

    # Default deterministic fallback
    period = float(args.default_period)
    return {
        "seasonality_type": "sine",
        "atoms": [
            {
                "type": "sine",
                "period": period,
                "frequency": 1.0 / period,
                "amplitude": float(args.default_amplitude),
                "phase": float(args.default_phase),
            }
        ],
    }


def _print_series(name: str, values: np.ndarray, head: int | None) -> None:
    count = values.size if head is None else min(int(head), values.size)
    print(f"=== {name}_values ===")
    print("index,value")
    for i in range(count):
        print(f"{i},{values[i]:.8f}")
    if count < values.size:
        print(f"... truncated: printed {count}/{values.size} points")


def _plot_components(
    t: np.ndarray,
    trend: np.ndarray,
    seasonality: np.ndarray,
    combined: np.ndarray,
    out_path: Path,
    show_window: bool,
    line_width: float,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install with: .\\.venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, trend, linewidth=line_width, color="#1f77b4")
    axes[0].set_title("Trend (Line)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, seasonality, linewidth=line_width, color="#2ca02c")
    axes[1].set_title("Seasonality (Line)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, combined, linewidth=line_width, color="#d62728")
    axes[2].set_title("Trend + Seasonality (Line)")
    axes[2].set_xlabel("t")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show_window:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend + Seasonality debug/visualization tool (no noise/causal/anomaly)."
    )
    parser.add_argument("--n", type=int, default=256, help="Series length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--head", type=int, default=20, help="Print first N values for each component")

    parser.add_argument("--sample-random", action="store_true", help="Sample both components from config")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.json"),
        help="Config path used with --sample-random",
    )

    parser.add_argument("--trend-params-json", type=str, default=None, help="JSON string for full trend params")
    parser.add_argument("--trend-params-file", type=str, default=None, help="Path to JSON trend params file")

    parser.add_argument(
        "--seasonality-params-json",
        type=str,
        default=None,
        help="JSON string for full seasonality params",
    )
    parser.add_argument(
        "--seasonality-params-file",
        type=str,
        default=None,
        help="Path to JSON seasonality params file",
    )

    # deterministic fallback params when not using random/json
    parser.add_argument("--k0", type=float, default=0.0, help="Fallback trend intercept")
    parser.add_argument("--k1", type=float, default=0.02, help="Fallback trend slope")
    parser.add_argument("--default-period", type=float, default=24.0, help="Fallback seasonality period")
    parser.add_argument("--default-amplitude", type=float, default=1.0, help="Fallback seasonality amplitude")
    parser.add_argument("--default-phase", type=float, default=0.0, help="Fallback seasonality phase")

    parser.add_argument("--plot", action="store_true", help="Save a 3-panel visualization")
    parser.add_argument(
        "--plot-out",
        type=str,
        default=str(ROOT / "outputs" / "trend_seasonality_panel.png"),
        help="Output image path",
    )
    parser.add_argument("--line-width", type=float, default=1.5, help="Line width")
    parser.add_argument(
        "--point-size",
        type=float,
        default=None,
        help="Deprecated alias of --line-width (kept for backward compatibility)",
    )
    parser.add_argument("--show-plot", action="store_true", help="Show interactive window")

    args = parser.parse_args()

    n = int(args.n)
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(args.seed)

    trend_params = _load_trend_params(args=args, n=n, rng=rng)
    seasonality_params = _load_seasonality_params(args=args, n=n, rng=rng)

    trend = render_trend(t=t, params=trend_params)
    seasonality = render_seasonality(t=t, params=seasonality_params)
    combined = trend + seasonality

    print("=== trend_params ===")
    print(json.dumps(trend_params, ensure_ascii=False, indent=2))
    print("=== seasonality_params ===")
    print(json.dumps(seasonality_params, ensure_ascii=False, indent=2))

    summary = {
        "length": n,
        "trend": {
            "min": float(np.min(trend)),
            "max": float(np.max(trend)),
            "mean": float(np.mean(trend)),
            "std": float(np.std(trend)),
        },
        "seasonality": {
            "min": float(np.min(seasonality)),
            "max": float(np.max(seasonality)),
            "mean": float(np.mean(seasonality)),
            "std": float(np.std(seasonality)),
        },
        "combined": {
            "min": float(np.min(combined)),
            "max": float(np.max(combined)),
            "mean": float(np.mean(combined)),
            "std": float(np.std(combined)),
        },
    }
    print("=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    _print_series(name="trend", values=trend, head=args.head)
    _print_series(name="seasonality", values=seasonality, head=args.head)
    _print_series(name="combined", values=combined, head=args.head)

    if args.plot:
        plot_path = Path(args.plot_out)
        line_width = float(args.point_size) if args.point_size is not None else float(args.line_width)
        _plot_components(
            t=t,
            trend=trend,
            seasonality=seasonality,
            combined=combined,
            out_path=plot_path,
            show_window=bool(args.show_plot),
            line_width=line_width,
        )
        print(f"=== figure_saved ===\n{plot_path.resolve()}")


if __name__ == "__main__":
    main()
