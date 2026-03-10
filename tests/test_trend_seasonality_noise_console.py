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

from synthtsad.components.noise import render_noise, sample_noise_params
from synthtsad.components.seasonality import render_seasonality, sample_seasonality_params
from synthtsad.components.trend import render_trend, sample_trend_params
from synthtsad.config import GeneratorConfig, load_config


def _parse_noise_windows(raw: str) -> list[dict[str, int | float]]:
    text = raw.strip()
    if not text:
        return []

    windows: list[dict[str, int | float]] = []
    chunks = [c.strip() for c in text.split(";") if c.strip()]
    for chunk in chunks:
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --noise-windows chunk: {chunk}. Expected format start:end:v, e.g. 20:50:0.6"
            )
        start = int(parts[0])
        end = int(parts[1])
        v = float(parts[2])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid noise window [{start}, {end}); require 0 <= start < end")
        windows.append({"start": start, "end": end, "v": v})
    return windows


def _series_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _print_series(name: str, values: np.ndarray, head: int | None) -> None:
    count = values.size if head is None else min(int(head), values.size)
    print(f"=== {name}_values ===")
    print("index,value")
    for i in range(count):
        print(f"{i},{values[i]:.8f}")
    if count < values.size:
        print(f"... truncated: printed {count}/{values.size} points")


def _load_trend_params(
    args: argparse.Namespace,
    n: int,
    rng: np.random.Generator,
    cfg: GeneratorConfig | None,
) -> dict[str, Any]:
    if args.trend_params_json:
        return json.loads(args.trend_params_json)
    if args.trend_params_file:
        return json.loads(Path(args.trend_params_file).read_text(encoding="utf-8"))
    if args.sample_random:
        if cfg is None:
            raise ValueError("Config is required for --sample-random")
        return sample_trend_params(n=n, config=cfg, rng=rng)
    return {
        "trend_type": "increase",
        "k0": float(args.k0),
        "k1": float(args.k1),
    }


def _load_seasonality_params(
    args: argparse.Namespace,
    n: int,
    rng: np.random.Generator,
    cfg: GeneratorConfig | None,
) -> dict[str, Any]:
    if args.seasonality_params_json:
        return json.loads(args.seasonality_params_json)
    if args.seasonality_params_file:
        return json.loads(Path(args.seasonality_params_file).read_text(encoding="utf-8"))
    if args.sample_random:
        if cfg is None:
            raise ValueError("Config is required for --sample-random")
        return sample_seasonality_params(n=n, config=cfg, rng=rng)

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


def _load_noise_params(
    args: argparse.Namespace,
    n: int,
    rng: np.random.Generator,
    cfg: GeneratorConfig | None,
) -> dict[str, Any]:
    if args.noise_params_json:
        return json.loads(args.noise_params_json)
    if args.noise_params_file:
        return json.loads(Path(args.noise_params_file).read_text(encoding="utf-8"))
    if args.sample_random:
        if cfg is None:
            raise ValueError("Config is required for --sample-random")
        return sample_noise_params(n=n, config=cfg, rng=rng)

    stochastic_seed = int(args.noise_stochastic_seed) if args.noise_stochastic_seed is not None else int(args.seed + 1)
    return {
        "noise_level": "manual",
        "sigma0": float(args.noise_sigma0),
        "volatility_windows": _parse_noise_windows(args.noise_windows),
        "stochastic_seed": stochastic_seed,
    }


def _plot_components(
    t: np.ndarray,
    trend: np.ndarray,
    seasonality: np.ndarray,
    noise: np.ndarray,
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    ax00 = axes[0, 0]
    ax00.plot(t, trend, color="#1f77b4", linewidth=line_width)
    ax00.set_title("Trend")
    ax00.grid(alpha=0.3)

    ax01 = axes[0, 1]
    ax01.plot(t, seasonality, color="#2ca02c", linewidth=line_width)
    ax01.set_title("Seasonality")
    ax01.grid(alpha=0.3)

    ax10 = axes[1, 0]
    ax10.plot(t, noise, color="#ff7f0e", linewidth=line_width)
    ax10.set_title("Noise")
    ax10.set_xlabel("t")
    ax10.grid(alpha=0.3)

    ax11 = axes[1, 1]
    ax11.plot(t, trend, color="#1f77b4", linewidth=max(1.0, line_width * 0.9), alpha=0.9, label="trend")
    ax11.plot(
        t,
        seasonality,
        color="#2ca02c",
        linewidth=max(1.0, line_width * 0.9),
        alpha=0.9,
        label="seasonality",
    )
    ax11.plot(t, noise, color="#ff7f0e", linewidth=max(1.0, line_width * 0.9), alpha=0.9, label="noise")
    ax11.plot(t, combined, color="#111111", linewidth=max(1.2, line_width * 1.2), alpha=0.95, label="sum")
    ax11.set_title("All Components + Sum")
    ax11.set_xlabel("t")
    ax11.grid(alpha=0.3)
    ax11.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show_window:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend + Seasonality + Noise debug/visualization tool (no causal/anomaly).",
    )
    parser.add_argument("--n", type=int, default=256, help="Series length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parameter sampling")
    parser.add_argument("--head", type=int, default=20, help="Print first N points per component")

    parser.add_argument("--sample-random", action="store_true", help="Sample all components from config")
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

    parser.add_argument("--noise-params-json", type=str, default=None, help="JSON string for full noise params")
    parser.add_argument("--noise-params-file", type=str, default=None, help="Path to JSON noise params file")

    parser.add_argument("--k0", type=float, default=0.0, help="Fallback trend intercept")
    parser.add_argument("--k1", type=float, default=0.02, help="Fallback trend slope")
    parser.add_argument("--default-period", type=float, default=24.0, help="Fallback seasonality period")
    parser.add_argument("--default-amplitude", type=float, default=1.0, help="Fallback seasonality amplitude")
    parser.add_argument("--default-phase", type=float, default=0.0, help="Fallback seasonality phase")

    parser.add_argument("--noise-sigma0", type=float, default=0.08, help="Fallback noise sigma0")
    parser.add_argument(
        "--noise-windows",
        type=str,
        default="",
        help="Fallback volatility windows as start:end:v;start:end:v",
    )
    parser.add_argument(
        "--noise-stochastic-seed",
        type=int,
        default=None,
        help="Fallback stochastic seed for rendering noise",
    )

    parser.add_argument("--plot", action="store_true", help="Save a 2x2 visualization panel")
    parser.add_argument(
        "--plot-out",
        type=str,
        default=str(ROOT / "outputs" / "trend_seasonality_noise_panel.png"),
        help="Output image path",
    )
    parser.add_argument("--line-width", type=float, default=1.5, help="Line width")
    parser.add_argument("--show-plot", action="store_true", help="Show interactive window")

    args = parser.parse_args()

    n = int(args.n)
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(args.seed)
    cfg = load_config(Path(args.config)) if args.sample_random else None

    trend_params = _load_trend_params(args=args, n=n, rng=rng, cfg=cfg)
    seasonality_params = _load_seasonality_params(args=args, n=n, rng=rng, cfg=cfg)
    noise_params = _load_noise_params(args=args, n=n, rng=rng, cfg=cfg)

    trend = render_trend(t=t, params=trend_params)
    seasonality = render_seasonality(t=t, params=seasonality_params)
    noise = render_noise(n=n, params=noise_params)
    combined = trend + seasonality + noise

    print("=== trend_params ===")
    print(json.dumps(trend_params, ensure_ascii=False, indent=2))
    print("=== seasonality_params ===")
    print(json.dumps(seasonality_params, ensure_ascii=False, indent=2))
    print("=== noise_params ===")
    print(json.dumps(noise_params, ensure_ascii=False, indent=2))

    summary = {
        "length": n,
        "trend": _series_stats(trend),
        "seasonality": _series_stats(seasonality),
        "noise": _series_stats(noise),
        "combined": _series_stats(combined),
    }
    print("=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    _print_series(name="trend", values=trend, head=args.head)
    _print_series(name="seasonality", values=seasonality, head=args.head)
    _print_series(name="noise", values=noise, head=args.head)
    _print_series(name="combined", values=combined, head=args.head)

    if args.plot:
        plot_path = Path(args.plot_out)
        _plot_components(
            t=t,
            trend=trend,
            seasonality=seasonality,
            noise=noise,
            combined=combined,
            out_path=plot_path,
            show_window=bool(args.show_plot),
            line_width=float(args.line_width),
        )
        print(f"=== figure_saved ===\n{plot_path.resolve()}")


if __name__ == "__main__":
    main()
