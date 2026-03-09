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

from synthtsad.components.trend import render_trend, sample_trend_params
from synthtsad.config import load_config


def _parse_float_list(raw: str) -> list[float]:
    s = raw.strip()
    if not s:
        return []
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def _parse_int_list(raw: str) -> list[int]:
    s = raw.strip()
    if not s:
        return []
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def _build_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    trend_type = args.trend_type
    if trend_type in {"increase", "decrease", "keep_steady"}:
        if args.k0 is None or args.k1 is None:
            raise ValueError("increase/decrease/keep_steady requires --k0 and --k1")
        return {"trend_type": trend_type, "k0": float(args.k0), "k1": float(args.k1)}

    if trend_type == "multiple":
        if args.k0 is None or args.k1 is None:
            raise ValueError("multiple requires --k0 and --k1")
        cps = _parse_int_list(args.change_points)
        deltas = _parse_float_list(args.slope_deltas)
        if len(cps) != len(deltas):
            raise ValueError("--change-points and --slope-deltas must have the same length")
        return {
            "trend_type": "multiple",
            "k0": float(args.k0),
            "k1": float(args.k1),
            "change_points": cps,
            "slope_deltas": deltas,
        }

    if trend_type == "arima":
        p = int(args.p)
        q = int(args.q)
        d_order = int(args.d)
        phi = _parse_float_list(args.phi)
        theta = _parse_float_list(args.theta)
        if len(phi) != p:
            raise ValueError(f"--phi expects {p} values for p={p}")
        if len(theta) != q:
            raise ValueError(f"--theta expects {q} values for q={q}")
        seed = int(args.stochastic_seed if args.stochastic_seed is not None else args.seed)
        sigma = float(args.sigma)
        return {
            "trend_type": "arima",
            "p": p,
            "d": d_order,
            "q": q,
            "phi": phi,
            "theta": theta,
            "sigma": sigma,
            "noise_scale": sigma,
            "base_level": float(args.base_level),
            "stochastic_seed": seed,
        }

    raise ValueError(f"Unsupported trend type: {trend_type}")


def _load_params(args: argparse.Namespace) -> dict[str, Any]:
    if args.params_json:
        return json.loads(args.params_json)

    if args.params_file:
        return json.loads(Path(args.params_file).read_text(encoding="utf-8"))

    if args.sample_random:
        cfg = load_config(Path(args.config))
        rng = np.random.default_rng(args.seed)
        return sample_trend_params(n=args.n, config=cfg, rng=rng)

    return _build_params_from_args(args)


def _print_series(trend: np.ndarray, head: int | None) -> None:
    if head is None:
        count = trend.size
    else:
        count = min(int(head), trend.size)

    print("index,value")
    for i in range(count):
        print(f"{i},{trend[i]:.8f}")
    if count < trend.size:
        print(f"... truncated: printed {count}/{trend.size} points")


def _plot_scatter(
    t: np.ndarray,
    trend: np.ndarray,
    out_path: Path,
    title: str,
    show_window: bool,
    point_size: float,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with: .\\.venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(t, trend, s=point_size)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("trend(t)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show_window:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend-only debugging tool. Generate trend values from custom parameters.",
    )
    parser.add_argument("--n", type=int, default=128, help="Trend length")
    parser.add_argument("--seed", type=int, default=42, help="Global seed (used for random sampling)")
    parser.add_argument("--head", type=int, default=None, help="Print first N points only; default prints all")
    parser.add_argument("--plot-scatter", action="store_true", help="Save scatter plot of generated trend")
    parser.add_argument(
        "--plot-out",
        type=str,
        default=str(ROOT / "outputs" / "trend_scatter.png"),
        help="Scatter plot output path",
    )
    parser.add_argument("--show-plot", action="store_true", help="Display plotting window")
    parser.add_argument("--point-size", type=float, default=10.0, help="Scatter point size")

    parser.add_argument("--params-json", type=str, default=None, help="Full trend params JSON string")
    parser.add_argument("--params-file", type=str, default=None, help="Path to JSON file containing full trend params")
    parser.add_argument("--sample-random", action="store_true", help="Sample trend params from config")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.json"), help="Config path")

    parser.add_argument(
        "--trend-type",
        choices=["increase", "decrease", "keep_steady", "multiple", "arima"],
        default="increase",
        help="Manual mode trend type (ignored when --params-json/--params-file/--sample-random is used)",
    )

    parser.add_argument("--k0", type=float, default=None, help="Linear/multiple intercept")
    parser.add_argument("--k1", type=float, default=None, help="Linear/multiple slope")
    parser.add_argument("--change-points", type=str, default="", help="multiple mode, e.g. 20,50,80")
    parser.add_argument("--slope-deltas", type=str, default="", help="multiple mode, e.g. 0.02,-0.01,0.03")

    parser.add_argument("--p", type=int, default=1, help="ARIMA p")
    parser.add_argument("--d", type=int, default=1, help="ARIMA d")
    parser.add_argument("--q", type=int, default=1, help="ARIMA q")
    parser.add_argument("--phi", type=str, default="0.4", help="AR coeff list, e.g. 0.4,-0.1")
    parser.add_argument("--theta", type=str, default="0.2", help="MA coeff list, e.g. 0.2")
    parser.add_argument("--sigma", type=float, default=0.05, help="ARIMA innovation std")
    parser.add_argument("--base-level", type=float, default=0.0, help="ARIMA level offset")
    parser.add_argument("--stochastic-seed", type=int, default=None, help="ARIMA stochastic seed")

    args = parser.parse_args()

    params = _load_params(args)
    t = np.arange(args.n, dtype=float)
    trend = render_trend(t=t, params=params)

    print("=== trend_params ===")
    print(json.dumps(params, ensure_ascii=False, indent=2))
    print("=== trend_summary ===")
    print(
        json.dumps(
            {
                "length": int(trend.size),
                "min": float(np.min(trend)),
                "max": float(np.max(trend)),
                "mean": float(np.mean(trend)),
                "std": float(np.std(trend)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== trend_values ===")
    _print_series(trend=trend, head=args.head)

    if args.plot_scatter:
        plot_path = Path(args.plot_out)
        _plot_scatter(
            t=t,
            trend=trend,
            out_path=plot_path,
            title=f"Trend Scatter ({params.get('trend_type', 'unknown')})",
            show_window=bool(args.show_plot),
            point_size=float(args.point_size),
        )
        print(f"=== scatter_saved ===\n{plot_path.resolve()}")


if __name__ == "__main__":
    main()
