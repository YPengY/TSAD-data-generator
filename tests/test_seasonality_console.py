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

from synthtsad.components.seasonality import (
    WAVELET_REGISTRY,
    render_seasonality,
    sample_seasonality_params,
)
from synthtsad.config import load_config


def _parse_float_list(raw: str) -> list[float]:
    s = raw.strip()
    if not s:
        return []
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def _parse_str_list(raw: str) -> list[str]:
    s = raw.strip()
    if not s:
        return []
    return [str(v.strip()) for v in s.split(",") if v.strip()]


def _parse_theta_list(raw: str) -> list[dict[str, float]]:
    s = raw.strip()
    if not s:
        return []
    items: list[dict[str, float]] = []
    for chunk in [x.strip() for x in s.split(";") if x.strip()]:
        obj = json.loads(chunk)
        if not isinstance(obj, dict):
            raise ValueError(f"Each theta JSON must be an object, got: {chunk}")
        items.append({str(k): float(v) for k, v in obj.items()})
    return items


def _parse_theta_kv_list(raw: str) -> list[dict[str, float]]:
    s = raw.strip()
    if not s:
        return []
    items: list[dict[str, float]] = []
    groups = [x.strip() for x in s.split(";") if x.strip()]
    for group in groups:
        theta: dict[str, float] = {}
        for pair in [x.strip() for x in group.split(",") if x.strip()]:
            if "=" not in pair:
                raise ValueError(f"Invalid theta key-value pair: {pair}; expected key=value")
            key, val = pair.split("=", 1)
            theta[str(key.strip())] = float(val.strip())
        items.append(theta)
    return items


def _expand_param(values: list[Any], count: int, default: Any, name: str) -> list[Any]:
    if not values:
        return [default for _ in range(count)]
    if len(values) == 1 and count > 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} expects either 1 value or {count} values, got {len(values)}")
    return values


def _build_preset_params(name: str) -> dict[str, Any]:
    if name == "sine_single":
        return {
            "seasonality_type": "sine",
            "atoms": [
                {
                    "type": "sine",
                    "period": 24.0,
                    "frequency": 1.0 / 24.0,
                    "amplitude": 1.0,
                    "phase": 0.0,
                },
            ],
        }
    if name == "sine_multi":
        return {
            "seasonality_type": "sine",
            "atoms": [
                {
                    "type": "sine",
                    "period": 24.0,
                    "frequency": 1.0 / 24.0,
                    "amplitude": 1.1,
                    "phase": 0.0,
                },
                {
                    "type": "sine",
                    "period": 12.0,
                    "frequency": 1.0 / 12.0,
                    "amplitude": 0.6,
                    "phase": 0.8,
                },
                {
                    "type": "sine",
                    "period": 48.0,
                    "frequency": 1.0 / 48.0,
                    "amplitude": 0.4,
                    "phase": 1.4,
                },
            ],
        }
    if name == "wavelet_family_mix":
        return {
            "seasonality_type": "wavelet",
            "atoms": [
                {
                    "type": "wavelet",
                    "period": 28.0,
                    "frequency": 1.0 / 28.0,
                    "amplitude": 1.0,
                    "phase": 0.4,
                    "family": "morlet",
                    "scale": 0.10,
                    "shift": 0.15,
                    "theta": {"omega": 8.5},
                },
                {
                    "type": "wavelet",
                    "period": 28.0,
                    "frequency": 1.0 / 28.0,
                    "amplitude": 0.85,
                    "phase": 0.4,
                    "family": "shan",
                    "scale": 0.18,
                    "shift": 0.62,
                    "theta": {"bandwidth": 6.5, "center": 1.2},
                },
                {
                    "type": "wavelet",
                    "period": 14.0,
                    "frequency": 1.0 / 14.0,
                    "amplitude": 0.55,
                    "phase": 1.2,
                    "family": "haar",
                    "scale": 0.22,
                    "shift": 0.32,
                    "theta": {},
                },
            ],
        }
    if name == "wavelet_scale_shift":
        return {
            "seasonality_type": "wavelet",
            "atoms": [
                {
                    "type": "wavelet",
                    "period": 36.0,
                    "frequency": 1.0 / 36.0,
                    "amplitude": 1.0,
                    "phase": 0.3,
                    "family": "mexh",
                    "scale": 0.10,
                    "shift": 0.20,
                    "theta": {},
                },
                {
                    "type": "wavelet",
                    "period": 36.0,
                    "frequency": 1.0 / 36.0,
                    "amplitude": 1.0,
                    "phase": 0.3,
                    "family": "mexh",
                    "scale": 0.26,
                    "shift": 0.56,
                    "theta": {},
                },
            ],
        }
    raise ValueError(f"Unsupported preset: {name}")


def _build_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    seasonality_type = args.seasonality_type
    if seasonality_type == "none":
        return {"seasonality_type": "none", "atoms": []}

    atoms_count = max(1, int(args.atoms))
    periods = _expand_param(_parse_float_list(args.periods), atoms_count, 24.0, "--periods")
    amplitudes = _expand_param(_parse_float_list(args.amplitudes), atoms_count, 1.0, "--amplitudes")
    phases = _expand_param(_parse_float_list(args.phases), atoms_count, 0.0, "--phases")

    atoms: list[dict[str, Any]] = []
    if seasonality_type != "wavelet":
        for i in range(atoms_count):
            period = float(periods[i])
            atom = {
                "type": seasonality_type,
                "period": period,
                "frequency": 1.0 / period,
                "amplitude": float(amplitudes[i]),
                "phase": float(phases[i]),
            }
            atoms.append(atom)
        return {"seasonality_type": seasonality_type, "atoms": atoms}

    families = _expand_param(
        _parse_str_list(args.families), atoms_count, str(args.family), "--families"
    )
    scales = _expand_param(
        _parse_float_list(args.scales), atoms_count, float(args.scale), "--scales"
    )
    shifts = _expand_param(
        _parse_float_list(args.shifts), atoms_count, float(args.shift), "--shifts"
    )
    raw_theta: list[dict[str, float]] = []
    if args.theta_jsons.strip():
        raw_theta = _parse_theta_list(args.theta_jsons)
    elif args.theta_kvs.strip():
        raw_theta = _parse_theta_kv_list(args.theta_kvs)
    theta_list = _expand_param(raw_theta, atoms_count, {}, "--theta-jsons/--theta-kvs")

    for i in range(atoms_count):
        family = str(families[i])
        if family not in WAVELET_REGISTRY:
            raise ValueError(
                f"Unsupported wavelet family: {family}; supported={sorted(WAVELET_REGISTRY)}"
            )
        period = float(periods[i])
        atom = {
            "type": "wavelet",
            "period": period,
            "frequency": 1.0 / period,
            "amplitude": float(amplitudes[i]),
            "phase": float(phases[i]),
            "family": family,
            "scale": float(scales[i]),
            "shift": float(shifts[i]),
            "theta": dict(theta_list[i]),
        }
        atoms.append(atom)

    return {"seasonality_type": "wavelet", "atoms": atoms}


def _load_params(args: argparse.Namespace) -> dict[str, Any]:
    if args.params_json:
        return json.loads(args.params_json)

    if args.params_file:
        return json.loads(Path(args.params_file).read_text(encoding="utf-8"))

    if args.sample_random:
        cfg = load_config(Path(args.config))
        rng = np.random.default_rng(args.seed)
        return sample_seasonality_params(n=args.n, config=cfg, rng=rng)

    if args.preset != "none":
        return _build_preset_params(args.preset)

    return _build_params_from_args(args)


def _print_series(values: np.ndarray, head: int | None) -> None:
    count = values.size if head is None else min(int(head), values.size)
    print("index,value")
    for i in range(count):
        print(f"{i},{values[i]:.8f}")
    if count < values.size:
        print(f"... truncated: printed {count}/{values.size} points")


def _plot_scatter(
    t: np.ndarray,
    y: np.ndarray,
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
    ax.scatter(t, y, s=point_size)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("seasonality(t)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show_window:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seasonality-only debugging tool. Generate seasonality values from custom parameters.",
    )
    parser.add_argument("--n", type=int, default=128, help="Series length")
    parser.add_argument(
        "--seed", type=int, default=42, help="Global seed (used for random sampling)"
    )
    parser.add_argument(
        "--head", type=int, default=None, help="Print first N points only; default prints all"
    )
    parser.add_argument(
        "--plot-scatter", action="store_true", help="Save scatter plot of generated seasonality"
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=str(ROOT / "outputs" / "seasonality_scatter.png"),
        help="Scatter plot output path",
    )
    parser.add_argument("--show-plot", action="store_true", help="Display plotting window")
    parser.add_argument("--point-size", type=float, default=10.0, help="Scatter point size")

    parser.add_argument(
        "--params-json", type=str, default=None, help="Full seasonality params JSON string"
    )
    parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Path to JSON file containing full seasonality params",
    )
    parser.add_argument(
        "--sample-random", action="store_true", help="Sample seasonality params from config"
    )
    parser.add_argument(
        "--config", type=str, default=str(ROOT / "configs" / "default.json"), help="Config path"
    )
    parser.add_argument(
        "--preset",
        choices=["none", "sine_single", "sine_multi", "wavelet_family_mix", "wavelet_scale_shift"],
        default="none",
        help="Quick preset (ignored when --params-json/--params-file/--sample-random is used)",
    )

    parser.add_argument(
        "--seasonality-type",
        choices=["none", "sine", "square", "triangle", "wavelet"],
        default="wavelet",
        help="Manual mode seasonality type",
    )
    parser.add_argument("--atoms", type=int, default=1, help="Manual mode number of atoms")
    parser.add_argument(
        "--periods", type=str, default="24", help="Comma-separated periods, e.g. 24,48"
    )
    parser.add_argument("--amplitudes", type=str, default="1.0", help="Comma-separated amplitudes")
    parser.add_argument(
        "--phases", type=str, default="0.0", help="Comma-separated phases in radians"
    )

    parser.add_argument(
        "--family", type=str, default="morlet", help="Default wavelet family for manual mode"
    )
    parser.add_argument("--families", type=str, default="", help="Comma-separated wavelet families")
    parser.add_argument(
        "--scale", type=float, default=0.16, help="Default wavelet scale for manual mode"
    )
    parser.add_argument("--scales", type=str, default="", help="Comma-separated wavelet scales")
    parser.add_argument(
        "--shift", type=float, default=0.0, help="Default wavelet shift for manual mode"
    )
    parser.add_argument("--shifts", type=str, default="", help="Comma-separated wavelet shifts")
    parser.add_argument(
        "--theta-jsons",
        type=str,
        default="",
        help='Semicolon-separated JSON objects per wavelet atom, e.g. \'{"omega":8};{"bandwidth":6,"center":1.2}\'',
    )
    parser.add_argument(
        "--theta-kvs",
        type=str,
        default="",
        help="Semicolon-separated key=value groups per wavelet atom, e.g. omega=9;bandwidth=6,center=1.2",
    )

    args = parser.parse_args()

    params = _load_params(args)
    t = np.arange(args.n, dtype=float)
    seasonality = render_seasonality(t=t, params=params)

    print("=== seasonality_params ===")
    print(json.dumps(params, ensure_ascii=False, indent=2))
    print("=== seasonality_summary ===")
    print(
        json.dumps(
            {
                "length": int(seasonality.size),
                "seasonality_type": str(params.get("seasonality_type", "unknown")),
                "num_atoms": int(len(params.get("atoms", [])))
                if isinstance(params.get("atoms", []), list)
                else 0,
                "min": float(np.min(seasonality)),
                "max": float(np.max(seasonality)),
                "mean": float(np.mean(seasonality)),
                "std": float(np.std(seasonality)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== seasonality_values ===")
    _print_series(values=seasonality, head=args.head)

    if args.plot_scatter:
        plot_path = Path(args.plot_out)
        _plot_scatter(
            t=t,
            y=seasonality,
            out_path=plot_path,
            title=f"Seasonality Scatter ({params.get('seasonality_type', 'unknown')})",
            show_window=bool(args.show_plot),
            point_size=float(args.point_size),
        )
        print(f"=== scatter_saved ===\n{plot_path.resolve()}")


if __name__ == "__main__":
    main()
