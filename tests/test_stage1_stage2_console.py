from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthtsad.causal.arx import ARXSystem
from synthtsad.causal.dag import CausalGraphSampler
from synthtsad.config import load_config, load_config_from_raw
from synthtsad.pipeline import SyntheticGeneratorPipeline


def _series_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _print_selected_nodes(
    name: str,
    values: np.ndarray,
    selected_nodes: list[int],
    head: int,
) -> None:
    print(f"=== {name}_values ===")
    print("index,node,value")
    count = min(int(head), values.shape[0])
    for index in range(count):
        for node in selected_nodes:
            print(f"{index},{node},{values[index, node]:.8f}")
    if count < values.shape[0]:
        print(f"... truncated: printed {count}/{values.shape[0]} timesteps")


def _build_effective_config(
    config_path: Path,
    n: int,
    num_series: int,
    seed: int,
) -> object:
    cfg = load_config(config_path)
    raw = dict(cfg.raw)
    raw["num_samples"] = 1
    raw["sequence_length"] = {"min": int(n), "max": int(n)}
    raw["num_series"] = {"min": int(num_series), "max": int(num_series)}
    raw["seed"] = int(seed)
    raw["causal"] = {
        **raw["causal"],
        "num_nodes": {"min": int(num_series), "max": int(num_series)},
    }
    return load_config_from_raw(raw)


def _plot_stage1_stage2(
    t: np.ndarray,
    stage1: np.ndarray,
    stage2: np.ndarray,
    delta: np.ndarray,
    graph_parents: list[list[int]],
    topo_order: list[int],
    selected_nodes: list[int],
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

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    _plot_dag(ax=ax00, graph_parents=graph_parents, topo_order=topo_order, selected_nodes=selected_nodes)

    for node in selected_nodes:
        ax01.plot(t, stage1[:, node], linewidth=line_width, label=f"node {node}")
    ax01.set_title("Stage1: Trend + Seasonality + Noise")
    ax01.set_xlabel("t")
    ax01.set_ylabel("value")
    ax01.grid(alpha=0.3)
    ax01.legend(loc="best")

    for node in selected_nodes:
        ax10.plot(t, stage2[:, node], linewidth=line_width, label=f"node {node}")
    ax10.set_title("Stage2: After DAG + ARX Mixing")
    ax10.set_xlabel("t")
    ax10.set_ylabel("value")
    ax10.grid(alpha=0.3)
    ax10.legend(loc="best")

    for node in selected_nodes:
        ax11.plot(t, delta[:, node], linewidth=line_width, label=f"node {node}")
    ax11.set_title("Only Stage2 Effect: Stage2 - Stage1")
    ax11.set_xlabel("t")
    ax11.set_ylabel("delta")
    ax11.grid(alpha=0.3)
    ax11.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show_window:
        plt.show()
    plt.close(fig)


def _plot_dag(
    ax,
    graph_parents: list[list[int]],
    topo_order: list[int],
    selected_nodes: list[int],
) -> None:
    from matplotlib.patches import Circle, FancyArrowPatch  # type: ignore

    num_nodes = len(graph_parents)
    depths = [0 for _ in range(num_nodes)]
    for node in topo_order:
        parents = graph_parents[node]
        if parents:
            depths[node] = max(depths[parent] + 1 for parent in parents)

    levels: dict[int, list[int]] = {}
    for node in topo_order:
        levels.setdefault(depths[node], []).append(node)

    positions: dict[int, tuple[float, float]] = {}
    max_depth = max(levels) if levels else 0
    for depth, nodes in levels.items():
        y_positions = np.linspace(0.85, 0.15, num=len(nodes)) if len(nodes) > 1 else np.array([0.5])
        x = 0.12 if max_depth == 0 else 0.12 + 0.76 * (depth / max_depth)
        for node, y in zip(nodes, y_positions):
            positions[node] = (float(x), float(y))

    for child, parents in enumerate(graph_parents):
        for parent in parents:
            x1, y1 = positions[parent]
            x2, y2 = positions[child]
            arrow = FancyArrowPatch(
                (x1 + 0.03, y1),
                (x2 - 0.03, y2),
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.5,
                color="#4a4a4a",
                alpha=0.9,
            )
            ax.add_patch(arrow)

    for node, (x, y) in positions.items():
        face = "#d95f02" if node in selected_nodes else "#1f77b4"
        circle = Circle((x, y), radius=0.04, facecolor=face, edgecolor="#222222", linewidth=1.2)
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha="center", va="center", fontsize=10, color="white", weight="bold")

    edge_count = sum(len(parents) for parents in graph_parents)
    ax.set_title(f"DAG: parent -> child (edges={edge_count})")
    ax.text(
        0.02,
        0.02,
        "orange = plotted nodes\nblue = other nodes",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        va="bottom",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual smoke test for Stage 1 baseline generation and Stage 2 causal mixing.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.json"),
        help="Config path",
    )
    parser.add_argument("--n", type=int, default=240, help="Fixed sequence length")
    parser.add_argument("--num-series", type=int, default=4, help="Fixed number of series/nodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--head", type=int, default=12, help="Print first N timesteps for selected nodes")
    parser.add_argument("--max-plot-nodes", type=int, default=4, help="Maximum number of nodes shown in line plots")
    parser.add_argument("--line-width", type=float, default=1.5, help="Plot line width")
    parser.add_argument("--plot", action="store_true", help="Save a visualization panel")
    parser.add_argument(
        "--plot-out",
        type=str,
        default=str(ROOT / "outputs" / "stage1_stage2_panel.png"),
        help="Output image path",
    )
    parser.add_argument("--show-plot", action="store_true", help="Show an interactive plot window")
    args = parser.parse_args()

    cfg = _build_effective_config(
        config_path=Path(args.config),
        n=int(args.n),
        num_series=int(args.num_series),
        seed=int(args.seed),
    )
    rng = np.random.default_rng(cfg.seed)
    pipeline = SyntheticGeneratorPipeline(cfg)

    n = int(args.n)
    d = int(args.num_series)
    t = np.arange(n, dtype=float)

    stage1_params = pipeline._sample_stage1_params(n=n, d=d, rng=rng)
    x_stage1 = pipeline._realize_stage1(t=t, stage1_params=stage1_params)

    graph = CausalGraphSampler(cfg).sample_graph(num_nodes=d, rng=rng)
    arx = ARXSystem(cfg, graph)
    stage2_params = arx.sample_params(rng)
    x_stage2, causal_state = arx.simulate_with_params(x_base=x_stage1, n_steps=n, params=stage2_params)
    delta = x_stage2 - x_stage1

    selected_nodes = list(range(min(d, max(1, int(args.max_plot_nodes)))))
    edge_count = int(np.sum(graph.adjacency))
    summary = {
        "length": n,
        "num_series": d,
        "selected_nodes": selected_nodes,
        "stage2_edge_count": edge_count,
        "topo_order": [int(v) for v in graph.topo_order],
        "stage1": _series_stats(x_stage1),
        "stage2": _series_stats(x_stage2),
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "delta_abs_max": float(np.max(np.abs(delta))),
        "latent_state": _series_stats(causal_state.z),
    }

    print("=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=== panel_guide ===")
    print(
        json.dumps(
            {
                "panel_1_dag": "真实 DAG 结构图。箭头方向是 parent -> child，表示 stage2 中谁影响谁。",
                "panel_2_stage1": "stage1 的原始基线序列，只包含 trend + seasonality + noise，还没有因果混合。",
                "panel_3_stage2": "stage2 经过 DAG + ARX 混合后的序列，是 stage1 进入因果系统后的输出。",
                "panel_4_delta": "stage2 - stage1，只保留因果混合带来的改变量；绝对值越大，说明 stage2 影响越强。",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== graph_parents ===")
    print(json.dumps(graph.parents, ensure_ascii=False, indent=2))
    print("=== stage1_param_types ===")
    print(
        json.dumps(
            [
                {
                    "node": int(spec["node"]),
                    "trend_type": str(spec["trend"]["trend_type"]),
                    "seasonality_type": str(spec["seasonality"]["seasonality_type"]),
                    "noise_level": str(spec["noise"]["noise_level"]),
                }
                for spec in stage1_params
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== stage2_params ===")
    print(json.dumps(stage2_params, ensure_ascii=False, indent=2))

    _print_selected_nodes("stage1", x_stage1, selected_nodes, int(args.head))
    _print_selected_nodes("stage2", x_stage2, selected_nodes, int(args.head))
    _print_selected_nodes("delta", delta, selected_nodes, int(args.head))

    if args.plot:
        plot_path = Path(args.plot_out)
        _plot_stage1_stage2(
            t=t,
            stage1=x_stage1,
            stage2=x_stage2,
            delta=delta,
            graph_parents=graph.parents,
            topo_order=graph.topo_order,
            selected_nodes=selected_nodes,
            out_path=plot_path,
            show_window=bool(args.show_plot),
            line_width=float(args.line_width),
        )
        print(f"=== figure_saved ===\n{plot_path.resolve()}")


if __name__ == "__main__":
    main()
