"""Run the initial Paper 2 auditability MVP.

Usage:
    uv run python -m paper2.run_mvp --steps 800 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper2.audit_game import (
    GameConfig,
    TrainConfig,
    cross_play_matrix,
    evaluate_protocol,
    join_curve,
    message_dictionary,
    probe_scores,
    strategic_opacity_gap,
    train_reinforce,
    with_monitor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--outdir", type=Path, default=Path("paper2/output/mvp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    base_config = GameConfig()
    train_config = TrainConfig(steps=args.steps, batch_size=args.batch_size)
    arms = [
        {
            "arm": "no_comm",
            "config": base_config,
            "comm_mode": "none",
            "monitor_penalty": 0.0,
            "flagged_messages": (),
        },
        {
            "arm": "free_comm",
            "config": base_config,
            "comm_mode": "learned",
            "monitor_penalty": 0.0,
            "flagged_messages": (),
        },
        {
            "arm": "naive_monitor",
            "config": with_monitor(base_config, 1.0),
            "comm_mode": "learned",
            "monitor_penalty": 0.08,
            "flagged_messages": (0,),
        },
        {
            "arm": "semantic_bottleneck",
            "config": base_config,
            "comm_mode": "semantic",
            "monitor_penalty": 0.0,
            "flagged_messages": (),
        },
    ]

    result_rows = []
    curve_rows = []
    dictionary_rows = []
    trained = {}

    for arm in arms:
        trained[arm["arm"]] = {}
        for seed in args.seeds:
            params, history = train_reinforce(
                arm["config"],
                train_config,
                seed=seed,
                comm_mode=arm["comm_mode"],
                monitor_penalty=arm["monitor_penalty"],
                flagged_messages=arm["flagged_messages"],
            )
            trained[arm["arm"]][seed] = params
            metrics = evaluate_protocol(
                params,
                arm["config"],
                seed=10_000 + seed,
                n_episodes=args.eval_episodes,
                comm_mode=arm["comm_mode"],
                monitor_penalty=arm["monitor_penalty"],
                flagged_messages=arm["flagged_messages"],
            )
            probes = probe_scores(
                params,
                arm["config"],
                seed=20_000 + seed,
                n_episodes=args.eval_episodes,
                comm_mode=arm["comm_mode"],
            )
            result_rows.append(
                {
                    "arm": arm["arm"],
                    "seed": seed,
                    "final_train_reward": history[-1]["train_reward"],
                    **metrics,
                    **probes,
                    "strategic_opacity_gap": strategic_opacity_gap(metrics, probes),
                }
            )
            for row in join_curve(
                params,
                arm["config"],
                seed=40_000 + seed,
                n_episodes=args.eval_episodes,
                comm_mode=arm["comm_mode"],
            ):
                curve_rows.append({"arm": arm["arm"], "seed": seed, **row})
            for row in message_dictionary(
                params,
                arm["config"],
                seed=50_000 + seed,
                n_episodes=args.eval_episodes,
                comm_mode=arm["comm_mode"],
            ):
                dictionary_rows.append({"arm": arm["arm"], "seed": seed, **row})

    results = pd.DataFrame(result_rows)
    curves = pd.DataFrame(curve_rows)
    dictionaries = pd.DataFrame(dictionary_rows)
    results.to_csv(args.outdir / "results.csv", index=False)
    curves.to_csv(args.outdir / "join_curves.csv", index=False)
    dictionaries.to_csv(args.outdir / "message_dictionary.csv", index=False)

    _plot_frontier(results, args.outdir / "auditability_frontier.png")
    _plot_join_curves(curves, args.outdir / "join_curves.png")

    if len(args.seeds) >= 2:
        seeds, matrix = cross_play_matrix(
            trained["free_comm"],
            base_config,
            comm_mode="learned",
            n_episodes=max(1_000, args.eval_episodes // 2),
        )
        pd.DataFrame(matrix, index=seeds, columns=seeds).to_csv(args.outdir / "cross_play_free_comm.csv")
        _plot_cross_play(seeds, matrix, args.outdir / "cross_play_free_comm.png")

    print(f"Wrote Paper 2 MVP outputs to {args.outdir}")
    print(results.groupby("arm")[["welfare", "message_value", "decodability_gap", "strategic_opacity_gap"]].mean())


def _plot_frontier(results: pd.DataFrame, path: Path) -> None:
    means = results.groupby("arm", as_index=False)[["welfare", "strategic_opacity_gap"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(means["strategic_opacity_gap"], means["welfare"], s=80)
    for _, row in means.iterrows():
        ax.annotate(row["arm"], (row["strategic_opacity_gap"], row["welfare"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Strategic opacity gap")
    ax.set_ylabel("Mean payoff")
    ax.set_title("Auditability-performance frontier")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_join_curves(curves: pd.DataFrame, path: Path) -> None:
    means = curves.groupby(["arm", "theta_mid"], as_index=False)["join_rate"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, data in means.groupby("arm"):
        ax.plot(data["theta_mid"], data["join_rate"], marker="o", label=arm)
    ax.set_xlabel("Regime strength theta")
    ax.set_ylabel("Join rate")
    ax.set_title("Global-game join curves")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_cross_play(seeds: list[int], matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(seeds)), labels=seeds)
    ax.set_yticks(range(len(seeds)), labels=seeds)
    ax.set_xlabel("Receiver seed")
    ax.set_ylabel("Sender seed")
    ax.set_title("Cross-play welfare: free communication")
    fig.colorbar(image, ax=ax, label="Mean payoff")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

