"""Run the Paper 2 experiment suite.

This script produces two kinds of evidence:

1. Protocol-validation experiments with known communication codes. These verify
   that the auditability metrics distinguish useful public communication from
   useful opaque communication.
2. Learned-agent REINFORCE baselines from ``audit_game.py``. These are the
   initial learned-agent runs; the next upgrade is PPO/IPPO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper2.audit_game import (
    AuditShaping,
    GameConfig,
    TrainConfig,
    cross_play_matrix as learned_cross_play_matrix,
    evaluate_protocol as evaluate_learned,
    join_curve as learned_join_curve,
    message_dictionary as learned_message_dictionary,
    population as learned_population,
    probe_scores,
    strategic_opacity_gap,
    train_reinforce,
    with_monitor,
)
from paper2.audit_transfer import auditor_transfer_scores, ood_strategic_opacity_gap
from paper2.ppo_agents import (
    PPOConfig,
    evaluate_ppo_metrics,
    ppo_cross_play_matrix,
    ppo_join_curve,
    ppo_message_dictionary,
    ppo_population,
    ppo_probe_scores,
    train_ippo,
)
from paper2.protocol_suite import (
    ProtocolConfig,
    cross_play_matrix,
    evaluate_protocol_arm,
    join_curve,
    message_dictionary,
    population as protocol_population,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("paper2/output/full"))
    parser.add_argument("--protocol-episodes", type=int, default=30_000)
    parser.add_argument("--crossplay-episodes", type=int, default=12_000)
    parser.add_argument("--protocol-seeds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--learned", action="store_true", help="Run learned-agent baselines too.")
    parser.add_argument("--learned-trainer", choices=["ppo", "reinforce"], default="ppo")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--learned-steps", type=int, default=1_000)
    parser.add_argument("--ppo-updates", type=int, default=220)
    parser.add_argument("--ppo-minibatch-size", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learned-batch-size", type=int, default=768)
    parser.add_argument("--learned-eval-episodes", type=int, default=4_000)
    parser.add_argument("--learned-seeds", type=int, nargs="+", default=list(range(6)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "figures").mkdir(exist_ok=True)

    protocol_config = ProtocolConfig()
    protocol_results, protocol_curves, protocol_dictionary, crossplays, protocol_transfer = run_protocol_suite(args, protocol_config)

    protocol_results.to_csv(args.outdir / "protocol_results.csv", index=False)
    protocol_curves.to_csv(args.outdir / "protocol_join_curves.csv", index=False)
    protocol_dictionary.to_csv(args.outdir / "protocol_message_dictionary.csv", index=False)
    protocol_transfer.to_csv(args.outdir / "protocol_transfer.csv", index=False)
    for name, frame in crossplays.items():
        frame.to_csv(args.outdir / f"crossplay_{name}.csv", index=False)

    plot_protocol_frontier(protocol_results, args.outdir / "figures" / "protocol_frontier.png")
    plot_join_curves(protocol_curves, args.outdir / "figures" / "protocol_join_curves.png")
    plot_protocol_sweeps(protocol_results, args.outdir / "figures")
    for name, frame in crossplays.items():
        plot_crossplay(frame, args.outdir / "figures" / f"crossplay_{name}.png", title=f"Cross-play welfare: {name}")

    learned_results = pd.DataFrame()
    if args.learned:
        if args.learned_trainer == "ppo":
            learned_results, learned_curves, learned_dictionary, learned_crossplays = run_ppo_suite(args)
        else:
            learned_results, learned_curves, learned_dictionary, learned_crossplays = run_reinforce_suite(args)
        learned_results.to_csv(args.outdir / "learned_results.csv", index=False)
        learned_curves.to_csv(args.outdir / "learned_join_curves.csv", index=False)
        learned_dictionary.to_csv(args.outdir / "learned_message_dictionary.csv", index=False)
        plot_learned_frontier(learned_results, args.outdir / "figures" / "learned_frontier.png")
        plot_join_curves(learned_curves, args.outdir / "figures" / "learned_join_curves.png", arm_col="arm")
        for name, matrix in learned_crossplays.items():
            matrix.to_csv(args.outdir / f"learned_crossplay_{name}.csv")
            plot_matrix(matrix, args.outdir / "figures" / f"learned_crossplay_{name}.png", title=f"Learned cross-play: {name}")

    write_summary(args.outdir, protocol_results, crossplays, learned_results)
    print_summary(protocol_results, learned_results, args.outdir)


def learned_arm_specs(base_config: GameConfig) -> list[dict]:
    """Learned-agent arms shared by the PPO and REINFORCE suites.

    Five arms span the auditability frontier:
      no_comm              silent control (no message channel).
      free_comm            unconstrained learned code -- the headline arm.
      audit_regularized    auditability-regularized: a bonus for emitting the
                           canonical token for its content. A *proxy* for an
                           auditor, not a learned adversary; lands in the
                           auditable region as a positive control.
      monitored_hidden     the obvious/canonical token is penalized but alternate
                           tokens stay available -- learned hidden-code stress test.
      semantic_bottleneck  message forced to equal content (maximally auditable).
    """

    shape_weight = 0.15
    return [
        {"arm": "no_comm", "config": base_config, "comm_mode": "none", "shaping": None},
        {"arm": "free_comm", "config": base_config, "comm_mode": "learned", "shaping": None},
        {
            "arm": "audit_regularized",
            "config": base_config,
            "comm_mode": "learned",
            "shaping": AuditShaping(mode="align_canonical", weight=shape_weight, canonical_tokens=(0, 1, 2)),
        },
        {
            "arm": "monitored_hidden",
            "config": base_config,
            "comm_mode": "learned",
            "shaping": AuditShaping(mode="hide_canonical", weight=shape_weight, canonical_tokens=(0, 1, 2)),
        },
        {"arm": "semantic_bottleneck", "config": base_config, "comm_mode": "semantic", "shaping": None},
    ]


# Arms whose self-play vs cross-play comparison is informative (learned codes).
LEARNED_CROSSPLAY_ARMS = ("free_comm", "audit_regularized", "monitored_hidden")


def compute_transfer_row(
    arm: str,
    populations_by_seed: dict,
    k_messages: int,
    message_value: float,
) -> dict[str, float | str]:
    """OOD auditor-transfer summary for one arm's trained populations."""

    transfer = auditor_transfer_scores(populations_by_seed, k_messages)
    return {
        "arm": arm,
        "message_value": message_value,
        **transfer,
        "ood_strategic_opacity_gap": ood_strategic_opacity_gap(message_value, transfer),
    }


def run_protocol_suite(
    args: argparse.Namespace,
    config: ProtocolConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    arms = [
        "no_comm",
        "canonical_semantic",
        "private_code",
        "monitored_hidden_code",
        "lossy_auditable",
    ]
    result_rows = []
    curve_frames = []
    dictionary_frames = []
    for arm in arms:
        for seed in args.protocol_seeds:
            print(f"[protocol] arm={arm} seed={seed}", flush=True)
            result_rows.append(
                evaluate_protocol_arm(
                    config,
                    arm=arm,
                    seed=seed,
                    n_episodes=args.protocol_episodes,
                )
            )
            curve_frames.append(join_curve(config, arm=arm, seed=seed, n_episodes=args.protocol_episodes))
            dictionary_frames.append(message_dictionary(config, arm=arm, seed=seed, n_episodes=args.protocol_episodes))

    # OOD auditor transfer: calibrate a probe on one seed's codebook and apply it
    # to the others. Idiosyncratic (seed-varying) codes resist transfer; fixed or
    # public codes do not.
    transfer_episodes = min(args.protocol_episodes, 6_000)
    transfer_rows = []
    for arm in arms:
        pops = {
            seed: protocol_population(config, arm=arm, seed=seed, n_episodes=transfer_episodes)
            for seed in args.protocol_seeds
        }
        mv = float(
            np.mean(
                [
                    evaluate_protocol_arm(config, arm=arm, seed=seed, n_episodes=transfer_episodes)["message_value"]
                    for seed in args.protocol_seeds
                ]
            )
        )
        transfer_rows.append(compute_transfer_row(arm, pops, config.k_messages, mv))
    protocol_transfer = pd.DataFrame(transfer_rows)

    crossplays = {
        "canonical_semantic": cross_play_matrix(
            config,
            arm="canonical_semantic",
            seeds=args.protocol_seeds,
            n_episodes=args.crossplay_episodes,
        ),
        "private_code": cross_play_matrix(
            config,
            arm="private_code",
            seeds=args.protocol_seeds,
            n_episodes=args.crossplay_episodes,
        ),
        "monitored_hidden_code": cross_play_matrix(
            config,
            arm="monitored_hidden_code",
            seeds=args.protocol_seeds,
            n_episodes=args.crossplay_episodes,
        ),
    }
    results = pd.DataFrame(result_rows)
    sweeps = run_protocol_sweeps(args, config)
    results = pd.concat([results, sweeps], ignore_index=True)
    return (
        results,
        pd.concat(curve_frames, ignore_index=True),
        pd.concat(dictionary_frames, ignore_index=True),
        crossplays,
        protocol_transfer,
    )


def run_protocol_sweeps(args: argparse.Namespace, base_config: ProtocolConfig) -> pd.DataFrame:
    rows = []
    sweep_seeds = args.protocol_seeds[: min(6, len(args.protocol_seeds))]
    sweep_episodes = max(3_000, args.protocol_episodes // 3)
    for sigma in [0.20, 0.35, 0.55, 0.75]:
        config = ProtocolConfig(
            n_agents=base_config.n_agents,
            theta_low=base_config.theta_low,
            theta_high=base_config.theta_high,
            sigma=sigma,
            payoff_types=base_config.payoff_types,
            content_margin=base_config.content_margin,
            message_weight=base_config.message_weight,
            k_messages=base_config.k_messages,
        )
        for arm in ["no_comm", "canonical_semantic", "private_code", "monitored_hidden_code"]:
            for seed in sweep_seeds:
                row = evaluate_protocol_arm(config, arm=arm, seed=10_000 + seed, n_episodes=sweep_episodes)
                row["sweep"] = "sigma"
                row["sweep_value"] = sigma
                rows.append(row)
    for message_weight in [0.0, 0.15, 0.28, 0.45, 0.65]:
        config = ProtocolConfig(
            n_agents=base_config.n_agents,
            theta_low=base_config.theta_low,
            theta_high=base_config.theta_high,
            sigma=base_config.sigma,
            payoff_types=base_config.payoff_types,
            content_margin=base_config.content_margin,
            message_weight=message_weight,
            k_messages=base_config.k_messages,
        )
        for arm in ["no_comm", "canonical_semantic", "private_code", "monitored_hidden_code"]:
            for seed in sweep_seeds:
                row = evaluate_protocol_arm(config, arm=arm, seed=20_000 + seed, n_episodes=sweep_episodes)
                row["sweep"] = "message_weight"
                row["sweep_value"] = message_weight
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_reinforce_suite(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    base_config = GameConfig(sigma=0.35, k_messages=4)
    train_config = TrainConfig(
        steps=args.learned_steps,
        batch_size=args.learned_batch_size,
        learning_rate=0.025,
        reward_mode="team",
    )
    arms = [
        {"arm": "no_comm", "config": base_config, "comm_mode": "none", "monitor_penalty": 0.0, "flagged_messages": ()},
        {"arm": "free_comm", "config": base_config, "comm_mode": "learned", "monitor_penalty": 0.0, "flagged_messages": ()},
        {
            "arm": "naive_monitor",
            "config": with_monitor(base_config, 1.0),
            "comm_mode": "learned",
            "monitor_penalty": 0.10,
            "flagged_messages": (0,),
        },
        {"arm": "semantic_bottleneck", "config": base_config, "comm_mode": "semantic", "monitor_penalty": 0.0, "flagged_messages": ()},
    ]
    result_rows = []
    curve_rows = []
    dictionary_rows = []
    trained = {}

    for arm in arms:
        trained[arm["arm"]] = {}
        for seed in args.learned_seeds:
            print(f"[reinforce] arm={arm['arm']} seed={seed}", flush=True)
            params, history = train_reinforce(
                arm["config"],
                train_config,
                seed=seed,
                comm_mode=arm["comm_mode"],
                monitor_penalty=arm["monitor_penalty"],
                flagged_messages=arm["flagged_messages"],
            )
            trained[arm["arm"]][seed] = params
            metrics = evaluate_learned(
                params,
                arm["config"],
                seed=200_000 + seed,
                n_episodes=args.learned_eval_episodes,
                comm_mode=arm["comm_mode"],
                monitor_penalty=arm["monitor_penalty"],
                flagged_messages=arm["flagged_messages"],
            )
            probes = probe_scores(
                params,
                arm["config"],
                seed=210_000 + seed,
                n_episodes=args.learned_eval_episodes,
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
            for row in learned_join_curve(
                params,
                arm["config"],
                seed=220_000 + seed,
                n_episodes=args.learned_eval_episodes,
                comm_mode=arm["comm_mode"],
            ):
                curve_rows.append({"arm": arm["arm"], "seed": seed, **row})
            for row in learned_message_dictionary(
                params,
                arm["config"],
                seed=230_000 + seed,
                n_episodes=args.learned_eval_episodes,
                comm_mode=arm["comm_mode"],
            ):
                dictionary_rows.append({"arm": arm["arm"], "seed": seed, **row})

    crossplays = {}
    for arm_name in ("free_comm", "naive_monitor"):
        seeds, matrix = learned_cross_play_matrix(
            trained[arm_name],
            base_config,
            comm_mode="learned",
            n_episodes=max(1_500, args.learned_eval_episodes // 2),
        )
        crossplays[arm_name] = pd.DataFrame(matrix, index=seeds, columns=seeds)

    return pd.DataFrame(result_rows), pd.DataFrame(curve_rows), pd.DataFrame(dictionary_rows), crossplays


def run_ppo_suite(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    base_config = GameConfig(sigma=0.35, k_messages=4)
    ppo_config = PPOConfig(
        updates=args.ppo_updates,
        batch_size=args.learned_batch_size,
        minibatch_size=min(args.ppo_minibatch_size, args.learned_batch_size * base_config.n_agents),
        epochs=args.ppo_epochs,
        learning_rate=3e-4,
        entropy_coef=0.025,
        hidden_dim=96,
        device=args.device,
    )
    arms = [
        {"arm": "no_comm", "config": base_config, "comm_mode": "none", "monitor_penalty": 0.0, "flagged_messages": ()},
        {"arm": "free_comm", "config": base_config, "comm_mode": "learned", "monitor_penalty": 0.0, "flagged_messages": ()},
        {
            "arm": "naive_monitor",
            "config": with_monitor(base_config, 1.0),
            "comm_mode": "learned",
            "monitor_penalty": 0.10,
            "flagged_messages": (0,),
        },
        {"arm": "semantic_bottleneck", "config": base_config, "comm_mode": "semantic", "monitor_penalty": 0.0, "flagged_messages": ()},
    ]
    result_rows = []
    curve_rows = []
    dictionary_rows = []
    trained = {}

    for arm in arms:
        trained[arm["arm"]] = {}
        for seed in args.learned_seeds:
            print(f"[ppo] arm={arm['arm']} seed={seed}", flush=True)
            policy, history = train_ippo(
                arm["config"],
                ppo_config,
                seed=seed,
                comm_mode=arm["comm_mode"],
                monitor_penalty=arm["monitor_penalty"],
                flagged_messages=arm["flagged_messages"],
            )
            trained[arm["arm"]][seed] = policy
            metrics = evaluate_ppo_metrics(
                policy,
                seed=400_000 + seed,
                n_episodes=args.learned_eval_episodes,
                comm_mode=arm["comm_mode"],
                monitor_penalty=arm["monitor_penalty"],
                flagged_messages=arm["flagged_messages"],
            )
            probes = ppo_probe_scores(
                policy,
                seed=410_000 + seed,
                n_episodes=args.learned_eval_episodes,
                comm_mode=arm["comm_mode"],
            )
            result_rows.append(
                {
                    "arm": arm["arm"],
                    "seed": seed,
                    "trainer": "ppo",
                    "final_train_reward": history[-1]["reward"],
                    **metrics,
                    **probes,
                    "strategic_opacity_gap": strategic_opacity_gap(metrics, probes),
                }
            )
            for row in ppo_join_curve(
                policy,
                seed=420_000 + seed,
                n_episodes=args.learned_eval_episodes,
                comm_mode=arm["comm_mode"],
            ):
                curve_rows.append({"arm": arm["arm"], "seed": seed, **row})
            for row in ppo_message_dictionary(
                policy,
                seed=430_000 + seed,
                n_episodes=args.learned_eval_episodes,
                comm_mode=arm["comm_mode"],
            ):
                dictionary_rows.append({"arm": arm["arm"], "seed": seed, **row})

    crossplays = {}
    for arm_name in ("free_comm", "naive_monitor"):
        seeds, matrix = ppo_cross_play_matrix(
            trained[arm_name],
            comm_mode="learned",
            n_episodes=max(1_500, args.learned_eval_episodes // 2),
        )
        crossplays[arm_name] = pd.DataFrame(matrix, index=seeds, columns=seeds)

    return pd.DataFrame(result_rows), pd.DataFrame(curve_rows), pd.DataFrame(dictionary_rows), crossplays


def plot_protocol_frontier(results: pd.DataFrame, path: Path) -> None:
    results = results[results.get("sweep", pd.Series(index=results.index, dtype=object)).isna()]
    means = results.groupby("arm", as_index=False)[["welfare", "strategic_opacity_gap", "auditor_decodability"]].mean()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    scatter = ax.scatter(
        means["strategic_opacity_gap"],
        means["welfare"],
        c=means["auditor_decodability"],
        s=90,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    label_offsets = {
        "canonical_semantic": (8, -18),
        "lossy_auditable": (8, -16),
        "monitored_hidden_code": (8, 6),
        "private_code": (-70, 8),
        "no_comm": (8, 6),
    }
    for _, row in means.iterrows():
        ax.annotate(
            row["arm"],
            (row["strategic_opacity_gap"], row["welfare"]),
            xytext=label_offsets.get(row["arm"], (6, 4)),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Strategic opacity gap")
    ax.set_ylabel("Mean payoff")
    ax.set_title("Auditability-performance frontier")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Auditor decodability")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_learned_frontier(results: pd.DataFrame, path: Path) -> None:
    means = results.groupby("arm", as_index=False)[["welfare", "strategic_opacity_gap"]].mean()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(means["strategic_opacity_gap"], means["welfare"], s=90)
    for _, row in means.iterrows():
        ax.annotate(row["arm"], (row["strategic_opacity_gap"], row["welfare"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Strategic opacity gap")
    ax.set_ylabel("Mean payoff")
    ax.set_title("Learned-agent auditability frontier")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_join_curves(curves: pd.DataFrame, path: Path, *, arm_col: str = "arm") -> None:
    means = curves.groupby([arm_col, "theta_mid"], as_index=False)["join_rate"].mean()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for arm, data in means.groupby(arm_col):
        ax.plot(data["theta_mid"], data["join_rate"], marker="o", linewidth=1.6, label=arm)
    ax.set_xlabel("Regime strength theta")
    ax.set_ylabel("Join rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Join curves by regime strength")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_protocol_sweeps(results: pd.DataFrame, figdir: Path) -> None:
    if "sweep" not in results.columns:
        return
    sweeps = results[results["sweep"].notna()]
    if sweeps.empty:
        return
    for sweep_name, data in sweeps.groupby("sweep"):
        means = data.groupby(["sweep_value", "arm"], as_index=False)[
            ["welfare", "message_value", "strategic_opacity_gap"]
        ].mean()
        for metric in ["welfare", "message_value", "strategic_opacity_gap"]:
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            for arm, arm_data in means.groupby("arm"):
                ax.plot(arm_data["sweep_value"], arm_data[metric], marker="o", label=arm)
            ax.set_xlabel(sweep_name)
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"{metric.replace('_', ' ').title()} by {sweep_name}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figdir / f"sweep_{sweep_name}_{metric}.png", dpi=220)
            plt.close(fig)


def plot_crossplay(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    matrix = frame.pivot(index="sender_seed", columns="receiver_seed", values="welfare").sort_index().sort_index(axis=1)
    plot_matrix(matrix, path, title=title)


def plot_matrix(matrix: pd.DataFrame, path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.2))
    image = ax.imshow(matrix.to_numpy(), cmap="viridis")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=list(matrix.columns))
    ax.set_yticks(np.arange(matrix.shape[0]), labels=list(matrix.index))
    ax.set_xlabel("Receiver seed")
    ax.set_ylabel("Sender seed")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Mean payoff")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(
    outdir: Path,
    protocol_results: pd.DataFrame,
    crossplays: dict[str, pd.DataFrame],
    learned_results: pd.DataFrame,
) -> None:
    primary_protocol = protocol_results[protocol_results.get("sweep", pd.Series(index=protocol_results.index, dtype=object)).isna()]
    protocol_summary = primary_protocol.groupby("arm").agg(
        welfare_mean=("welfare", "mean"),
        welfare_sd=("welfare", "std"),
        message_value_mean=("message_value", "mean"),
        auditor_decodability_mean=("auditor_decodability", "mean"),
        receiver_decodability_mean=("receiver_decodability", "mean"),
        strategic_opacity_gap_mean=("strategic_opacity_gap", "mean"),
    )
    protocol_summary.to_csv(outdir / "protocol_summary.csv")

    sweep_protocol = protocol_results[protocol_results.get("sweep", pd.Series(index=protocol_results.index, dtype=object)).notna()]
    if not sweep_protocol.empty:
        sweep_protocol.groupby(["sweep", "sweep_value", "arm"]).agg(
            welfare_mean=("welfare", "mean"),
            message_value_mean=("message_value", "mean"),
            auditor_decodability_mean=("auditor_decodability", "mean"),
            strategic_opacity_gap_mean=("strategic_opacity_gap", "mean"),
        ).to_csv(outdir / "protocol_sweep_summary.csv")

    xplay_rows = []
    for name, frame in crossplays.items():
        diag = frame[frame["sender_seed"] == frame["receiver_seed"]]["welfare"].mean()
        off = frame[frame["sender_seed"] != frame["receiver_seed"]]["welfare"].mean()
        xplay_rows.append({"arm": name, "self_play_welfare": diag, "cross_play_welfare": off, "cross_play_gap": diag - off})
    pd.DataFrame(xplay_rows).to_csv(outdir / "crossplay_summary.csv", index=False)

    if not learned_results.empty:
        learned_results.groupby("arm").agg(
            welfare_mean=("welfare", "mean"),
            message_value_mean=("message_value", "mean"),
            decodability_gap_mean=("decodability_gap", "mean"),
            strategic_opacity_gap_mean=("strategic_opacity_gap", "mean"),
        ).to_csv(outdir / "learned_summary.csv")

    manifest = {
        "protocol_rows": int(len(protocol_results)),
        "learned_rows": int(len(learned_results)),
        "outputs": sorted(p.name for p in outdir.iterdir()),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def print_summary(protocol_results: pd.DataFrame, learned_results: pd.DataFrame, outdir: Path) -> None:
    print(f"Wrote full Paper 2 experiment outputs to {outdir}")
    print("\nProtocol summary:")
    print(
        protocol_results[protocol_results.get("sweep", pd.Series(index=protocol_results.index, dtype=object)).isna()].groupby("arm")[
            ["welfare", "message_value", "auditor_decodability", "receiver_decodability", "strategic_opacity_gap"]
        ].mean()
    )
    if not learned_results.empty:
        print("\nLearned-agent summary:")
        print(learned_results.groupby("arm")[["welfare", "message_value", "decodability_gap", "strategic_opacity_gap"]].mean())


if __name__ == "__main__":
    main()
