"""Run the full Paper 2 experiment bundle.

This is the paper-facing pipeline. It trains the core auditability-frontier
suite, runs stress tests, adds a compact algorithm diagnostic, exports static
publication figures, and writes the LaTeX manuscript.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from paper2.continuous_audit_game import (
    ContinuousGameConfig,
    ContinuousPPOConfig,
    ContinuousPolicy,
    ReceiverObservation,
    evaluate_metrics,
    evaluate_policy,
    join_curve,
    replace_ppo_config,
    resolve_device,
    train_continuous_policy,
)
from paper2.paper_figures import ARM_LABELS, ARM_ORDER, SHIFT_LABELS, aggregate_metric, arm_label, write_all_figures
from paper2.paper_writer import write_manuscript


@dataclass(frozen=True)
class ArmSpec:
    name: str
    receiver_observation: ReceiverObservation
    hidden_cost: float = 0.0
    residual_causal_cost: float = 0.0
    active_monitor_cost: float = 0.0
    message_cost: float = 0.002
    monitor_full_message: bool = False
    monitor_passive: bool = False
    monitor_context: bool = False
    monitor_updates: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("paper2/output/paper2_full"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--algorithm-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--algorithm-updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--minibatch-size", type=int, default=6144)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=10_000)
    parser.add_argument("--robustness-episodes", type=int, default=5_000)
    parser.add_argument("--auditor-episodes", type=int, default=6_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-dim", type=int, default=6)
    parser.add_argument("--auditor-dim", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--monitor-updates", type=int, default=1)
    parser.add_argument("--monitor-lr", type=float, default=3e-4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-stress-tests", action="store_true")
    parser.add_argument("--skip-algorithm-diagnostic", action="store_true")
    parser.add_argument("--skip-paper", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.seeds = args.seeds or [0]
        args.algorithm_seeds = args.algorithm_seeds or [0]
        args.updates = min(args.updates, 5)
        args.algorithm_updates = min(args.algorithm_updates or args.updates, 5)
        args.batch_size = min(args.batch_size, 512)
        args.minibatch_size = min(args.minibatch_size, args.batch_size)
        args.ppo_epochs = min(args.ppo_epochs, 1)
        args.eval_episodes = min(args.eval_episodes, 800)
        args.robustness_episodes = min(args.robustness_episodes, 800)
        args.auditor_episodes = min(args.auditor_episodes, 800)
    else:
        args.seeds = args.seeds or [0, 1, 2, 3]
        args.algorithm_seeds = args.algorithm_seeds or [0, 1, 2]
        args.algorithm_updates = args.algorithm_updates or args.updates

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "tables").mkdir(exist_ok=True)

    if args.plot_only:
        outputs = read_outputs(args.outdir)
        write_all_figures(args.outdir, **outputs)
        write_tables(args.outdir, outputs["results"], outputs.get("robustness"), outputs.get("crossplay_summary"), outputs.get("algorithm_results"))
        if not args.skip_paper:
            write_manuscript(args.outdir)
        print(f"Regenerated paper outputs from {args.outdir}", flush=True)
        return

    game_config = ContinuousGameConfig(
        sigma=args.sigma,
        message_dim=args.message_dim,
        auditor_dim=args.auditor_dim,
    )
    base_config = ContinuousPPOConfig(
        updates=args.updates,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        epochs=args.ppo_epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
        learning_rule="ppo",
        monitor_updates=args.monitor_updates,
        monitor_learning_rate=args.monitor_lr,
    )

    print(f"Device request={args.device}; resolved={resolve_device(args.device)}", flush=True)
    print(f"Main suite seeds={args.seeds}; updates={args.updates}; batch={args.batch_size}", flush=True)
    arms = default_arms()
    results, history, curves, trained = run_suite(
        arms,
        game_config,
        base_config,
        seeds=args.seeds,
        eval_episodes=args.eval_episodes,
        learning_rule="ppo",
    )

    robustness = pd.DataFrame()
    crossplay = pd.DataFrame()
    crossplay_summary = pd.DataFrame()
    auditor_curve = pd.DataFrame()
    if not args.skip_stress_tests:
        robustness = run_robustness_checks(trained, game_config, eval_episodes=args.robustness_episodes)
        crossplay, crossplay_summary = run_crossplay_checks(
            trained,
            eval_episodes=max(800, args.robustness_episodes // 2),
        )
        auditor_curve = run_finite_auditor_checks(trained, eval_episodes=args.auditor_episodes)

    algorithm_results = pd.DataFrame()
    if not args.skip_algorithm_diagnostic:
        ppo_subset = results[results["arm"].isin(algorithm_arms())].copy()
        ppo_subset["learning_rule"] = "ppo"
        reinforce_config = replace(
            base_config,
            updates=args.algorithm_updates,
            epochs=1,
            learning_rule="reinforce",
        )
        print(
            f"Algorithm diagnostic seeds={args.algorithm_seeds}; updates={args.algorithm_updates}; rules=['ppo', 'reinforce']",
            flush=True,
        )
        reinforce_results, _, _, _ = run_suite(
            [arm for arm in arms if arm.name in algorithm_arms()],
            game_config,
            reinforce_config,
            seeds=args.algorithm_seeds,
            eval_episodes=args.eval_episodes,
            learning_rule="reinforce",
        )
        algorithm_results = pd.concat([ppo_subset, reinforce_results], ignore_index=True)

    write_outputs(
        args.outdir,
        results=results,
        history=history,
        curves=curves,
        robustness=robustness,
        crossplay=crossplay,
        crossplay_summary=crossplay_summary,
        auditor_curve=auditor_curve,
        algorithm_results=algorithm_results,
    )
    write_tables(args.outdir, results, robustness, crossplay_summary, algorithm_results)
    write_all_figures(
        args.outdir,
        results=results,
        curves=curves,
        robustness=robustness,
        crossplay_summary=crossplay_summary,
        auditor_curve=auditor_curve,
        algorithm_results=algorithm_results,
    )
    write_research_design(args.outdir)
    write_manifest(args, game_config, base_config, arms, args.outdir)
    if not args.skip_paper:
        tex_path = write_manuscript(args.outdir)
        print(f"Wrote manuscript source to {tex_path}", flush=True)

    summary = aggregate_metric(
        results,
        [
            "welfare",
            "message_value",
            "hidden_causal_influence",
            "auditor_content_balanced_accuracy",
            "live_monitor_content_accuracy",
        ],
    )
    print(f"\nWrote Paper 2 bundle to {args.outdir}", flush=True)
    print(
        summary[
            [
                "arm",
                "welfare_mean",
                "message_value_mean",
                "hidden_causal_influence_mean",
                "auditor_content_balanced_accuracy_mean",
                "live_monitor_content_accuracy_mean",
            ]
        ]
    )


def default_arms() -> list[ArmSpec]:
    return [
        ArmSpec("no_comm", receiver_observation="none", message_cost=0.0),
        ArmSpec("visible_bottleneck", receiver_observation="visible", hidden_cost=0.04),
        ArmSpec("asymmetric_free", receiver_observation="full"),
        ArmSpec("public_monitored", receiver_observation="visible", hidden_cost=0.04, active_monitor_cost=0.12),
        ArmSpec("private_monitored", receiver_observation="full", active_monitor_cost=0.12),
        ArmSpec("monitored_penalty", receiver_observation="full", active_monitor_cost=0.12, residual_causal_cost=0.60),
    ]


def algorithm_arms() -> set[str]:
    return {"visible_bottleneck", "asymmetric_free", "public_monitored", "private_monitored", "monitored_penalty"}


def run_suite(
    arms: list[ArmSpec],
    game_config: ContinuousGameConfig,
    base_config: ContinuousPPOConfig,
    *,
    seeds: list[int],
    eval_episodes: int,
    learning_rule: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[int, ContinuousPolicy]]]:
    result_rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | int | str]] = []
    trained: dict[str, dict[int, ContinuousPolicy]] = {}

    for arm in arms:
        trained[arm.name] = {}
        config = replace_ppo_config(
            base_config,
            hidden_cost=arm.hidden_cost,
            residual_causal_cost=arm.residual_causal_cost,
            active_monitor_cost=arm.active_monitor_cost,
            message_cost=arm.message_cost,
            monitor_full_message=arm.monitor_full_message,
            monitor_passive=arm.monitor_passive,
            monitor_context=arm.monitor_context,
            **({"monitor_updates": arm.monitor_updates} if arm.monitor_updates is not None else {}),
        )
        for seed in seeds:
            print(f"[{learning_rule}] arm={arm_label(arm.name)} seed={seed}", flush=True)
            policy, history = train_continuous_policy(
                game_config,
                config,
                seed=seed,
                receiver_observation=arm.receiver_observation,
            )
            trained[arm.name][seed] = policy
            metrics = evaluate_metrics(policy, seed=900_000 + 10_000 * len(result_rows) + seed, n_episodes=eval_episodes)
            result_rows.append(
                {
                    "arm": arm.name,
                    "arm_label": arm_label(arm.name),
                    "seed": seed,
                    "learning_rule": learning_rule,
                    "receiver_observation": arm.receiver_observation,
                    "hidden_cost": arm.hidden_cost,
                    "residual_causal_cost": arm.residual_causal_cost,
                    "active_monitor_cost": arm.active_monitor_cost,
                    "message_cost": arm.message_cost,
                    **metrics,
                }
            )
            for row in history:
                history_rows.append({"arm": arm.name, "arm_label": arm_label(arm.name), "seed": seed, "learning_rule": learning_rule, **row})
            for row in join_curve(policy, seed=910_000 + 10_000 * len(result_rows) + seed, n_episodes=eval_episodes):
                curve_rows.append({"arm": arm.name, "arm_label": arm_label(arm.name), "seed": seed, "learning_rule": learning_rule, **row})

    return pd.DataFrame(result_rows), pd.DataFrame(history_rows), pd.DataFrame(curve_rows), trained


def run_robustness_checks(
    trained: dict[str, dict[int, ContinuousPolicy]],
    base_config: ContinuousGameConfig,
    *,
    eval_episodes: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    shifts = robustness_configs(base_config)
    for arm_index, arm in enumerate(trained):
        for seed_index, seed in enumerate(sorted(trained[arm])):
            policy = trained[arm][seed]
            for shift_index, (shift, config) in enumerate(shifts):
                metrics = evaluate_metrics(
                    policy,
                    seed=1_100_000 + 10_000 * arm_index + 100 * seed_index + shift_index,
                    n_episodes=eval_episodes,
                    config_override=config,
                )
                rows.append(
                    {
                        "arm": arm,
                        "arm_label": arm_label(arm),
                        "seed": seed,
                        "shift": shift,
                        "shift_label": SHIFT_LABELS.get(shift, shift),
                        "sigma": config.sigma,
                        "theta_low": config.theta_low,
                        "theta_high": config.theta_high,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def robustness_configs(base_config: ContinuousGameConfig) -> list[tuple[str, ContinuousGameConfig]]:
    return [
        ("train_distribution", base_config),
        ("cleaner_signals", replace(base_config, sigma=max(0.08, base_config.sigma * 0.60))),
        ("noisier_signals", replace(base_config, sigma=min(0.90, base_config.sigma * 1.60))),
        (
            "knife_edge_regimes",
            replace(
                base_config,
                theta_low=max(0.05, base_config.theta_low + 0.45),
                theta_high=min(1.25, base_config.theta_high - 0.45),
            ),
        ),
        (
            "harder_regimes",
            replace(
                base_config,
                theta_low=base_config.theta_low + 0.25,
                theta_high=base_config.theta_high + 0.15,
            ),
        ),
    ]


def run_crossplay_checks(
    trained: dict[str, dict[int, ContinuousPolicy]],
    *,
    eval_episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str | bool]] = []
    for arm_index, arm in enumerate(trained):
        seeds = sorted(trained[arm])
        for sender_index, sender_seed in enumerate(seeds):
            for receiver_index, receiver_seed in enumerate(seeds):
                sender_policy = trained[arm][sender_seed]
                receiver_policy = trained[arm][receiver_seed]
                paired_policy = ContinuousPolicy(
                    sender=sender_policy.sender,
                    receiver=receiver_policy.receiver,
                    config=sender_policy.config,
                    projection=sender_policy.projection,
                    receiver_observation=sender_policy.receiver_observation,
                    device=sender_policy.device,
                    monitor=sender_policy.monitor,
                )
                metrics = evaluate_metrics(
                    paired_policy,
                    seed=1_300_000 + 10_000 * arm_index + 100 * sender_index + receiver_index,
                    n_episodes=eval_episodes,
                )
                rows.append(
                    {
                        "arm": arm,
                        "arm_label": arm_label(arm),
                        "sender_seed": sender_seed,
                        "receiver_seed": receiver_seed,
                        "self_play": sender_seed == receiver_seed,
                        **metrics,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, pd.DataFrame()

    summary_rows: list[dict[str, float | int | str]] = []
    for arm, data in frame.groupby("arm", sort=False):
        self_play = data[data["self_play"]]
        cross_play = data[~data["self_play"]]
        self_welfare = float(self_play["welfare"].mean()) if not self_play.empty else float("nan")
        cross_welfare = float(cross_play["welfare"].mean()) if not cross_play.empty else float("nan")
        summary_rows.append(
            {
                "arm": arm,
                "arm_label": arm_label(str(arm)),
                "self_play_welfare": self_welfare,
                "cross_play_welfare": cross_welfare,
                "crossplay_welfare_gap": self_welfare - cross_welfare,
                "self_play_message_value": float(self_play["message_value"].mean()) if not self_play.empty else float("nan"),
                "cross_play_message_value": float(cross_play["message_value"].mean()) if not cross_play.empty else float("nan"),
                "n_self_pairs": int(len(self_play)),
                "n_cross_pairs": int(len(cross_play)),
            }
        )
    return frame, pd.DataFrame(summary_rows)


def run_finite_auditor_checks(
    trained: dict[str, dict[int, ContinuousPolicy]],
    *,
    eval_episodes: int,
) -> pd.DataFrame:
    sample_sizes = [25, 50, 100, 250, 500, 1_000, 2_000, 5_000]
    rows: list[dict[str, float | int | str]] = []
    for arm_index, arm in enumerate(trained):
        for seed_index, seed in enumerate(sorted(trained[arm])):
            policy = trained[arm][seed]
            batch = evaluate_policy(
                policy,
                seed=1_500_000 + 10_000 * arm_index + seed_index,
                n_episodes=eval_episodes,
            )
            x = batch.auditor_observation.reshape(-1, batch.auditor_observation.shape[-1]).detach().cpu().numpy()
            y = batch.content.reshape(-1).detach().cpu().numpy()
            for row in finite_probe_curve(x, y, sample_sizes=sample_sizes, seed=1_700_000 + 1_000 * arm_index + seed):
                rows.append({"arm": arm, "arm_label": arm_label(arm), "seed": seed, **row})
    return pd.DataFrame(rows)


def finite_probe_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sample_sizes: list[int],
    seed: int,
) -> list[dict[str, float | int]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return []
    stratify = y if counts.min() >= 2 else None
    x_pool, x_test, y_pool, y_test = train_test_split(
        x,
        y,
        test_size=0.40,
        random_state=seed,
        stratify=stratify,
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    used_sizes: set[int] = set()
    for requested_size in sample_sizes:
        train_size = min(int(requested_size), len(y_pool))
        if train_size in used_sizes or train_size < max(2, classes.size):
            continue
        used_sizes.add(train_size)

        if train_size == len(y_pool):
            x_train = x_pool
            y_train = y_pool
        else:
            _, pool_counts = np.unique(y_pool, return_counts=True)
            train_stratify = y_pool if pool_counts.min() >= 2 and train_size >= classes.size * 2 else None
            try:
                x_train, _, y_train, _ = train_test_split(
                    x_pool,
                    y_pool,
                    train_size=train_size,
                    random_state=seed + train_size,
                    stratify=train_stratify,
                )
            except ValueError:
                chosen = rng.choice(len(y_pool), size=train_size, replace=False)
                x_train = x_pool[chosen]
                y_train = y_pool[chosen]

        if np.unique(y_train).size < 2:
            score = float("nan")
        else:
            model = LogisticRegression(max_iter=1_000)
            model.fit(x_train, y_train)
            score = float(balanced_accuracy_score(y_test, model.predict(x_test)))
        rows.append(
            {
                "train_samples": train_size,
                "auditor_content_balanced_accuracy": score,
                "n_test": int(len(y_test)),
                "chance_accuracy": float(1.0 / classes.size),
            }
        )
    return rows


def write_outputs(
    outdir: Path,
    *,
    results: pd.DataFrame,
    history: pd.DataFrame,
    curves: pd.DataFrame,
    robustness: pd.DataFrame,
    crossplay: pd.DataFrame,
    crossplay_summary: pd.DataFrame,
    auditor_curve: pd.DataFrame,
    algorithm_results: pd.DataFrame,
) -> None:
    results.to_csv(outdir / "continuous_results.csv", index=False)
    history.to_csv(outdir / "continuous_history.csv", index=False)
    curves.to_csv(outdir / "continuous_join_curves.csv", index=False)
    aggregate_metric(
        results,
        [
            "welfare",
            "message_value",
            "hidden_causal_influence",
            "auditor_content_balanced_accuracy",
            "live_monitor_content_accuracy",
            "live_monitor_content_balanced_accuracy",
        ],
    ).to_csv(outdir / "continuous_summary.csv", index=False)
    if not robustness.empty:
        robustness.to_csv(outdir / "robustness_results.csv", index=False)
    if not crossplay.empty:
        crossplay.to_csv(outdir / "crossplay_results.csv", index=False)
    if not crossplay_summary.empty:
        crossplay_summary.to_csv(outdir / "crossplay_summary.csv", index=False)
    if not auditor_curve.empty:
        auditor_curve.to_csv(outdir / "auditor_sample_curve.csv", index=False)
    if not algorithm_results.empty:
        algorithm_results.to_csv(outdir / "algorithm_results.csv", index=False)


def read_outputs(outdir: Path) -> dict[str, pd.DataFrame]:
    return {
        "results": pd.read_csv(outdir / "continuous_results.csv"),
        "curves": read_optional(outdir / "continuous_join_curves.csv"),
        "robustness": read_optional(outdir / "robustness_results.csv"),
        "crossplay_summary": read_optional(outdir / "crossplay_summary.csv"),
        "auditor_curve": read_optional(outdir / "auditor_sample_curve.csv"),
        "algorithm_results": read_optional(outdir / "algorithm_results.csv"),
    }


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_tables(
    outdir: Path,
    results: pd.DataFrame,
    robustness: pd.DataFrame | None,
    crossplay_summary: pd.DataFrame | None,
    algorithm_results: pd.DataFrame | None,
) -> None:
    table_dir = outdir / "tables"
    table_dir.mkdir(exist_ok=True)
    summary = aggregate_metric(
        results,
        [
            "welfare",
            "message_value",
            "hidden_causal_influence",
            "auditor_content_balanced_accuracy",
            "live_monitor_content_accuracy",
            "probe_strategic_opacity_gap",
        ],
    )
    display = pd.DataFrame(
        {
            "Design": summary["arm"].map(arm_label),
            "Payoff": summary["welfare_mean"],
            "Message value": summary["message_value_mean"],
            "Hidden use": summary["hidden_causal_influence_mean"],
            "Offline probe": summary["auditor_content_balanced_accuracy_mean"],
            "Live monitor": summary["live_monitor_content_accuracy_mean"],
            "Opacity gap": summary["probe_strategic_opacity_gap_mean"],
        }
    )
    if crossplay_summary is not None and not crossplay_summary.empty:
        display = display.merge(
            crossplay_summary.assign(Design=crossplay_summary["arm"].map(arm_label))[["Design", "crossplay_welfare_gap"]],
            on="Design",
            how="left",
        ).rename(columns={"crossplay_welfare_gap": "Cross-play loss"})
    write_table_pair(table_dir / "main_results", display)

    if robustness is not None and not robustness.empty:
        robust = robustness.groupby(["arm", "shift"], sort=False)[["welfare", "message_value", "hidden_causal_influence"]].mean().reset_index()
        train = robust[robust["shift"] == "train_distribution"][["arm", "welfare"]].rename(columns={"welfare": "training_payoff"})
        robust = robust.merge(train, on="arm", how="left")
        robust["Payoff change"] = robust["welfare"] - robust["training_payoff"]
        robust["Design"] = robust["arm"].map(arm_label)
        robust["Shift"] = robust["shift"].map(lambda shift: SHIFT_LABELS.get(str(shift), str(shift)))
        robust_display = robust[["Design", "Shift", "welfare", "Payoff change", "message_value", "hidden_causal_influence"]].rename(
            columns={
                "welfare": "Payoff",
                "message_value": "Message value",
                "hidden_causal_influence": "Hidden use",
            }
        )
        write_table_pair(table_dir / "robustness_summary", robust_display)

    if algorithm_results is not None and not algorithm_results.empty:
        algo = (
            algorithm_results.groupby(["learning_rule", "arm"], sort=False)[
                [
                    "welfare",
                    "message_value",
                    "hidden_causal_influence",
                    "auditor_content_balanced_accuracy",
                    "live_monitor_content_accuracy",
                ]
            ]
            .mean()
            .reset_index()
        )
        algo["Learning rule"] = algo["learning_rule"].map({"ppo": "PPO", "reinforce": "REINFORCE"}).fillna(algo["learning_rule"])
        algo["Design"] = algo["arm"].map(arm_label)
        algo_display = algo[
            [
                "Learning rule",
                "Design",
                "welfare",
                "message_value",
                "hidden_causal_influence",
                "auditor_content_balanced_accuracy",
                "live_monitor_content_accuracy",
            ]
        ].rename(
            columns={
                "welfare": "Payoff",
                "message_value": "Message value",
                "hidden_causal_influence": "Hidden use",
                "auditor_content_balanced_accuracy": "Offline probe",
                "live_monitor_content_accuracy": "Live monitor",
            }
        )
        write_table_pair(table_dir / "algorithm_diagnostic", algo_display)


def write_table_pair(path_base: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path_base.with_suffix(".csv"), index=False)
    path_base.with_suffix(".md").write_text(markdown_table(frame) + "\n")
    path_base.with_suffix(".tex").write_text(frame.to_latex(index=False, escape=True, float_format="%.3f"))


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [markdown_value(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def write_research_design(outdir: Path) -> None:
    lines = [
        "# Paper 2 Research Design",
        "",
        "Question: can trained strategic agents communicate useful private information while remaining auditable when monitoring is itself part of the incentive environment?",
        "",
        "Main estimands:",
        "",
        "- Message value: payoff with real messages minus payoff after messages are shuffled.",
        "- Hidden causal influence: change in receiver join probability when monitor-hidden message components are removed.",
        "- Offline auditor readability: finite-sample balanced accuracy for recovering strategic content from the monitored projection.",
        "- Live monitor readability: accuracy of the adversarial monitor that trains against the agents in active-monitor treatments.",
        "- Cross-play loss: self-play payoff minus payoff when senders and receivers come from different seeds.",
        "",
        "The paper's identification logic is the comparison between Public Monitored and Private Monitored. Both arms face the same live monitor penalty, but only Private Monitored gives the receiver access to the monitor-hidden null space. If hidden use rises and live monitor accuracy falls in Private Monitored, concealment is load-bearing rather than a passive measurement artifact.",
        "",
        "Public Only and Private Allowed are passive baselines. Monitored + Penalty asks whether reward design can preserve message value while pushing hidden causal influence back down.",
        "",
        "The algorithm diagnostic compares PPO against a REINFORCE-style policy gradient. It is a robustness diagnostic for the training procedure, not the main theoretical claim. The next algorithmic extension is an off-policy actor-critic baseline such as SAC/DDPG adapted to the continuous sender channel.",
        "",
    ]
    (outdir / "research_design.md").write_text("\n".join(lines))


def write_manifest(
    args: argparse.Namespace,
    game_config: ContinuousGameConfig,
    ppo_config: ContinuousPPOConfig,
    arms: list[ArmSpec],
    outdir: Path,
) -> None:
    manifest = {
        "runner": "paper2.run_paper2",
        "resolved_device": resolve_device(args.device),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "game_config": asdict(game_config),
        "ppo_config": asdict(ppo_config),
        "arms": [asdict(arm) for arm in arms],
        "arm_labels": ARM_LABELS,
        "arm_order": ARM_ORDER,
        "outputs": sorted(path.name for path in outdir.iterdir()),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
