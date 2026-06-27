"""Run continuous-channel auditability-frontier experiments.

Paper 2 asks one direct question: can monitoring keep communication useful
without letting agents move payoff-relevant information into a hidden channel?

Agents send continuous messages. The auditor sees only a projection of each
message. If the hidden part of the message changes receiver behavior, monitoring
has failed operationally even if the auditor sees a clean public signal.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


SHIFT_ORDER = [
    "train_distribution",
    "cleaner_signals",
    "noisier_signals",
    "knife_edge_regimes",
    "harder_regimes",
]
SHIFT_LABELS = {
    "train_distribution": "Training\nworld",
    "cleaner_signals": "Cleaner\nsignals",
    "noisier_signals": "Noisier\nsignals",
    "knife_edge_regimes": "Knife-edge\nregimes",
    "harder_regimes": "Harder\nregimes",
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    receiver_observation: ReceiverObservation
    hidden_cost: float = 0.0
    residual_causal_cost: float = 0.0
    message_cost: float = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("paper2/output/auditability_frontier"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--updates", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=8_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-dim", type=int, default=6)
    parser.add_argument("--auditor-dim", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--fast", action="store_true", help="Tiny run for smoke tests.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from existing CSV outputs.")
    parser.add_argument("--skip-academic-checks", action="store_true", help="Skip cross-play, shift, and finite-auditor analyses.")
    parser.add_argument("--auditor-episodes", type=int, default=5_000)
    parser.add_argument("--robustness-episodes", type=int, default=4_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.seeds = args.seeds or [0]
        args.updates = min(args.updates, 5)
        args.batch_size = min(args.batch_size, 512)
        args.minibatch_size = min(args.minibatch_size, args.batch_size)
        args.ppo_epochs = min(args.ppo_epochs, 1)
        args.eval_episodes = min(args.eval_episodes, 800)
        args.auditor_episodes = min(args.auditor_episodes, 800)
        args.robustness_episodes = min(args.robustness_episodes, 800)
    else:
        args.seeds = args.seeds or [0, 1, 2]

    args.outdir.mkdir(parents=True, exist_ok=True)
    figdir = args.outdir / "figures"
    figdir.mkdir(exist_ok=True)

    configure_plot_style()
    if args.plot_only:
        results = pd.read_csv(args.outdir / "continuous_results.csv")
        curves = pd.read_csv(args.outdir / "continuous_join_curves.csv")
        robustness = read_optional_csv(args.outdir / "robustness_results.csv")
        crossplay_summary = read_optional_csv(args.outdir / "crossplay_summary.csv")
        auditor_curve = read_optional_csv(args.outdir / "auditor_sample_curve.csv")
        summary = summarize_results(results)
        summary.to_csv(args.outdir / "continuous_summary.csv")
        write_figures(results, curves, figdir, robustness=robustness, crossplay_summary=crossplay_summary, auditor_curve=auditor_curve)
        write_paper_tables(args.outdir, results, robustness=robustness, crossplay_summary=crossplay_summary)
        print(f"Regenerated figures from {args.outdir}", flush=True)
        return

    game_config = ContinuousGameConfig(
        sigma=args.sigma,
        message_dim=args.message_dim,
        auditor_dim=args.auditor_dim,
    )
    base_ppo = ContinuousPPOConfig(
        updates=args.updates,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        epochs=args.ppo_epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    selected_arms = filter_arms(default_arms(), args.arms)
    print(f"Device request={args.device}; resolved={resolve_device(args.device)}", flush=True)
    print(f"Running arms={[arm.name for arm in selected_arms]} seeds={args.seeds}", flush=True)

    result_rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | int | str]] = []
    trained: dict[str, dict[int, ContinuousPolicy]] = {}

    for arm in selected_arms:
        trained[arm.name] = {}
        ppo_config = replace_ppo_config(
            base_ppo,
            hidden_cost=arm.hidden_cost,
            residual_causal_cost=arm.residual_causal_cost,
            message_cost=arm.message_cost,
        )
        for seed in args.seeds:
            print(f"[continuous] arm={arm.name} seed={seed}", flush=True)
            policy, history = train_continuous_policy(
                game_config,
                ppo_config,
                seed=seed,
                receiver_observation=arm.receiver_observation,
            )
            trained[arm.name][seed] = policy
            metrics = evaluate_metrics(policy, seed=900_000 + seed, n_episodes=args.eval_episodes)
            result_rows.append(
                {
                    "arm": arm.name,
                    "seed": seed,
                    "receiver_observation": arm.receiver_observation,
                    "hidden_cost": arm.hidden_cost,
                    "residual_causal_cost": arm.residual_causal_cost,
                    "message_cost": arm.message_cost,
                    **metrics,
                }
            )
            for row in history:
                history_rows.append({"arm": arm.name, "seed": seed, **row})
            for row in join_curve(policy, seed=910_000 + seed, n_episodes=args.eval_episodes):
                curve_rows.append({"arm": arm.name, "seed": seed, **row})

    results = pd.DataFrame(result_rows)
    history = pd.DataFrame(history_rows)
    curves = pd.DataFrame(curve_rows)
    robustness = pd.DataFrame()
    crossplay = pd.DataFrame()
    crossplay_summary = pd.DataFrame()
    auditor_curve = pd.DataFrame()
    if not args.skip_academic_checks:
        robustness = run_robustness_checks(
            trained,
            game_config,
            eval_episodes=args.robustness_episodes,
        )
        crossplay, crossplay_summary = run_crossplay_checks(
            trained,
            eval_episodes=max(800, args.robustness_episodes // 2),
        )
        auditor_curve = run_finite_auditor_checks(
            trained,
            eval_episodes=args.auditor_episodes,
        )
    summary = summarize_results(results)

    results.to_csv(args.outdir / "continuous_results.csv", index=False)
    history.to_csv(args.outdir / "continuous_history.csv", index=False)
    curves.to_csv(args.outdir / "continuous_join_curves.csv", index=False)
    summary.to_csv(args.outdir / "continuous_summary.csv")
    if not robustness.empty:
        robustness.to_csv(args.outdir / "robustness_results.csv", index=False)
    if not crossplay.empty:
        crossplay.to_csv(args.outdir / "crossplay_results.csv", index=False)
    if not crossplay_summary.empty:
        crossplay_summary.to_csv(args.outdir / "crossplay_summary.csv", index=False)
    if not auditor_curve.empty:
        auditor_curve.to_csv(args.outdir / "auditor_sample_curve.csv", index=False)

    write_figures(results, curves, figdir, robustness=robustness, crossplay_summary=crossplay_summary, auditor_curve=auditor_curve)
    write_paper_tables(args.outdir, results, robustness=robustness, crossplay_summary=crossplay_summary)
    write_research_design_note(args.outdir)
    write_manifest(args, game_config, base_ppo, selected_arms, args.outdir)

    print(f"\nWrote continuous auditability outputs to {args.outdir}", flush=True)
    print(summary[["welfare_mean", "message_value_mean", "hidden_causal_influence_mean", "hidden_probe_gain_mean"]])


def default_arms() -> list[ArmSpec]:
    return [
        ArmSpec("no_comm", receiver_observation="none", message_cost=0.0),
        ArmSpec("visible_bottleneck", receiver_observation="visible", hidden_cost=0.04),
        ArmSpec("asymmetric_free", receiver_observation="full"),
        ArmSpec("null_penalized", receiver_observation="full", hidden_cost=0.05),
        ArmSpec("causal_regularized", receiver_observation="full", residual_causal_cost=0.60),
    ]


def filter_arms(arms: list[ArmSpec], requested: list[str] | None) -> list[ArmSpec]:
    if requested is None:
        return arms
    known = {arm.name: arm for arm in arms}
    missing = sorted(set(requested) - set(known))
    if missing:
        raise ValueError(f"unknown arms: {missing}; known arms: {sorted(known)}")
    return [known[name] for name in requested]


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "welfare",
        "message_value",
        "join_rate",
        "success_rate",
        "hidden_norm_share",
        "hidden_causal_influence",
        "hidden_probe_gain",
        "auditor_content_balanced_accuracy",
        "full_content_balanced_accuracy",
        "causal_strategic_opacity_gap",
        "probe_strategic_opacity_gap",
    ]
    summary = results.groupby("arm")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.sort_values("welfare_mean", ascending=False)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def robustness_configs(base_config: ContinuousGameConfig) -> list[tuple[str, ContinuousGameConfig]]:
    """Evaluation-only shifts for testing whether the learned protocol is brittle."""

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
                        "seed": seed,
                        "shift": shift,
                        "sigma": config.sigma,
                        "theta_low": config.theta_low,
                        "theta_high": config.theta_high,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def run_crossplay_checks(
    trained: dict[str, dict[int, ContinuousPolicy]],
    *,
    eval_episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str | bool]] = []
    for arm_index, arm in enumerate(trained):
        seeds = sorted(trained[arm])
        if not seeds:
            continue
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
                )
                metrics = evaluate_metrics(
                    paired_policy,
                    seed=1_300_000 + 10_000 * arm_index + 100 * sender_index + receiver_index,
                    n_episodes=eval_episodes,
                )
                rows.append(
                    {
                        "arm": arm,
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
    for arm, data in frame.groupby("arm"):
        self_play = data[data["self_play"]]
        cross_play = data[~data["self_play"]]
        self_welfare = float(self_play["welfare"].mean()) if not self_play.empty else float("nan")
        cross_welfare = float(cross_play["welfare"].mean()) if not cross_play.empty else float("nan")
        self_message = float(self_play["message_value"].mean()) if not self_play.empty else float("nan")
        cross_message = float(cross_play["message_value"].mean()) if not cross_play.empty else float("nan")
        summary_rows.append(
            {
                "arm": arm,
                "self_play_welfare": self_welfare,
                "cross_play_welfare": cross_welfare,
                "crossplay_welfare_gap": self_welfare - cross_welfare,
                "self_play_message_value": self_message,
                "cross_play_message_value": cross_message,
                "crossplay_message_value_gap": self_message - cross_message,
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
                rows.append({"arm": arm, "seed": seed, **row})
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


ARM_ORDER = ["no_comm", "visible_bottleneck", "null_penalized", "causal_regularized", "asymmetric_free"]
ARM_LABELS = {
    "no_comm": "No Messages",
    "visible_bottleneck": "Public Only",
    "asymmetric_free": "Private Allowed",
    "null_penalized": "Private Penalty",
    "causal_regularized": "Influence Penalty",
}


def configure_plot_style() -> None:
    px.defaults.template = "plotly_white"


def write_figures(
    results: pd.DataFrame,
    curves: pd.DataFrame,
    figdir: Path,
    *,
    robustness: pd.DataFrame | None = None,
    crossplay_summary: pd.DataFrame | None = None,
    auditor_curve: pd.DataFrame | None = None,
) -> None:
    plot_frontier(results, figdir / "continuous_frontier.png")
    plot_hidden_tradeoff(results, figdir / "hidden_channel_tradeoff.png")
    plot_join_curves(curves, figdir / "continuous_join_curves.png")
    plot_control_diagnostics(results, figdir / "control_diagnostics.png")
    if robustness is not None and not robustness.empty:
        plot_robustness_shift(robustness, figdir / "robustness_shift.png")
    if crossplay_summary is not None and not crossplay_summary.empty:
        plot_crossplay_summary(crossplay_summary, figdir / "crossplay_penalty.png")
    if auditor_curve is not None and not auditor_curve.empty:
        plot_auditor_sample_curve(auditor_curve, figdir / "auditor_sample_curve.png")
    write_figure_notes(
        results,
        figdir / "figure_notes.md",
        robustness=robustness,
        crossplay_summary=crossplay_summary,
        auditor_curve=auditor_curve,
    )


def add_arm_label(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["arm_label"] = labeled["arm"].map(lambda arm: ARM_LABELS.get(str(arm), str(arm)))
    ordered_labels = [ARM_LABELS.get(arm, arm) for arm in ARM_ORDER if arm in set(labeled["arm"])]
    ordered_labels.extend(sorted(set(labeled["arm_label"]) - set(ordered_labels)))
    labeled["arm_label"] = pd.Categorical(labeled["arm_label"], categories=ordered_labels, ordered=True)
    return labeled.sort_values("arm_label", kind="stable")


def category_labels(frame: pd.DataFrame) -> list[str]:
    return [ARM_LABELS.get(arm, arm) for arm in ordered_arms(frame)]


def save_chart(fig: go.Figure, path: Path, *, width: int = 900, height: int = 560) -> None:
    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(l=90, r=40, t=70, b=70),
        font=dict(size=14),
        legend_title_text="",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path.with_suffix(".html"), include_plotlyjs="cdn")
    fig.write_image(str(path), scale=2)
    fig.write_image(str(path.with_suffix(".svg")))


def ordered_arms(frame: pd.DataFrame) -> list[str]:
    present = set(frame["arm"])
    ordered = [arm for arm in ARM_ORDER if arm in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def aggregate_metric(results: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    grouped = results.groupby("arm")[metrics].agg(["mean", "sem"]).reset_index()
    grouped.columns = ["arm"] + [f"{metric}_{stat}" for metric, stat in grouped.columns[1:]]
    for metric in metrics:
        grouped[f"{metric}_sem"] = grouped[f"{metric}_sem"].fillna(0.0)
    order = {arm: index for index, arm in enumerate(ordered_arms(results))}
    return grouped.sort_values("arm", key=lambda values: values.map(order)).reset_index(drop=True)


def plot_frontier(results: pd.DataFrame, path: Path) -> None:
    metrics = [
        "welfare",
        "message_value",
        "hidden_causal_influence",
        "auditor_content_balanced_accuracy",
    ]
    means = aggregate_metric(results, metrics)
    means = add_arm_label(means)
    no_comm = means.loc[means["arm"] == "no_comm", "welfare_mean"]
    baseline = float(no_comm.iloc[0]) if not no_comm.empty else float(means["welfare_mean"].min())

    fig = px.scatter(
        means,
        x="hidden_causal_influence_mean",
        y="welfare_mean",
        color="arm_label",
        error_x="hidden_causal_influence_sem",
        error_y="welfare_sem",
        category_orders={"arm_label": category_labels(results)},
        title="Payoff and hidden dependence",
    )
    fig.update_traces(marker=dict(size=13), error_x=dict(thickness=1.2), error_y=dict(thickness=1.2))
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray")
    fig.add_annotation(
        xref="paper",
        x=1,
        y=baseline,
        text="No-communication payoff",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=12, color="gray"),
    )
    fig.update_xaxes(title="Hidden dependence in receiver behavior", tickformat=".0%")
    fig.update_yaxes(title="Mean payoff", tickformat=".2f")
    save_chart(fig, path)


def plot_hidden_tradeoff(results: pd.DataFrame, path: Path) -> None:
    metrics = ["message_value", "hidden_causal_influence", "hidden_probe_gain", "hidden_norm_share"]
    means = aggregate_metric(results, metrics)
    means = add_arm_label(means)
    fig = px.scatter(
        means,
        x="hidden_causal_influence_mean",
        y="message_value_mean",
        color="arm_label",
        error_x="hidden_causal_influence_sem",
        error_y="message_value_sem",
        category_orders={"arm_label": category_labels(results)},
        title="Message value and hidden dependence",
    )
    fig.update_traces(marker=dict(size=13), error_x=dict(thickness=1.2), error_y=dict(thickness=1.2))
    fig.add_hline(y=0.0, line_color="gray")
    fig.update_xaxes(title="Hidden dependence in receiver behavior", tickformat=".0%")
    fig.update_yaxes(title="Payoff gained from messages", tickformat=".2f")
    save_chart(fig, path)


def plot_join_curves(curves: pd.DataFrame, path: Path) -> None:
    if curves.empty:
        return
    plot_df = curves.copy()
    plot_df["message_group"] = np.where(plot_df["arm"] == "no_comm", "No Messages", "Messages Allowed")
    stats = plot_df.groupby(["message_group", "theta_mid"])["join_rate"].agg(["mean", "sem"]).reset_index()
    stats["sem"] = stats["sem"].fillna(0.0)
    stats["message_group"] = pd.Categorical(
        stats["message_group"],
        categories=["No Messages", "Messages Allowed"],
        ordered=True,
    )
    stats = stats.sort_values(["message_group", "theta_mid"])
    fig = px.line(
        stats,
        x="theta_mid",
        y="mean",
        color="message_group",
        error_y="sem",
        category_orders={"message_group": ["No Messages", "Messages Allowed"]},
        title="Decision rule by regime strength",
    )
    fig.update_traces(line=dict(width=2.4), error_y=dict(thickness=1.0))
    fig.update_xaxes(
        title="Regime strength",
        range=[float(curves["theta_mid"].min()) - 0.08, float(curves["theta_mid"].max()) + 0.16],
    )
    fig.update_yaxes(title="Join rate", range=[-0.03, 1.03], tickformat=".0%")
    save_chart(fig, path)


def plot_control_diagnostics(results: pd.DataFrame, path: Path) -> None:
    labels = category_labels(results)
    specs = [
        ("message_value", "Message value", ".2f"),
        ("hidden_causal_influence", "Hidden dependence", ".0%"),
        ("auditor_content_balanced_accuracy", "Auditor accuracy", ".0%"),
    ]
    means = aggregate_metric(results, [metric for metric, _, _ in specs])
    means = add_arm_label(means)
    category_array = list(reversed(labels))
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=[title for _, title, _ in specs],
    )
    for col, (metric, _title, tickformat) in enumerate(specs, start=1):
        values = means[f"{metric}_mean"].to_numpy(dtype=float)
        errors = means[f"{metric}_sem"].to_numpy(dtype=float)
        errors = np.where(np.isclose(values, 0.0), np.nan, errors)
        fig.add_trace(
            go.Bar(
                x=values,
                y=means["arm_label"],
                orientation="h",
                error_x=dict(type="data", array=errors, thickness=1.0),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        xmax = max(float(np.nanmax(values)) * 1.18, 0.02)
        if metric in {"hidden_causal_influence", "auditor_content_balanced_accuracy"}:
            xmax = 1.02
        fig.update_xaxes(range=[0, xmax], tickformat=tickformat, zeroline=False, showline=False, row=1, col=col)
        fig.update_yaxes(categoryorder="array", categoryarray=category_array, showline=False, ticks="", row=1, col=col)
    fig.add_vline(x=1 / 3, line_dash="dash", line_color="gray", row=1, col=3)
    fig.update_layout(title_text="Auditability metrics")
    save_chart(fig, path, width=1080, height=560)


def plot_robustness_shift(robustness: pd.DataFrame, path: Path) -> None:
    if robustness.empty:
        return
    shifts = [shift for shift in SHIFT_ORDER if shift in set(robustness["shift"])]
    if not shifts:
        return
    shift_categories = [SHIFT_LABELS.get(shift, shift).replace("\n", " ") for shift in shifts]
    stats = robustness.groupby(["arm", "shift"])[["welfare", "hidden_causal_influence"]].mean().reset_index()
    stats = add_arm_label(stats)
    stats["shift_label"] = stats["shift"].map(lambda shift: SHIFT_LABELS.get(str(shift), str(shift)).replace("\n", " "))
    stats["shift_label"] = pd.Categorical(stats["shift_label"], categories=shift_categories, ordered=True)
    stats = stats.sort_values(["arm_label", "shift_label"])

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.08,
        subplot_titles=["Payoff", "Hidden dependence"],
    )
    for label in category_labels(robustness):
        subset = stats[stats["arm_label"] == label]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["shift_label"],
                y=subset["welfare"],
                mode="lines+markers",
                name=str(label),
                legendgroup=str(label),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["shift_label"],
                y=subset["hidden_causal_influence"],
                mode="lines+markers",
                name=str(label),
                legendgroup=str(label),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(tickangle=20)
    fig.update_yaxes(title="Mean payoff", tickformat=".2f", row=1, col=1)
    fig.update_yaxes(title="Hidden dependence", tickformat=".0%", row=1, col=2)
    fig.update_layout(title_text="Distribution shift checks")
    save_chart(fig, path, width=1080, height=560)


def plot_crossplay_summary(crossplay_summary: pd.DataFrame, path: Path) -> None:
    if crossplay_summary.empty or "crossplay_welfare_gap" not in crossplay_summary:
        return
    data = crossplay_summary.dropna(subset=["crossplay_welfare_gap"]).copy()
    if "n_cross_pairs" in data:
        data = data[data["n_cross_pairs"] > 0]
    if data.empty:
        return
    data = add_arm_label(data)
    fig = px.bar(
        data,
        x="crossplay_welfare_gap",
        y="arm_label",
        orientation="h",
        category_orders={"arm_label": category_labels(data)},
        title="Cross-play loss",
    )
    fig.add_vline(x=0.0, line_color="gray")
    values = data["crossplay_welfare_gap"].to_numpy(dtype=float)
    xmin = float(np.nanmin(values))
    xmax = float(np.nanmax(values))
    span = max(0.025, xmax - xmin)
    fig.update_xaxes(
        title="Self-play payoff minus cross-play payoff",
        range=[min(-0.015, xmin - 0.10 * span), max(0.025, xmax + 0.18 * span)],
    )
    fig.update_yaxes(title="", categoryorder="array", categoryarray=list(reversed(category_labels(data))))
    save_chart(fig, path)


def plot_auditor_sample_curve(auditor_curve: pd.DataFrame, path: Path) -> None:
    if auditor_curve.empty:
        return
    data = auditor_curve.dropna(subset=["auditor_content_balanced_accuracy"]).copy()
    if data.empty:
        return
    data = add_arm_label(data)
    chance = float(data["chance_accuracy"].dropna().iloc[0]) if "chance_accuracy" in data and data["chance_accuracy"].notna().any() else 1 / 3
    sample_sizes = sorted(data["train_samples"].unique())

    stats = (
        data.groupby(["arm", "arm_label", "train_samples"], observed=True)["auditor_content_balanced_accuracy"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    stats["sem"] = stats["sem"].fillna(0.0)
    stats = stats.sort_values(["arm_label", "train_samples"])
    fig = px.line(
        stats,
        x="train_samples",
        y="mean",
        color="arm_label",
        error_y="sem",
        log_x=True,
        category_orders={"arm_label": category_labels(auditor_curve)},
        title="Auditor sample curve",
    )
    fig.add_hline(y=chance, line_dash="dash", line_color="gray")
    fig.add_annotation(
        xref="paper",
        x=1,
        y=chance,
        text="Chance",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=12, color="gray"),
    )
    fig.update_traces(line=dict(width=2.4), error_y=dict(thickness=1.0))
    fig.update_xaxes(
        title="Auditor training examples",
        tickmode="array",
        tickvals=sample_sizes,
        ticktext=[f"{int(sample_size):,}" for sample_size in sample_sizes],
    )
    fig.update_yaxes(
        title="Balanced accuracy reading strategic content",
        range=[max(0.0, chance - 0.08), 1.02],
        tickformat=".0%",
    )
    save_chart(fig, path)


def format_metric(value: float, *, percent: bool) -> str:
    if percent:
        return f"{value:.0%}"
    return f"{value:.2f}"


def write_figure_notes(
    results: pd.DataFrame,
    path: Path,
    *,
    robustness: pd.DataFrame | None = None,
    crossplay_summary: pd.DataFrame | None = None,
    auditor_curve: pd.DataFrame | None = None,
) -> None:
    """Write a short plain-English caption guide next to the exported figures."""

    means = aggregate_metric(
        results,
        [
            "welfare",
            "message_value",
            "hidden_causal_influence",
            "auditor_content_balanced_accuracy",
        ],
    )
    by_arm = {str(row["arm"]): row for _, row in means.iterrows()}
    free = by_arm.get("asymmetric_free")
    bottleneck = by_arm.get("visible_bottleneck")
    null_penalized = by_arm.get("null_penalized")

    lines = [
        "# Figure Notes",
        "",
        "Main point: monitoring is not enough if receivers can still use message parts the auditor cannot see.",
        "",
        "Read the figures with three quantities:",
        "",
        "- Messages help: payoff with real messages minus payoff with shuffled messages.",
        "- Hidden use: how much the receiver's join probability changes when the hidden part of the message is removed or reshuffled.",
        "- Auditor can read content: how accurately a simple auditor recovers the sender's strategic content from the monitored projection.",
        "",
    ]
    if free is not None and bottleneck is not None:
        lines.extend(
            [
                "Current demo run:",
                "",
                f"- The unconstrained channel has message value {free['message_value_mean']:.2f} and hidden use {free['hidden_causal_influence_mean']:.1%}.",
                f"- The visible bottleneck has message value {bottleneck['message_value_mean']:.2f} and hidden use {bottleneck['hidden_causal_influence_mean']:.1%}.",
                "- That is the core tradeoff: the freer channel coordinates better, but it relies more on information hidden from the auditor.",
            ]
        )
    if null_penalized is not None:
        lines.append(
            f"- The null-space penalty is the clean control test: hidden use falls to {null_penalized['hidden_causal_influence_mean']:.1%}, with message value {null_penalized['message_value_mean']:.2f}."
        )
    if crossplay_summary is not None and not crossplay_summary.empty:
        cross = crossplay_summary.dropna(subset=["crossplay_welfare_gap"])
        if "n_cross_pairs" in cross:
            cross = cross[cross["n_cross_pairs"] > 0]
        if not cross.empty:
            worst = cross.loc[cross["crossplay_welfare_gap"].idxmax()]
            best = cross.loc[cross["crossplay_welfare_gap"].idxmin()]
            lines.extend(
                [
                    "",
                    "Cross-play check:",
                    "",
                    f"- The largest self-play-to-cross-play payoff loss is {worst['crossplay_welfare_gap']:.3f} for {ARM_LABELS.get(str(worst['arm']), str(worst['arm']))}.",
                    f"- The smallest loss is {best['crossplay_welfare_gap']:.3f} for {ARM_LABELS.get(str(best['arm']), str(best['arm']))}.",
                    "- This separates a generally interpretable protocol from a private code that only works with its original partner.",
                ]
            )
    if robustness is not None and not robustness.empty:
        robust_mean = robustness.groupby(["arm", "shift"])["welfare"].mean().reset_index()
        train = robust_mean[robust_mean["shift"] == "train_distribution"][["arm", "welfare"]].rename(
            columns={"welfare": "train_welfare"}
        )
        robust_mean = robust_mean.merge(train, on="arm", how="left")
        robust_mean["welfare_drop"] = robust_mean["train_welfare"] - robust_mean["welfare"]
        shifted = robust_mean[robust_mean["shift"] != "train_distribution"].dropna(subset=["welfare_drop"])
        if not shifted.empty:
            worst_shift = shifted.loc[shifted["welfare_drop"].idxmax()]
            lines.extend(
                [
                    "",
                    "Shift check:",
                    "",
                    f"- The largest out-of-distribution payoff drop is {worst_shift['welfare_drop']:.3f} for {ARM_LABELS.get(str(worst_shift['arm']), str(worst_shift['arm']))} under {SHIFT_LABELS.get(str(worst_shift['shift']), str(worst_shift['shift'])).replace(chr(10), ' ').lower()}.",
                    "- This is the external-validity diagnostic: policies should not be judged only on the world that trained them.",
                ]
            )
    if auditor_curve is not None and not auditor_curve.empty:
        auditor = auditor_curve.dropna(subset=["auditor_content_balanced_accuracy"])
        if not auditor.empty:
            max_n = int(auditor["train_samples"].max())
            final = auditor[auditor["train_samples"] == max_n].groupby("arm")["auditor_content_balanced_accuracy"].mean()
            if not final.empty:
                top_arm = str(final.idxmax())
                lines.extend(
                    [
                        "",
                        "Finite-auditor check:",
                        "",
                        f"- With {max_n:,} labeled examples, the easiest design for the auditor is {ARM_LABELS.get(top_arm, top_arm)} at {final.max():.1%} balanced accuracy.",
                        "- This turns auditability into a sample-complexity object, not just an asymptotic probe score.",
                    ]
                )
    lines.extend(
        [
            "",
            "Paper sentence: learned communication can improve coordination, but the useful channel becomes strategically opaque unless the receiver is restricted or the reward penalizes hidden influence.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_paper_tables(
    outdir: Path,
    results: pd.DataFrame,
    *,
    robustness: pd.DataFrame | None = None,
    crossplay_summary: pd.DataFrame | None = None,
) -> None:
    table_dir = outdir / "tables"
    table_dir.mkdir(exist_ok=True)

    main = aggregate_metric(
        results,
        [
            "welfare",
            "message_value",
            "hidden_causal_influence",
            "auditor_content_balanced_accuracy",
            "probe_strategic_opacity_gap",
        ],
    )
    main_display = pd.DataFrame(
        {
            "Design": [ARM_LABELS.get(str(arm), str(arm)) for arm in main["arm"]],
            "Mean payoff": main["welfare_mean"],
            "Message value": main["message_value_mean"],
            "Hidden use": main["hidden_causal_influence_mean"],
            "Auditor accuracy": main["auditor_content_balanced_accuracy_mean"],
            "Opacity gap": main["probe_strategic_opacity_gap_mean"],
        }
    )
    if crossplay_summary is not None and not crossplay_summary.empty:
        cross = crossplay_summary[["arm", "crossplay_welfare_gap"]].copy()
        main_display = main_display.merge(
            cross.assign(Design=cross["arm"].map(lambda arm: ARM_LABELS.get(str(arm), str(arm))))[
                ["Design", "crossplay_welfare_gap"]
            ],
            on="Design",
            how="left",
        )
        main_display = main_display.rename(columns={"crossplay_welfare_gap": "Cross-play loss"})
    write_table_pair(table_dir / "main_results", main_display)

    if robustness is not None and not robustness.empty:
        robust = robustness.groupby(["arm", "shift"])[
            ["welfare", "message_value", "hidden_causal_influence"]
        ].mean().reset_index()
        train = robust[robust["shift"] == "train_distribution"][["arm", "welfare"]].rename(
            columns={"welfare": "training payoff"}
        )
        robust = robust.merge(train, on="arm", how="left")
        robust["Payoff change"] = robust["welfare"] - robust["training payoff"]
        robust["Design"] = robust["arm"].map(lambda arm: ARM_LABELS.get(str(arm), str(arm)))
        robust["Shift"] = robust["shift"].map(lambda shift: SHIFT_LABELS.get(str(shift), str(shift)).replace("\n", " "))
        robust_display = robust[
            ["Design", "Shift", "welfare", "Payoff change", "message_value", "hidden_causal_influence"]
        ].rename(
            columns={
                "welfare": "Mean payoff",
                "message_value": "Message value",
                "hidden_causal_influence": "Hidden use",
            }
        )
        robust_display["Design"] = pd.Categorical(
            robust_display["Design"],
            categories=[ARM_LABELS.get(arm, arm) for arm in ARM_ORDER],
            ordered=True,
        )
        robust_display["Shift"] = pd.Categorical(
            robust_display["Shift"],
            categories=[SHIFT_LABELS[shift].replace("\n", " ") for shift in SHIFT_ORDER],
            ordered=True,
        )
        robust_display = robust_display.sort_values(["Design", "Shift"]).reset_index(drop=True)
        write_table_pair(table_dir / "robustness_summary", robust_display)


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


def write_research_design_note(outdir: Path) -> None:
    lines = [
        "# Research Design: Controllable Strategic Agents Under Monitoring",
        "",
        "## Question",
        "",
        "Can learned agents develop useful strategic communication while remaining operationally auditable?",
        "",
        "The object is not whether an auditor can read some summary of the message. The object is whether payoff-relevant receiver behavior depends on message components outside the auditor's observation.",
        "",
        "## Core Treatment Contrast",
        "",
        "- `no_comm`: baseline coordination from private signals only.",
        "- `visible_bottleneck`: the receiver observes exactly what the auditor observes.",
        "- `asymmetric_free`: the receiver observes the full message while the auditor observes only `z = A m`.",
        "- `null_penalized`: the receiver still sees the full message, but messages are penalized for energy in the auditor-hidden null space.",
        "- `causal_regularized`: the receiver sees the full message, but training penalizes dependence on hidden message components.",
        "",
        "## Main Estimands",
        "",
        "- `message_value`: payoff with real messages minus payoff after messages are shuffled across agents.",
        "- `hidden_causal_influence`: change in receiver join probability when the hidden component of the message is removed.",
        "- `auditor_content_balanced_accuracy`: finite probe accuracy for recovering sender strategic content from the monitored projection.",
        "- `crossplay_welfare_gap`: payoff loss when sender and receiver networks come from different random seeds.",
        "",
        "## Hypotheses",
        "",
        "H1. Communication increases payoff relative to no communication.",
        "",
        "H2. Unconstrained full-channel communication creates higher message value but also higher hidden causal influence.",
        "",
        "H3. Bottlenecks and hidden-channel penalties reduce hidden causal influence, possibly at a cost to message value.",
        "",
        "H4. A genuinely auditable protocol should survive three stress tests: distribution shifts, cross-play with independently trained partners, and finite-sample auditor calibration.",
        "",
        "## Paper-Level Claim",
        "",
        "The contribution is an auditability frontier for strategic multi-agent learning. The frontier measures the tradeoff between communication value and hidden causal dependence, then asks whether control mechanisms can move policies toward high value and low hidden use.",
        "",
        "The strongest version of the paper is not only a model of hidden signaling. It is a measurement framework for evaluating whether monitored AI agents remain strategically controllable once they can learn their own communication protocol.",
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
        "runner": "paper2.run_auditability_frontier",
        "resolved_device": resolve_device(args.device),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "game_config": asdict(game_config),
        "ppo_config": asdict(ppo_config),
        "arms": [asdict(arm) for arm in arms],
        "outputs": sorted(path.name for path in outdir.iterdir()),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
