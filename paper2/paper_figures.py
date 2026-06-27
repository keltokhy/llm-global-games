"""Publication figures for Paper 2.

These plots are intentionally static and low-ink.  The experiment CSVs are the
source of truth; this module only aggregates them and exports paper-ready
figures as PDF, PNG, and SVG.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRAL = {
    "open": TOKENS["panel"],
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

BLUE = {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
GOLD = {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
OLIVE = {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"}
PINK = {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"}

ARM_ORDER = [
    "no_comm",
    "visible_bottleneck",
    "asymmetric_free",
    "public_monitored",
    "private_monitored",
    "monitored_penalty",
]
ARM_LABELS = {
    "no_comm": "No Messages",
    "visible_bottleneck": "Public Only",
    "asymmetric_free": "Private Allowed",
    "public_monitored": "Public Monitored",
    "private_monitored": "Private Monitored",
    "monitored_penalty": "Monitored + Penalty",
}
ARM_COLORS = {
    "no_comm": NEUTRAL["mid"],
    "visible_bottleneck": BLUE["mid"],
    "asymmetric_free": ORANGE["mid"],
    "public_monitored": OLIVE["mid"],
    "private_monitored": PINK["mid"],
    "monitored_penalty": GOLD["dark"],
}

SHIFT_ORDER = [
    "train_distribution",
    "cleaner_signals",
    "noisier_signals",
    "knife_edge_regimes",
    "harder_regimes",
]
SHIFT_LABELS = {
    "train_distribution": "Training world",
    "cleaner_signals": "Cleaner signals",
    "noisier_signals": "Noisier signals",
    "knife_edge_regimes": "Knife-edge regimes",
    "harder_regimes": "Harder regimes",
}


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.7,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "axes.titleweight": "semibold",
        },
    )


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    title = textwrap.fill(str(title).strip(), width=78, break_long_words=False)
    subtitle = textwrap.fill(str(subtitle).strip(), width=112, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(top=max(0.56, 0.78 - 0.040 * (title_lines - 1) - 0.034 * (subtitle_lines - 1)))
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
        linespacing=1.08,
    )
    fig.text(
        left,
        0.928 - 0.040 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9.2,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    sns.despine(ax=ax)


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(path_base.with_suffix(suffix), bbox_inches="tight", dpi=240)
    plt.close(fig)


def write_all_figures(
    outdir: Path,
    *,
    results: pd.DataFrame,
    curves: pd.DataFrame,
    robustness: pd.DataFrame | None = None,
    crossplay_summary: pd.DataFrame | None = None,
    auditor_curve: pd.DataFrame | None = None,
    algorithm_results: pd.DataFrame | None = None,
) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    use_chart_theme()
    plot_auditability_frontier(results, figdir / "fig1_auditability_frontier")
    plot_metric_triptych(results, figdir / "fig2_control_metrics")
    if crossplay_summary is not None and not crossplay_summary.empty:
        plot_crossplay_loss(crossplay_summary, figdir / "fig3_crossplay_loss")
    if robustness is not None and not robustness.empty:
        plot_robustness_loss(robustness, figdir / "fig4_distribution_shift")
    if auditor_curve is not None and not auditor_curve.empty:
        plot_auditor_sample_curve(auditor_curve, figdir / "fig5_auditor_sample_curve")
    if not curves.empty:
        plot_decision_rule(curves, figdir / "fig6_decision_rule")
    if algorithm_results is not None and not algorithm_results.empty:
        plot_algorithm_diagnostic(algorithm_results, figdir / "fig7_algorithm_diagnostic")
    write_figure_notes(outdir, results, robustness, crossplay_summary, auditor_curve, algorithm_results)


def aggregate_metric(results: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    grouped = results.groupby("arm", sort=False)[metrics].agg(["mean", "sem"]).reset_index()
    grouped.columns = ["arm"] + [f"{metric}_{stat}" for metric, stat in grouped.columns[1:]]
    for metric in metrics:
        sem_col = f"{metric}_sem"
        grouped[sem_col] = grouped[sem_col].fillna(0.0)
    order = {arm: index for index, arm in enumerate(ordered_arms(results))}
    return grouped.sort_values("arm", key=lambda values: values.map(order)).reset_index(drop=True)


def ordered_arms(frame: pd.DataFrame) -> list[str]:
    present = set(frame["arm"].astype(str))
    ordered = [arm for arm in ARM_ORDER if arm in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def arm_label(arm: str) -> str:
    return ARM_LABELS.get(str(arm), str(arm).replace("_", " ").title())


def arm_color(arm: str) -> str:
    return ARM_COLORS.get(str(arm), NEUTRAL["dark"])


def percent_axis(ax: plt.Axes, axis: str = "x", *, decimals: int = 0) -> None:
    formatter = mticker.PercentFormatter(1.0, decimals=decimals)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def label_point(ax: plt.Axes, x: float, y: float, label: str, *, dx: float = 0.006, dy: float = 0.0) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        fontsize=8.8,
        color=TOKENS["ink"],
        va="center",
    )


def plot_auditability_frontier(results: pd.DataFrame, path_base: Path) -> None:
    metrics = ["message_value", "hidden_causal_influence", "welfare"]
    data = aggregate_metric(results, metrics)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y_span = max(float(data["message_value_mean"].max() - data["message_value_mean"].min()), 0.04)
    x_span = max(float(data["hidden_causal_influence_mean"].max() - data["hidden_causal_influence_mean"].min()), 0.04)

    for _, row in data.iterrows():
        arm = str(row["arm"])
        x = float(row["hidden_causal_influence_mean"])
        y = float(row["message_value_mean"])
        ax.errorbar(
            x,
            y,
            xerr=float(row["hidden_causal_influence_sem"]),
            yerr=float(row["message_value_sem"]),
            fmt="o",
            markersize=7.5,
            markerfacecolor=arm_color(arm),
            markeredgecolor=TOKENS["ink"],
            markeredgewidth=0.7,
            ecolor=NEUTRAL["base"],
            elinewidth=0.9,
            capsize=2,
            zorder=3,
        )
        default_dx = 0.055 * x_span
        dx, dy = {
            "no_comm": (default_dx, -0.030 * y_span),
            "visible_bottleneck": (default_dx * 1.20, 0.185 * y_span),
            "asymmetric_free": (default_dx, 0.000 * y_span),
            "public_monitored": (default_dx * 1.20, -0.030 * y_span),
            "private_monitored": (default_dx, -0.035 * y_span),
            "monitored_penalty": (default_dx, -0.165 * y_span),
        }.get(arm, (default_dx, 0.0))
        label_point(ax, x, y, arm_label(arm), dx=dx, dy=dy)

    ax.axvline(0, color=NEUTRAL["base"], linewidth=0.9)
    ax.axhline(0, color=NEUTRAL["base"], linewidth=0.9)
    ax.set_xlabel("Hidden causal influence")
    ax.set_ylabel("Message value")
    percent_axis(ax, "x", decimals=1 if float(data["hidden_causal_influence_mean"].max()) < 0.10 else 0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    _pad_limits(ax, data["hidden_causal_influence_mean"], data["message_value_mean"], x_min=0.0)
    add_chart_header(
        fig,
        ax,
        "The auditability frontier",
        "Message value is the payoff gain from real messages over shuffled messages. Hidden causal influence is the mean change in join probability when the monitor-hidden channel is removed. Dots are seed means with s.e.m. whiskers.",
    )
    save_figure(fig, path_base)


def plot_metric_triptych(results: pd.DataFrame, path_base: Path) -> None:
    metrics = [
        ("message_value", "Message value", False),
        ("hidden_causal_influence", "Hidden use", True),
        ("auditor_content_balanced_accuracy", "Offline probe", True),
        ("live_monitor_content_accuracy", "Live monitor", True),
    ]
    data = aggregate_metric(results, [metric for metric, _, _ in metrics])
    arms = list(reversed(ordered_arms(results)))
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.8), sharey=True)
    positions = np.arange(len(arms))
    arm_to_y = {arm: i for i, arm in enumerate(arms)}

    for ax, (metric, label, is_percent) in zip(axes, metrics):
        ax.axvline(0, color=NEUTRAL["base"], linewidth=0.8)
        if metric == "auditor_content_balanced_accuracy":
            ax.axvline(1 / 3, color=NEUTRAL["mid"], linewidth=0.9, linestyle=":")
        for _, row in data.iterrows():
            arm = str(row["arm"])
            y = arm_to_y[arm]
            value = float(row[f"{metric}_mean"])
            if not np.isfinite(value):
                continue
            sem = float(row[f"{metric}_sem"])
            ax.errorbar(
                value,
                y,
                xerr=sem,
                fmt="o",
                markersize=6.8,
                color=arm_color(arm),
                markeredgecolor=TOKENS["ink"],
                markeredgewidth=0.6,
                ecolor=NEUTRAL["base"],
                elinewidth=0.9,
                capsize=2,
                zorder=3,
            )
        ax.set_title(label, fontsize=10, color=TOKENS["ink"], pad=8)
        ax.set_xlabel("")
        ax.grid(axis="y", visible=False)
        ax.set_yticks(positions)
        ax.set_yticklabels([arm_label(arm) for arm in arms], fontsize=9)
        if is_percent:
            if metric == "hidden_causal_influence":
                percent_axis(ax, "x", decimals=1)
                xmax = max(0.05, float(data[f"{metric}_mean"].max()) * 1.35)
                ax.set_xlim(0, xmax)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            else:
                percent_axis(ax, "x")
                ax.set_xlim(0, 1.02)
        else:
            xmin = min(0.0, float(data[f"{metric}_mean"].min()) - 0.02)
            xmax = float(data[f"{metric}_mean"].max()) + 0.04
            ax.set_xlim(xmin, max(0.05, xmax))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        sns.despine(ax=ax)

    add_chart_header(
        fig,
        axes[0],
        "Control changes what communication means",
        "The same strategic game is trained under passive and active monitoring designs. Good designs keep message value high and hidden use low without letting live monitoring collapse.",
    )
    axes[0].set_title("Message value", fontsize=10, color=TOKENS["ink"], pad=8)
    fig.subplots_adjust(wspace=0.28)
    save_figure(fig, path_base)


def plot_crossplay_loss(crossplay_summary: pd.DataFrame, path_base: Path) -> None:
    data = crossplay_summary.dropna(subset=["crossplay_welfare_gap"]).copy()
    if "n_cross_pairs" in data:
        data = data[data["n_cross_pairs"] > 0]
    if data.empty:
        plot_placeholder(
            path_base,
            "Cross-play needs at least two seeds",
            "The smoke run has no off-diagonal sender-receiver pairs. The full run renders self-play payoff minus cross-play payoff by design.",
        )
        return
    arms = list(reversed([arm for arm in ordered_arms(data)]))
    data["order"] = data["arm"].map({arm: i for i, arm in enumerate(arms)})
    data = data.sort_values("order")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _, row in data.iterrows():
        arm = str(row["arm"])
        y = float(row["order"])
        value = float(row["crossplay_welfare_gap"])
        ax.hlines(y, 0, value, color=NEUTRAL["base"], linewidth=1.2)
        ax.scatter(value, y, s=48, color=arm_color(arm), edgecolor=TOKENS["ink"], linewidth=0.6, zorder=3)
        ax.text(value + 0.015, y, f"{value:.2f}", va="center", fontsize=8.6, color=TOKENS["ink"])
    ax.axvline(0, color=NEUTRAL["dark"], linewidth=0.9)
    ax.set_yticks(data["order"])
    ax.set_yticklabels([arm_label(str(arm)) for arm in data["arm"]], fontsize=9)
    ax.set_xlabel("Self-play payoff minus cross-play payoff")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    xmax = max(0.05, float(data["crossplay_welfare_gap"].max()) * 1.18)
    xmin = min(-0.03, float(data["crossplay_welfare_gap"].min()) - 0.03)
    ax.set_xlim(xmin, xmax)
    add_chart_header(
        fig,
        ax,
        "Private codes fail with new partners",
        "Cross-play pairs one seed's sender with another seed's receiver. A large loss means the protocol works in self-play but is not portable across independently trained agents.",
    )
    save_figure(fig, path_base)


def plot_robustness_loss(robustness: pd.DataFrame, path_base: Path) -> None:
    data = robustness.groupby(["arm", "shift"], sort=False)["welfare"].mean().reset_index()
    train = data[data["shift"] == "train_distribution"][["arm", "welfare"]].rename(columns={"welfare": "train_welfare"})
    data = data.merge(train, on="arm", how="left")
    data = data[data["shift"] != "train_distribution"].copy()
    data["payoff_loss"] = data["train_welfare"] - data["welfare"]
    worst = data.loc[data.groupby("arm")["payoff_loss"].idxmax()].copy()
    arms = list(reversed(ordered_arms(worst)))
    worst["order"] = worst["arm"].map({arm: i for i, arm in enumerate(arms)})
    worst = worst.sort_values("order")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _, row in worst.iterrows():
        arm = str(row["arm"])
        y = float(row["order"])
        value = float(row["payoff_loss"])
        ax.hlines(y, 0, value, color=NEUTRAL["base"], linewidth=1.2)
        ax.scatter(value, y, s=48, color=arm_color(arm), edgecolor=TOKENS["ink"], linewidth=0.6, zorder=3)
        label = SHIFT_LABELS.get(str(row["shift"]), str(row["shift"]))
        ax.text(value + 0.008, y, f"{value:.2f} ({label})", va="center", fontsize=8.4, color=TOKENS["ink"])
    ax.axvline(0, color=NEUTRAL["dark"], linewidth=0.9)
    ax.set_yticks(worst["order"])
    ax.set_yticklabels([arm_label(str(arm)) for arm in worst["arm"]], fontsize=9)
    ax.set_xlabel("Largest payoff loss under distribution shift")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(-0.03, float(worst["payoff_loss"].min()) - 0.02), max(0.06, float(worst["payoff_loss"].max()) * 1.20))
    add_chart_header(
        fig,
        ax,
        "Robustness is a separate requirement",
        "Each dot shows the worst payoff loss across cleaner signals, noisier signals, knife-edge regimes, and harder regimes, relative to the training distribution.",
    )
    save_figure(fig, path_base)


def plot_placeholder(path_base: Path, title: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axis("off")
    add_chart_header(fig, ax, title, subtitle)
    ax.text(
        0.02,
        0.55,
        "No plotted estimate in this run.",
        transform=ax.transAxes,
        fontsize=11,
        color=TOKENS["muted"],
        ha="left",
        va="center",
    )
    save_figure(fig, path_base)


def plot_auditor_sample_curve(auditor_curve: pd.DataFrame, path_base: Path) -> None:
    data = auditor_curve.dropna(subset=["auditor_content_balanced_accuracy"]).copy()
    if data.empty:
        return
    stats = (
        data.groupby(["arm", "train_samples"], sort=False)["auditor_content_balanced_accuracy"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    stats["sem"] = stats["sem"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    chance = float(data["chance_accuracy"].dropna().iloc[0]) if data["chance_accuracy"].notna().any() else 1 / 3
    for arm in ordered_arms(stats):
        part = stats[stats["arm"] == arm].sort_values("train_samples")
        color = arm_color(arm)
        ax.plot(
            part["train_samples"],
            part["mean"],
            color=color,
            linewidth=1.15,
            marker="o",
            markersize=3.8,
            label=arm_label(arm),
        )
        ax.fill_between(
            part["train_samples"].to_numpy(dtype=float),
            (part["mean"] - part["sem"]).to_numpy(dtype=float),
            (part["mean"] + part["sem"]).to_numpy(dtype=float),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(chance, color=NEUTRAL["mid"], linewidth=0.9, linestyle=":")
    ax.text(float(stats["train_samples"].min()) * 0.92, chance, "Chance", fontsize=8.3, color=NEUTRAL["mid"], va="center", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Auditor training examples")
    ax.set_ylabel("Balanced accuracy reading strategic content")
    percent_axis(ax, "y")
    ax.set_ylim(max(0, chance - 0.08), 1.03)
    ax.set_xlim(float(stats["train_samples"].min()) * 0.75, float(stats["train_samples"].max()) * 1.15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}" if x >= 1000 else f"{int(x)}"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=3, borderaxespad=0, fontsize=8.3)
    add_chart_header(
        fig,
        ax,
        "Auditability has a sample-complexity curve",
        "The auditor only sees the monitored projection. Curves show how many labeled examples are needed to decode the sender's strategic content from that projection.",
    )
    save_figure(fig, path_base)


def plot_decision_rule(curves: pd.DataFrame, path_base: Path) -> None:
    data = curves.copy()
    data["message_group"] = np.where(data["arm"] == "no_comm", "No Messages", "Messages Allowed")
    stats = data.groupby(["message_group", "theta_mid"], sort=False)["join_rate"].agg(["mean", "sem"]).reset_index()
    stats["sem"] = stats["sem"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    colors = {"No Messages": NEUTRAL["mid"], "Messages Allowed": BLUE["mid"]}
    for group in ["No Messages", "Messages Allowed"]:
        part = stats[stats["message_group"] == group].sort_values("theta_mid")
        if part.empty:
            continue
        ax.plot(part["theta_mid"], part["mean"], color=colors[group], linewidth=1.25)
        ax.fill_between(
            part["theta_mid"].to_numpy(dtype=float),
            (part["mean"] - part["sem"]).to_numpy(dtype=float),
            (part["mean"] + part["sem"]).to_numpy(dtype=float),
            color=colors[group],
            alpha=0.11,
            linewidth=0,
        )
        last = part.iloc[-1]
        ax.text(float(last["theta_mid"]) + 0.035, float(last["mean"]), group, va="center", fontsize=8.8, color=colors[group])
    ax.set_xlabel("Regime strength")
    ax.set_ylabel("Join rate")
    percent_axis(ax, "y")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(float(stats["theta_mid"].min()) - 0.06, float(stats["theta_mid"].max()) + 0.24)
    add_chart_header(
        fig,
        ax,
        "Messages mainly move the decision boundary",
        "Curves average over seeds and communication designs. The learned policies join less often as regimes become harder; messages sharpen that response around the threshold.",
    )
    save_figure(fig, path_base)


def plot_algorithm_diagnostic(algorithm_results: pd.DataFrame, path_base: Path) -> None:
    if "learning_rule" not in algorithm_results:
        return
    data = (
        algorithm_results.groupby(["learning_rule", "arm"], sort=False)[
            ["message_value", "hidden_causal_influence", "welfare"]
        ]
        .mean()
        .reset_index()
    )
    data["label"] = data["learning_rule"].map({"ppo": "PPO", "reinforce": "REINFORCE"}).fillna(data["learning_rule"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.7), sharey=True)
    arms = [arm for arm in ARM_ORDER if arm in set(data["arm"])]
    y_positions = np.arange(len(arms))
    offset = {"PPO": -0.11, "REINFORCE": 0.11}
    colors = {"PPO": BLUE["mid"], "REINFORCE": ORANGE["mid"]}
    for ax, metric, title, percent in [
        (axes[0], "message_value", "Message value", False),
        (axes[1], "hidden_causal_influence", "Hidden use", True),
    ]:
        for label in ["PPO", "REINFORCE"]:
            part = data[data["label"] == label].set_index("arm").reindex(arms).reset_index()
            ax.scatter(
                part[metric],
                y_positions + offset[label],
                s=44,
                color=colors[label],
                edgecolor=TOKENS["ink"],
                linewidth=0.55,
                label=label,
                zorder=3,
            )
        ax.set_title(title, fontsize=10, color=TOKENS["ink"], pad=8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([arm_label(arm) for arm in arms], fontsize=9)
        ax.grid(axis="y", visible=False)
        if percent:
            percent_axis(ax, "x")
            ax.set_xlim(0, max(0.12, float(data[metric].max()) * 1.25))
        else:
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.set_xlim(min(0, float(data[metric].min()) - 0.02), max(0.05, float(data[metric].max()) * 1.2))
        sns.despine(ax=ax)
    add_chart_header(
        fig,
        axes[0],
        "PPO is a control choice, not just an optimizer",
        "Algorithm diagnostics rerun the core communication designs with clipped PPO and an unclipped REINFORCE-style policy gradient. Both optimize the same strategic payoff and monitoring penalties.",
    )
    axes[0].set_title("Message value", fontsize=10, color=TOKENS["ink"], pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles[:2],
        labels[:2],
        loc="upper left",
        frameon=False,
        fontsize=9,
        borderaxespad=0.7,
        handletextpad=0.4,
    )
    fig.subplots_adjust(wspace=0.24)
    save_figure(fig, path_base)


def write_figure_notes(
    outdir: Path,
    results: pd.DataFrame,
    robustness: pd.DataFrame | None,
    crossplay_summary: pd.DataFrame | None,
    auditor_curve: pd.DataFrame | None,
    algorithm_results: pd.DataFrame | None,
) -> None:
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
    rows = {str(row["arm"]): row for _, row in summary.iterrows()}
    lines = [
        "# Figure Notes",
        "",
        "The paper figures are designed around one claim: communication can raise strategic payoff while creating hidden receiver dependence that the auditor cannot see.",
        "",
    ]
    for arm in ordered_arms(results):
        row = rows[arm]
        live_monitor = row["live_monitor_content_accuracy_mean"]
        live_text = "not trained" if pd.isna(live_monitor) else f"{live_monitor:.1%}"
        lines.append(
            f"- {arm_label(arm)}: payoff {row['welfare_mean']:.3f}, message value {row['message_value_mean']:.3f}, hidden use {row['hidden_causal_influence_mean']:.1%}, offline probe {row['auditor_content_balanced_accuracy_mean']:.1%}, live monitor {live_text}."
        )
    if crossplay_summary is not None and not crossplay_summary.empty:
        data = crossplay_summary.dropna(subset=["crossplay_welfare_gap"])
        if not data.empty:
            worst = data.loc[data["crossplay_welfare_gap"].idxmax()]
            lines.append(
                f"- Largest cross-play loss: {arm_label(str(worst['arm']))}, {worst['crossplay_welfare_gap']:.3f} payoff points."
            )
    if robustness is not None and not robustness.empty:
        data = robustness.groupby(["arm", "shift"], sort=False)["welfare"].mean().reset_index()
        train = data[data["shift"] == "train_distribution"][["arm", "welfare"]].rename(columns={"welfare": "train_welfare"})
        shifted = data.merge(train, on="arm", how="left")
        shifted = shifted[shifted["shift"] != "train_distribution"].copy()
        shifted["loss"] = shifted["train_welfare"] - shifted["welfare"]
        worst = shifted.loc[shifted["loss"].idxmax()]
        lines.append(
            f"- Largest shift loss: {arm_label(str(worst['arm']))} under {SHIFT_LABELS.get(str(worst['shift']), str(worst['shift']))}, {worst['loss']:.3f} payoff points."
        )
    if auditor_curve is not None and not auditor_curve.empty:
        max_n = int(auditor_curve["train_samples"].max())
        final = auditor_curve[auditor_curve["train_samples"] == max_n].groupby("arm")["auditor_content_balanced_accuracy"].mean()
        if not final.empty:
            lines.append(f"- Best finite auditor at {max_n:,} examples: {arm_label(str(final.idxmax()))}, {final.max():.1%}.")
    if algorithm_results is not None and not algorithm_results.empty:
        lines.append("- The algorithm diagnostic is not the main identification claim; it checks whether the measured frontier is specific to PPO.")
    (outdir / "figures" / "figure_notes.md").write_text("\n".join(lines) + "\n")


def _pad_limits(ax: plt.Axes, x: Iterable[float], y: Iterable[float], *, x_min: float | None = None) -> None:
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    x_span = max(float(np.nanmax(x_values) - np.nanmin(x_values)), 0.04)
    y_span = max(float(np.nanmax(y_values) - np.nanmin(y_values)), 0.04)
    xmin = float(np.nanmin(x_values)) - 0.12 * x_span
    xmax = float(np.nanmax(x_values)) + 0.62 * x_span
    ymin = float(np.nanmin(y_values)) - 0.20 * y_span
    ymax = float(np.nanmax(y_values)) + 0.24 * y_span
    if x_min is not None:
        xmin = min(xmin, x_min)
    if math.isclose(ymin, ymax):
        ymin -= 0.02
        ymax += 0.02
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
