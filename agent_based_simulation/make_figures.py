#!/usr/bin/env python3
"""
Figure generation for "LLM Agents in Global Games."

Generates all 17 figures to figures/ with sequential numbering matching paper order.

Main figures:
  01. Core sigmoid — join fraction vs theta (pure + comm)
  02. Cross-model r-value forest plot
  03. Falsification triptych — pure / scramble / flip
  04. Cross-model r-value bar chart
  05. Communication effect — dumbbell + sigmoid overlay
  06. Agent-level threshold vs theoretical attack mass
  07. Information design — all designs on one canvas
  08. Treatment effect delta(theta)
  09. Censorship — curves + slope decomposition
  10. Infodesign falsification — scramble/flip
  11. Stability decomposition — single-channel
  12. Surveillance chilling effect
  13. Propaganda dose-response
  14. Cross-model infodesign replication

Appendix:
  A1. Agent count robustness
  A2. Network topology
  A3. Bandwidth robustness

Usage: uv run python agent_based_simulation/make_figures.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent / "output"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Two-column arxiv layout dimensions ────────────────────────────
COL_W = 3.4    # \columnwidth in inches (single-column figure)
TEXT_W = 7.0   # \textwidth in inches (figure* spanning both columns)

# ── Style (sized for 1:1 rendering in two-column layout) ─────────
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.family": "serif",
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
})

# ── Colors ────────────────────────────────────────────────────────

# Treatment colors
C_PURE = "#636363"
C_COMM = "#2c7bb6"
C_FLIP = "#d7191c"
C_SCRAMBLE = "#fdae61"
C_NET = "#1a9641"
C_SURV = "#7b3294"
C_PROP = "#CC79A7"

# Information design colors
C_BASELINE = "#636363"
C_STABILITY = "#2c7bb6"
C_INSTABILITY = "#d7191c"
C_CENS_UP = "#1a9641"
C_CENS_LO = "#e66101"
C_PUBLIC = "#7b3294"

DESIGN_COLORS = {
    "baseline": C_BASELINE,
    "stability": C_STABILITY,
    "instability": C_INSTABILITY,
    "censor_upper": C_CENS_UP,
    "censor_lower": C_CENS_LO,
    "public_signal": C_PUBLIC,
    "scramble": C_SCRAMBLE,
    "flip": C_FLIP,
    "stability_clarity": "#abd9e9",
    "stability_direction": "#74add1",
    "stability_dissent": "#4575b4",
}

DESIGN_LABELS = {
    "baseline": "Baseline",
    "stability": "Stability",
    "instability": "Instability",
    "censor_upper": "Censor upper",
    "censor_lower": "Censor lower",
    "public_signal": "Public signal",
    "scramble": "Scramble",
    "flip": "Flip",
    "stability_clarity": "Clarity only",
    "stability_direction": "Direction only",
    "stability_dissent": "Dissent only",
}

# Model colors and short names
MODEL_COLORS = {
    "mistralai--mistral-small-creative": "#2c7bb6",
    "allenai--olmo-3-7b-instruct": "#d7191c",
    "arcee-ai--trinity-large-preview_free": "#fdae61",
    "meta-llama--llama-3.3-70b-instruct": "#abdda4",
    "minimax--minimax-m2-her": "#7b3294",
    "mistralai--ministral-3b-2512": "#e66101",
    "openai--gpt-oss-120b": "#1a9641",
    "qwen--qwen3-235b-a22b-2507": "#5e4fa2",
    "qwen--qwen3-30b-a3b-instruct-2507": "#f46d43",
}

SHORT_NAMES = {
    "mistralai--mistral-small-creative": "Mistral-Small",
    "allenai--olmo-3-7b-instruct": "OLMo-7B",
    "arcee-ai--trinity-large-preview_free": "Trinity",
    "meta-llama--llama-3.3-70b-instruct": "Llama-3.3-70B",
    "minimax--minimax-m2-her": "MiniMax-M2",
    "mistralai--ministral-3b-2512": "Ministral-3B",
    "openai--gpt-oss-120b": "GPT-OSS-120B",
    "qwen--qwen3-235b-a22b-2507": "Qwen3-235B",
    "qwen--qwen3-30b-a3b-instruct-2507": "Qwen3-30B",
}


# ── Helpers ───────────────────────────────────────────────────────

def logistic(x, b0, b1):
    return 1.0 / (1.0 + np.exp(b0 + b1 * x))


def fit_logistic(df, theta_col="theta", join_col="join_fraction"):
    d = df.dropna(subset=[theta_col, join_col])
    x, y = d[theta_col].values, d[join_col].values
    try:
        popt, pcov = curve_fit(logistic, x, y, p0=[0, 2], maxfev=10000)
        return popt, pcov
    except RuntimeError:
        return np.array([0.0, 0.0]), np.zeros((2, 2))


def binned(df, theta_col="theta", join_col="join_fraction", n_bins=15):
    d = df.dropna(subset=[theta_col, join_col]).copy()
    d["bin"] = pd.qcut(d[theta_col], n_bins, duplicates="drop")
    g = d.groupby("bin", observed=True)
    centers = g[theta_col].mean().values
    means = g[join_col].mean().values
    ses = g[join_col].sem().values
    order = np.argsort(centers)
    return centers[order], means[order], ses[order]


def design_curve(df, design, theta_col="theta", join_col="join_fraction"):
    """Get mean join by theta for a specific design."""
    sub = df[df["design"] == design].dropna(subset=[theta_col, join_col])
    if len(sub) == 0:
        return np.array([]), np.array([]), np.array([])
    g = sub.groupby(theta_col)[join_col].agg(["mean", "sem", "count"]).reset_index()
    g = g.sort_values(theta_col)
    return g[theta_col].values, g["mean"].values, g["sem"].values


def load_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_all_csvs(directory, pattern="*summary*.csv"):
    p = Path(directory)
    csvs = sorted(p.glob(f"**/{pattern}"))
    if csvs:
        return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    return pd.DataFrame()


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ── Load data ─────────────────────────────────────────────────────

def load_model_data(model_dir):
    base = ROOT / model_dir
    data = {}
    for treatment in ["pure", "comm", "scramble", "flip"]:
        f = base / f"experiment_{treatment}_summary.csv"
        data[treatment] = load_csv(f)
    return data


PRIMARY = "mistralai--mistral-small-creative"
primary_data = load_model_data(PRIMARY)

# All models with pure data
ALL_MODELS = [d.name for d in ROOT.iterdir()
              if d.is_dir() and (d / "experiment_pure_summary.csv").exists()]
ALL_MODELS.sort()

# Cross-model comparison
comp = load_csv(ROOT / "comparison" / "model_comparison_summary.csv")

# Information design data
info_all = load_csv(ROOT / PRIMARY / "experiment_infodesign_all_summary.csv")
comm_df = load_csv(ROOT / PRIMARY / "experiment_comm_summary.csv")
pure_df = load_csv(ROOT / PRIMARY / "experiment_pure_summary.csv")

# Surveillance & propaganda
surv = load_all_csvs(ROOT / "surveillance")
surv_cens = load_all_csvs(ROOT / "surveillance-x-censorship")
prop2 = load_all_csvs(ROOT / "propaganda-k2")
prop5 = load_all_csvs(ROOT / "propaganda-k5")
prop10 = load_all_csvs(ROOT / "propaganda-k10")

# Bandwidth robustness
bw005 = load_all_csvs(ROOT / "bandwidth-005")
bw030 = load_all_csvs(ROOT / "bandwidth-030")

# Models with infodesign data
INFODESIGN_MODELS = [d.name for d in ROOT.iterdir()
                     if d.is_dir() and (d / "experiment_infodesign_all_summary.csv").exists()]
INFODESIGN_MODELS.sort()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 01: Core sigmoid
# ═══════════════════════════════════════════════════════════════════

def fig01_sigmoid():
    pure = primary_data["pure"]
    comm = primary_data["comm"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    theta_grid = np.linspace(-3.5, 3.5, 200)

    (b0_p, b1_p), _ = fit_logistic(pure)
    (b0_c, b1_c), _ = fit_logistic(comm)

    cp, mp, sep = binned(pure, n_bins=15)
    cc, mc, sec = binned(comm, n_bins=15)

    ax.plot(theta_grid, logistic(theta_grid, b0_p, b1_p),
            color=C_PURE, linewidth=1.2, zorder=2)
    ax.plot(theta_grid, logistic(theta_grid, b0_c, b1_c),
            color=C_COMM, linewidth=1.2, zorder=2)

    ax.scatter(cp, mp, color=C_PURE, s=12, alpha=0.8, zorder=3,
               edgecolors="none", label="Pure")
    ax.errorbar(cp, mp, yerr=sep * 1.96, fmt="none", ecolor=C_PURE,
                alpha=0.3, linewidth=0.5, zorder=1)

    ax.scatter(cc, mc, color=C_COMM, s=12, alpha=0.8, zorder=3,
               marker="s", edgecolors="none", label="Communication")
    ax.errorbar(cc, mc, yerr=sec * 1.96, fmt="none", ecolor=C_COMM,
                alpha=0.3, linewidth=0.5, zorder=1)

    ts_p = -b0_p / b1_p
    ts_c = -b0_c / b1_c
    ax.axvline(ts_p, color=C_PURE, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axvline(ts_c, color=C_COMM, linestyle=":", linewidth=0.5, alpha=0.5)

    r_p = stats.pearsonr(pure["theta"], pure["join_fraction"])[0]
    r_c = stats.pearsonr(comm["theta"], comm["join_fraction"])[0]
    ax.text(0.03, 0.03,
            f"Pure: r = {r_p:.2f}, $\\theta^*$ = {ts_p:.2f}\n"
            f"Comm: r = {r_c:.2f}, $\\theta^*$ = {ts_c:.2f}",
            transform=ax.transAxes, fontsize=6, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.8, edgecolor="#ccc", linewidth=0.4))

    ax.legend(loc="upper right", fontsize=6)
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-3.5, 3.5)

    save(fig, "fig01_sigmoid")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 02: Cross-model forest plot
# ═══════════════════════════════════════════════════════════════════

def fig02_cross_model():
    if len(comp) == 0:
        print("  SKIPPED fig02 — no comparison data")
        return

    df = comp.copy()
    if "model" not in df.columns:
        if df.index.name == "model":
            df = df.reset_index()
        else:
            print("  SKIPPED fig02 — no model column")
            return

    df["abs_r_pure"] = df["r_pure"].abs()
    df["abs_r_comm"] = df["r_comm"].abs()
    df = df.sort_values("abs_r_pure", ascending=True)

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    y = np.arange(len(df))
    short_names = [SHORT_NAMES.get(m, m[:20]) for m in df["model"]]

    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row["abs_r_pure"], row["abs_r_comm"]], [i, i],
                color="#e0e0e0", linewidth=1.2, zorder=1)

    ax.scatter(df["abs_r_pure"], y, color=C_PURE, s=25, zorder=3,
               edgecolors="white", linewidths=0.3, label="Pure")
    ax.scatter(df["abs_r_comm"], y, color=C_COMM, s=25, zorder=3,
               marker="s", edgecolors="white", linewidths=0.3, label="Comm")

    if "r_scramble" in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            rs = row.get("r_scramble", np.nan)
            if pd.notna(rs) and abs(rs) > 0.3:
                ax.plot(abs(rs), i, marker="x", color=C_FLIP, markersize=5,
                        markeredgewidth=1.5, zorder=4)

    mean_r = df["abs_r_pure"].mean()
    ax.axvline(mean_r, color="#999", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.text(mean_r + 0.01, len(df) - 0.3, f"mean = {mean_r:.2f}",
            fontsize=6, color="#999", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=6.5)
    ax.set_xlabel(r"$|r(\theta, \mathrm{join\ fraction})|$")
    ax.set_xlim(0.3, 1.0)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_PURE,
               markersize=5, label="Pure"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_COMM,
               markersize=5, label="Comm"),
        Line2D([0], [0], marker="x", color=C_FLIP, markersize=5,
               markeredgewidth=1.5, linestyle="none",
               label="Scramble fails ($|r| > 0.3$)"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower right")

    plt.tight_layout()
    save(fig, "fig02_cross_model")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 03: Falsification triptych
# ═══════════════════════════════════════════════════════════════════

def fig03_falsification():
    models_with_flip = [m for m in ALL_MODELS
                        if (ROOT / m / "experiment_flip_summary.csv").exists()]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.5), sharey=True)
    theta_grid = np.linspace(-3.5, 3.5, 200)

    ax = axes[0]
    for model in models_with_flip[:4]:
        df = load_csv(ROOT / model / "experiment_pure_summary.csv")
        if len(df) == 0:
            continue
        c, m, se = binned(df, n_bins=10)
        color = MODEL_COLORS.get(model, "#999")
        ax.scatter(c, m, color=color, s=8, alpha=0.7, edgecolors="none")
        try:
            (b0, b1), _ = fit_logistic(df)
            ax.plot(theta_grid, logistic(theta_grid, b0, b1),
                    color=color, linewidth=0.8, alpha=0.7,
                    label=SHORT_NAMES.get(model, model[:15]))
        except Exception:
            pass
    ax.set_title("A. Pure (signal intact)")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Join fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-3.5, 3.5)
    ax.legend(fontsize=5, loc="upper right")

    ax = axes[1]
    for model in models_with_flip[:4]:
        df = load_csv(ROOT / model / "experiment_scramble_summary.csv")
        if len(df) == 0:
            continue
        c, m, se = binned(df, n_bins=8)
        color = MODEL_COLORS.get(model, "#999")
        ax.scatter(c, m, color=color, s=8, alpha=0.7, edgecolors="none")
        mean_j = df["join_fraction"].mean()
        ax.axhline(mean_j, color=color, linestyle="--", linewidth=0.6, alpha=0.5)
        r_val = stats.pearsonr(df["theta"], df["join_fraction"])[0]
        ax.text(0.97, 0.97 - models_with_flip[:4].index(model) * 0.09,
                f"r = {r_val:.2f}", transform=ax.transAxes, fontsize=5.5,
                ha="right", va="top", color=color)
    ax.set_title("B. Scramble (signal destroyed)")
    ax.set_xlabel(r"$\theta$")
    ax.set_xlim(-3.5, 3.5)

    ax = axes[2]
    for model in models_with_flip[:4]:
        df = load_csv(ROOT / model / "experiment_flip_summary.csv")
        if len(df) == 0:
            continue
        c, m, se = binned(df, n_bins=8)
        color = MODEL_COLORS.get(model, "#999")
        ax.scatter(c, m, color=color, s=8, alpha=0.7, edgecolors="none")
        try:
            (b0, b1), _ = fit_logistic(df)
            ax.plot(theta_grid, logistic(theta_grid, b0, b1),
                    color=color, linewidth=0.8, alpha=0.7)
        except Exception:
            pass
        r_val = stats.pearsonr(df["theta"], df["join_fraction"])[0]
        ax.text(0.03, 0.97 - models_with_flip[:4].index(model) * 0.09,
                f"r = +{r_val:.2f}", transform=ax.transAxes, fontsize=5.5,
                ha="left", va="top", color=color)
    ax.set_title("C. Flip (signal reversed)")
    ax.set_xlabel(r"$\theta$")
    ax.set_xlim(-3.5, 3.5)

    plt.tight_layout()
    save(fig, "fig03_falsification")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 04: Cross-model r-value bar chart
# ═══════════════════════════════════════════════════════════════════

def fig04_r_summary():
    if len(comp) == 0:
        print("  SKIPPED fig04 — no comparison data")
        return

    df = comp.copy()
    if "model" not in df.columns:
        if df.index.name == "model":
            df = df.reset_index()
        else:
            return

    df = df.sort_values("r_pure", ascending=False)
    short_names = [SHORT_NAMES.get(m, m[:20]) for m in df["model"]]

    fig, ax = plt.subplots(figsize=(TEXT_W, 2.8))
    x = np.arange(len(df))
    w = 0.35

    ax.bar(x - w/2, df["r_pure"].abs(), w, color=C_PURE, label="Pure", edgecolor="none")
    ax.bar(x + w/2, df["r_comm"].abs(), w, color=C_COMM, label="Comm", edgecolor="none")

    if "r_scramble" in df.columns:
        r_scram = df["r_scramble"].abs().fillna(0)
        ax.scatter(x, r_scram, color=C_SCRAMBLE, marker="D", s=12,
                   zorder=4, label="Scramble $|r|$", edgecolors="none")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylabel(r"$|r(\theta, \mathrm{join\ fraction})|$")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    save(fig, "fig04_r_summary")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 05: Communication effect — dumbbell + sigmoid
# ═══════════════════════════════════════════════════════════════════

def fig05_communication():
    pure = primary_data["pure"]
    comm = primary_data["comm"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # Panel A: Dumbbell by theta bin
    ax = axes[0]
    n_q = 8
    all_theta = pd.concat([pure["theta"], comm["theta"]])
    _, bin_edges = pd.qcut(all_theta, n_q, retbins=True, duplicates="drop")

    pure_binned = pure.copy()
    comm_binned = comm.copy()
    pure_binned["tbin"] = pd.cut(pure_binned["theta"], bins=bin_edges, include_lowest=True)
    comm_binned["tbin"] = pd.cut(comm_binned["theta"], bins=bin_edges, include_lowest=True)

    gp = pure_binned.groupby("tbin", observed=True)["join_fraction"].agg(["mean", "sem"])
    gc = comm_binned.groupby("tbin", observed=True)["join_fraction"].agg(["mean", "sem"])

    shared = gp.index.intersection(gc.index)
    gp, gc = gp.loc[shared], gc.loc[shared]

    y = np.arange(len(shared))
    bin_labels = [f"{iv.mid:.1f}" for iv in shared]

    for i in range(len(shared)):
        ax.plot([gp.iloc[i]["mean"], gc.iloc[i]["mean"]], [i, i],
                color="#ddd", linewidth=1.5, zorder=1)

    ax.scatter(gp["mean"], y, color=C_PURE, s=20, zorder=3,
               label="Pure", edgecolors="none")
    ax.scatter(gc["mean"], y, color=C_COMM, s=20, zorder=3,
               marker="s", label="Comm", edgecolors="none")

    mid_idx = len(shared) // 2
    ax.axhspan(-0.5, mid_idx - 0.5, color=C_COMM, alpha=0.04, zorder=0)

    delta_low = gc.iloc[:mid_idx]["mean"].mean() - gp.iloc[:mid_idx]["mean"].mean()
    delta_high = gc.iloc[mid_idx:]["mean"].mean() - gp.iloc[mid_idx:]["mean"].mean()
    ax.text(0.97, 0.02,
            f"Weak regime $\\Delta$ = +{delta_low:.2f}\n"
            f"Strong regime $\\Delta$ = +{delta_high:.2f}",
            transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.8, edgecolor="#ccc", linewidth=0.4))

    ax.set_yticks(y)
    ax.set_yticklabels(bin_labels)
    ax.set_ylabel(r"$\theta$ bin center")
    ax.set_xlabel("Join fraction")
    ax.set_title("A. Communication effect by regime strength")
    ax.legend(fontsize=6, loc="upper right")

    # Panel B: Sigmoid overlay
    ax = axes[1]
    theta_grid = np.linspace(-3.5, 3.5, 200)

    for label, df, color, marker in [
        ("Pure", pure, C_PURE, "o"),
        ("Communication", comm, C_COMM, "s"),
    ]:
        c, m, se = binned(df, n_bins=15)
        ax.scatter(c, m, color=color, s=10, marker=marker, alpha=0.7,
                   zorder=3, edgecolors="none")
        ax.fill_between(c, m - 1.96 * se, m + 1.96 * se,
                        color=color, alpha=0.08, zorder=1)
        (b0, b1), _ = fit_logistic(df)
        ax.plot(theta_grid, logistic(theta_grid, b0, b1),
                color=color, linewidth=1.0, zorder=2, label=label)

    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_title("B. Sigmoid comparison")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-3.5, 3.5)

    plt.tight_layout()
    save(fig, "fig05_communication")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 06: Agent-level threshold
# ═══════════════════════════════════════════════════════════════════

def fig06_agent_threshold():
    pure = primary_data["pure"]
    if "z" not in pure.columns or len(pure) == 0:
        print("  SKIPPED fig06 — no z column in pure data")
        return

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    d = pure.dropna(subset=["theta", "join_fraction"]).copy()

    d["theta_bin"] = pd.qcut(d["theta"], 20, duplicates="drop")
    g = d.groupby("theta_bin", observed=True).agg(
        theta_mean=("theta", "mean"),
        join_mean=("join_fraction", "mean"),
        join_se=("join_fraction", "sem"),
        n=("join_fraction", "count"),
        attack_mean=("theoretical_attack", "mean"),
    ).reset_index().sort_values("theta_mean")

    (b0, b1), _ = fit_logistic(pure)
    theta_star = -b0 / b1

    ax.scatter(g["theta_mean"], g["join_mean"], color=C_PURE, s=15,
               zorder=3, edgecolors="none")
    ax.errorbar(g["theta_mean"], g["join_mean"], yerr=g["join_se"] * 1.96,
                fmt="none", ecolor=C_PURE, alpha=0.3, linewidth=0.5, zorder=1)

    ax.plot(g["theta_mean"], g["attack_mean"], color="#d62728",
            linestyle="--", linewidth=1.0, label="Theoretical attack mass", zorder=2)

    theta_grid = np.linspace(d["theta"].min(), d["theta"].max(), 200)
    ax.plot(theta_grid, logistic(theta_grid, b0, b1), color=C_PURE,
            linewidth=1.0, label="Fitted logistic", zorder=2)

    ax.axvline(theta_star, color="#333", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.text(theta_star + 0.1, 0.52, f"$\\theta^*$ = {theta_star:.2f}",
            fontsize=6, color="#333")

    ax.legend(loc="upper right", fontsize=6)
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_ylim(-0.03, 1.03)

    save(fig, "fig06_agent_threshold")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 07: Information design — all designs
# ═══════════════════════════════════════════════════════════════════

def fig07_all_designs():
    if len(info_all) == 0:
        print("  SKIPPED fig07 — no infodesign data")
        return

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    main_designs = ["baseline", "stability", "instability",
                    "censor_upper", "censor_lower", "public_signal"]
    markers = {"baseline": "o", "stability": "s", "instability": "^",
               "censor_upper": "D", "censor_lower": "v", "public_signal": "P"}

    for design in main_designs:
        theta, mean, sem = design_curve(info_all, design)
        if len(theta) == 0:
            continue
        color = DESIGN_COLORS[design]
        label = DESIGN_LABELS[design]
        marker = markers.get(design, "o")

        ax.plot(theta, mean, color=color, linewidth=1.0, zorder=2)
        ax.scatter(theta, mean, color=color, s=10, marker=marker,
                   zorder=3, edgecolors="none", label=label)
        ax.fill_between(theta, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=color, alpha=0.1, zorder=1)

    ax.legend(fontsize=5.5, loc="upper right", ncol=2)
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_ylim(-0.03, 0.85)

    save(fig, "fig07_all_designs")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 08: Treatment effect delta
# ═══════════════════════════════════════════════════════════════════

def fig08_treatment_effect():
    if len(info_all) == 0:
        print("  SKIPPED fig08 — no infodesign data")
        return

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    theta_base, mean_base, _ = design_curve(info_all, "baseline")
    if len(theta_base) == 0:
        print("  SKIPPED fig08 — no baseline data")
        return

    base_lookup = dict(zip(theta_base, mean_base))

    designs = ["stability", "instability", "censor_upper",
               "censor_lower", "public_signal"]
    markers = {"stability": "s", "instability": "^",
               "censor_upper": "D", "censor_lower": "v", "public_signal": "P"}

    for design in designs:
        theta, mean, sem = design_curve(info_all, design)
        if len(theta) == 0:
            continue

        delta = []
        delta_theta = []
        for t, m in zip(theta, mean):
            if t in base_lookup:
                delta.append(m - base_lookup[t])
                delta_theta.append(t)

        if not delta:
            continue

        color = DESIGN_COLORS[design]
        label = DESIGN_LABELS[design]
        marker = markers.get(design, "o")

        ax.plot(delta_theta, delta, color=color, linewidth=1.0, zorder=2)
        ax.scatter(delta_theta, delta, color=color, s=10, marker=marker,
                   zorder=3, edgecolors="none", label=label)

    ax.axhline(0, color="#333", linewidth=0.6, linestyle="-", zorder=1)
    ax.legend(fontsize=5.5, loc="best")
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel(r"$\Delta$ join fraction (design $-$ baseline)")

    save(fig, "fig08_treatment_effect")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 09: Censorship
# ═══════════════════════════════════════════════════════════════════

def fig09_censorship():
    if len(info_all) == 0:
        print("  SKIPPED fig09 — no infodesign data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.8), sharey=True)

    ax = axes[0]
    for design in ["baseline", "censor_upper", "censor_lower"]:
        theta, mean, sem = design_curve(info_all, design)
        if len(theta) == 0:
            continue
        color = DESIGN_COLORS[design]
        label = DESIGN_LABELS[design]

        ax.plot(theta, mean, color=color, linewidth=1.0, zorder=2)
        ax.scatter(theta, mean, color=color, s=12, zorder=3,
                   edgecolors="none", label=label)
        ax.fill_between(theta, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=color, alpha=0.1, zorder=1)

    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_title("A. Censorship inverts equilibrium")
    ax.set_ylim(-0.03, 0.85)

    ax = axes[1]
    slope_data = []
    for design in ["baseline", "censor_upper", "censor_lower",
                    "stability", "instability", "public_signal"]:
        sub = info_all[info_all["design"] == design].dropna(
            subset=["theta", "join_fraction"])
        if len(sub) < 5:
            continue
        slope = np.polyfit(sub["theta"], sub["join_fraction"], 1)[0]
        slope_data.append({"design": design, "slope": slope})

    if slope_data:
        sd = pd.DataFrame(slope_data).sort_values("slope")
        y = np.arange(len(sd))
        colors = [DESIGN_COLORS.get(d, "#999") for d in sd["design"]]
        labels = [DESIGN_LABELS.get(d, d) for d in sd["design"]]

        ax.barh(y, sd["slope"], color=colors, edgecolor="none", height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.axvline(0, color="#333", linewidth=0.6)
        ax.set_xlabel(r"OLS slope ($\Delta$join / $\Delta\theta$)")
        ax.set_title("B. Slope decomposition")

        for i, (_, row) in enumerate(sd.iterrows()):
            offset = 0.015 if row["slope"] >= 0 else -0.015
            ha = "left" if row["slope"] >= 0 else "right"
            ax.text(row["slope"] + offset, i, f'{row["slope"]:.3f}',
                    fontsize=6, va="center", ha=ha, color="#333")

    plt.tight_layout()
    save(fig, "fig09_censorship")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 10: Infodesign falsification
# ═══════════════════════════════════════════════════════════════════

def fig10_infodesign_falsification():
    if len(info_all) == 0:
        print("  SKIPPED fig10 — no infodesign data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.8), sharey=True)

    for ax, designs, title in [
        (axes[0], ["baseline", "stability", "scramble"],
         "A. Scramble destroys treatment effect"),
        (axes[1], ["baseline", "stability", "flip"],
         "B. Flip reverses treatment effect"),
    ]:
        for design in designs:
            theta, mean, sem = design_curve(info_all, design)
            if len(theta) == 0:
                continue
            color = DESIGN_COLORS[design]
            label = DESIGN_LABELS[design]
            ax.plot(theta, mean, color=color, linewidth=1.0, zorder=2)
            ax.scatter(theta, mean, color=color, s=12, zorder=3,
                       edgecolors="none", label=label)
            ax.fill_between(theta, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=color, alpha=0.1, zorder=1)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel(r"$\theta$ (regime strength)")
        ax.set_title(title)
        ax.set_ylim(-0.03, 0.85)

    axes[0].set_ylabel("Join fraction")

    plt.tight_layout()
    save(fig, "fig10_infodesign_falsification")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 11: Stability decomposition
# ═══════════════════════════════════════════════════════════════════

def fig11_decomposition():
    if len(info_all) == 0:
        print("  SKIPPED fig11 — no infodesign data")
        return

    decomp_designs = ["baseline", "stability", "stability_clarity",
                      "stability_direction", "stability_dissent"]
    available = [d for d in decomp_designs if d in info_all["design"].values]

    if len(available) < 3:
        print(f"  SKIPPED fig11 — only {len(available)} decomposition designs")
        return

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    markers = {"baseline": "o", "stability": "s", "stability_clarity": "^",
               "stability_direction": "D", "stability_dissent": "v"}

    for design in available:
        theta, mean, sem = design_curve(info_all, design)
        if len(theta) == 0:
            continue
        color = DESIGN_COLORS.get(design, "#999")
        label = DESIGN_LABELS.get(design, design)
        marker = markers.get(design, "o")

        ax.plot(theta, mean, color=color, linewidth=1.0, zorder=2)
        ax.scatter(theta, mean, color=color, s=10, marker=marker,
                   zorder=3, edgecolors="none", label=label)
        ax.fill_between(theta, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=color, alpha=0.1, zorder=1)

    ax.legend(fontsize=5.5, loc="upper right")
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_ylim(-0.03, 0.85)

    save(fig, "fig11_decomposition")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 12: Surveillance chilling effect
# ═══════════════════════════════════════════════════════════════════

def fig12_surveillance():
    if len(surv) == 0:
        print("  SKIPPED fig12 — missing surveillance data")
        return

    surv_dir = ROOT / "surveillance"
    surv_models = [d.name for d in surv_dir.iterdir()
                   if d.is_dir() and (d / "experiment_comm_summary.csv").exists()]
    surv_models.sort()

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 3.0),
                              gridspec_kw={"width_ratios": [1.3, 1]})
    theta_grid = np.linspace(-3.5, 3.5, 200)

    ax = axes[0]
    model_colors_surv = ["#2c7bb6", "#d7191c", "#1a9641"]
    deltas = []

    for i, model in enumerate(surv_models):
        color = model_colors_surv[i % len(model_colors_surv)]
        name = SHORT_NAMES.get(model, model[:15])

        comm_f = ROOT / model / "experiment_comm_summary.csv"
        if not comm_f.exists():
            continue
        comm = pd.read_csv(comm_f)

        surv_f = surv_dir / model / "experiment_comm_summary.csv"
        surv_m = pd.read_csv(surv_f)

        delta = surv_m["join_fraction"].mean() - comm["join_fraction"].mean()
        deltas.append({"model": name, "delta": delta})

        # Comm baseline (dashed)
        try:
            popt, _ = curve_fit(logistic, comm["theta"].values,
                                comm["join_fraction"].values, p0=[0, 2], maxfev=10000)
            ax.plot(theta_grid, logistic(theta_grid, *popt),
                    color=color, linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
        except RuntimeError:
            pass

        # Surveillance (solid)
        c, m, se = binned(surv_m, n_bins=10)
        ax.scatter(c, m, color=color, s=8, alpha=0.6, zorder=3, edgecolors="none")
        try:
            popt, _ = curve_fit(logistic, surv_m["theta"].values,
                                surv_m["join_fraction"].values, p0=[0, 2], maxfev=10000)
            ax.plot(theta_grid, logistic(theta_grid, *popt),
                    color=color, linewidth=1.0, zorder=2, label=name)
        except RuntimeError:
            ax.plot(c, m, color=color, linewidth=1.0, zorder=2, label=name)

    handles = ax.get_legend_handles_labels()[0]
    handles.append(Line2D([0], [0], color="#666", linestyle="--", linewidth=0.8,
                          label="Comm baseline"))
    handles.append(Line2D([0], [0], color="#666", linestyle="-", linewidth=1.0,
                          label="+ Surveillance"))
    ax.legend(handles=handles, fontsize=5.5, loc="upper right")
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-3.5, 3.5)
    ax.set_title("A. Surveillance chilling effect by model")

    ax = axes[1]
    if deltas:
        ddf = pd.DataFrame(deltas).sort_values("delta")
        y = np.arange(len(ddf))
        colors = model_colors_surv[:len(ddf)]
        ax.barh(y, ddf["delta"] * 100, color=colors, edgecolor="none", height=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(ddf["model"])
        ax.axvline(0, color="#333", linewidth=0.6)
        ax.set_xlabel("Chilling effect (pp)")
        ax.set_title("B. $\\Delta$ join fraction (surveillance $-$ comm)")

        for i, (_, row) in enumerate(ddf.iterrows()):
            ax.text(row["delta"] * 100 - 0.8, i,
                    f'{row["delta"]*100:.1f}pp', fontsize=6, va="center",
                    ha="right", color="white", fontweight="bold")

    plt.tight_layout()
    save(fig, "fig12_surveillance")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 13: Propaganda dose-response
# ═══════════════════════════════════════════════════════════════════

def fig13_propaganda():
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 3.0),
                              gridspec_kw={"width_ratios": [1, 1, 0.7]})

    def binned_local(df, col, n_bins=12):
        d = df.dropna(subset=["theta", col]).copy()
        d["bin"] = pd.qcut(d["theta"], n_bins, duplicates="drop")
        g = d.groupby("bin", observed=True)
        centers = g["theta"].mean().values
        means = g[col].mean().values
        ses = g[col].sem().values
        order = np.argsort(centers)
        return centers[order], means[order], ses[order]

    prop_specs = [
        ("k=0 (baseline)", comm_df, 0, C_COMM, 1.0, "o"),
        ("k=2 bots", prop2, 2, C_PROP, 0.9, "s"),
        ("k=5 bots", prop5, 5, C_PROP, 0.6, "^"),
        ("k=10 bots", prop10, 10, C_PROP, 0.3, "D"),
    ]

    for _, df, k, *_ in prop_specs:
        if len(df) > 0:
            df["join_fraction_real"] = df["n_join"] / (25 - k)

    for panel_idx, (ax, col, title) in enumerate(zip(
        axes[:2],
        ["join_fraction", "join_fraction_real"],
        ["A. Regime-level outcome\n(all 25 agents)",
         "B. Real citizen behavior\n(excluding propaganda bots)"],
    )):
        dose_means = []
        for label, df, k, base_color, alpha, marker in prop_specs:
            if len(df) == 0:
                dose_means.append(np.nan)
                continue
            c, m, se = binned_local(df, col, n_bins=12)
            ax.plot(c, m, color=base_color, linewidth=0.9, alpha=max(alpha, 0.4),
                    zorder=2, label=label)
            ax.scatter(c, m, color=base_color, s=8, marker=marker,
                       alpha=max(alpha, 0.5), zorder=3, edgecolors="none")
            ax.fill_between(c, m - 1.96 * se, m + 1.96 * se,
                            color=base_color, alpha=0.06, zorder=1)
            dose_means.append(df[col].mean())

        valid = [(i, v) for i, v in enumerate(dose_means) if not np.isnan(v)]
        if len(valid) >= 2:
            labels_k = ["k=0", "k=2", "k=5", "k=10"]
            text = "Mean join:\n" + "\n".join(
                [f"  {labels_k[i]}: {v:.1%}" for i, v in valid])
            ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=5.5,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="#ccc", linewidth=0.4))

        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"$\theta$ (regime strength)")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-0.03, 1.03)
        if panel_idx == 0:
            ax.set_ylabel("Join fraction")
            ax.legend(fontsize=5.5, loc="center right")

    # Panel C: Cross-model at k=5
    ax = axes[2]
    prop5_dir = ROOT / "propaganda-k5"
    prop5_models = [d.name for d in prop5_dir.iterdir()
                    if d.is_dir() and (d / "experiment_comm_summary.csv").exists()]
    prop5_models.sort()

    behavioral_deltas = []
    for model in prop5_models:
        p5 = pd.read_csv(prop5_dir / model / "experiment_comm_summary.csv")
        p5["join_fraction_real"] = p5["n_join"] / (25 - 5)
        comm_f = ROOT / model / "experiment_comm_summary.csv"
        if not comm_f.exists():
            continue
        comm_base = pd.read_csv(comm_f)
        delta = p5["join_fraction_real"].mean() - comm_base["join_fraction"].mean()
        behavioral_deltas.append({
            "model": SHORT_NAMES.get(model, model[:15]),
            "delta": delta,
        })

    if behavioral_deltas:
        bdf = pd.DataFrame(behavioral_deltas).sort_values("delta")
        y = np.arange(len(bdf))
        bar_colors = ["#2c7bb6", "#d7191c", "#1a9641"][:len(bdf)]
        ax.barh(y, bdf["delta"] * 100, color=bar_colors, edgecolor="none", height=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(bdf["model"], fontsize=6)
        ax.axvline(0, color="#333", linewidth=0.6)
        ax.set_xlabel("Behavioral $\\Delta$ (pp)")
        ax.set_title("C. Cross-model\n(k=5, real citizens)", fontsize=8)

        for i, (_, row) in enumerate(bdf.iterrows()):
            ax.text(row["delta"] * 100 - 0.3, i,
                    f'{row["delta"]*100:.1f}pp', fontsize=5.5, va="center",
                    ha="right", color="white", fontweight="bold")

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig13_propaganda")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 14: Cross-model infodesign replication
# ═══════════════════════════════════════════════════════════════════

def fig14_cross_model_infodesign():
    if not INFODESIGN_MODELS:
        print("  SKIPPED fig14 — no models with infodesign data")
        return

    results = []
    for model_dir in INFODESIGN_MODELS:
        df = load_csv(ROOT / model_dir / "experiment_infodesign_all_summary.csv")
        if len(df) == 0:
            continue
        for design in ["baseline", "stability"]:
            sub = df[df["design"] == design].dropna(subset=["theta", "join_fraction"])
            if len(sub) < 3:
                continue
            slope = np.polyfit(sub["theta"], sub["join_fraction"], 1)[0]
            mean_j = sub["join_fraction"].mean()
            results.append({
                "model": model_dir, "design": design,
                "slope": slope, "mean_join": mean_j, "n_obs": len(sub),
            })

    if not results:
        print("  SKIPPED fig14 — no valid results")
        return

    rdf = pd.DataFrame(results)
    pivot = rdf.pivot(index="model", columns="design", values="slope").dropna()
    if len(pivot) < 2:
        print("  SKIPPED fig14 — not enough models")
        return

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 3.0))

    ax = axes[0]
    models = pivot.index.tolist()
    y = np.arange(len(models))
    names = [SHORT_NAMES.get(m, m[:15]) for m in models]

    for i in range(len(models)):
        ax.plot([pivot.iloc[i]["baseline"], pivot.iloc[i]["stability"]],
                [i, i], color="#ddd", linewidth=1.5, zorder=1)

    ax.scatter(pivot["baseline"], y, color=C_BASELINE, s=20, zorder=3,
               label="Baseline", edgecolors="none")
    ax.scatter(pivot["stability"], y, color=C_STABILITY, s=20, zorder=3,
               marker="s", label="Stability", edgecolors="none")
    ax.axvline(0, color="#333", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel(r"OLS slope ($\Delta$join / $\Delta\theta$)")
    ax.set_title("A. Slope: Baseline vs Stability")
    ax.legend(fontsize=6, loc="best")

    ax = axes[1]
    pivot_mean = rdf.pivot(index="model", columns="design",
                           values="mean_join").dropna()
    models2 = pivot_mean.index.tolist()
    y2 = np.arange(len(models2))
    names2 = [SHORT_NAMES.get(m, m[:15]) for m in models2]

    for i in range(len(models2)):
        ax.plot([pivot_mean.iloc[i]["baseline"], pivot_mean.iloc[i]["stability"]],
                [i, i], color="#ddd", linewidth=1.5, zorder=1)

    ax.scatter(pivot_mean["baseline"], y2, color=C_BASELINE, s=20,
               zorder=3, label="Baseline", edgecolors="none")
    ax.scatter(pivot_mean["stability"], y2, color=C_STABILITY, s=20,
               zorder=3, marker="s", label="Stability", edgecolors="none")

    ax.set_yticks(y2)
    ax.set_yticklabels(names2, fontsize=6.5)
    ax.set_xlabel("Mean join fraction")
    ax.set_title("B. Mean join: Baseline vs Stability")
    ax.legend(fontsize=6, loc="best")

    plt.tight_layout()
    save(fig, "fig14_cross_model_infodesign")


# ═══════════════════════════════════════════════════════════════════
# FIGURE A1: Agent count robustness
# ═══════════════════════════════════════════════════════════════════

def figA1_agent_count():
    n_variants = {
        5:   ROOT / "mistralai--mistral-small-creative-n5" / PRIMARY / "experiment_pure_summary.csv",
        10:  ROOT / "mistralai--mistral-small-creative-n10" / PRIMARY / "experiment_pure_summary.csv",
        25:  ROOT / PRIMARY / "experiment_pure_summary.csv",
        50:  ROOT / "mistralai--mistral-small-creative-n50" / PRIMARY / "experiment_pure_summary.csv",
        100: ROOT / "mistralai--mistral-small-creative-n100" / PRIMARY / "experiment_pure_summary.csv",
    }

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    theta_grid = np.linspace(-3.5, 3.5, 200)

    cmap = plt.cm.viridis
    n_keys = sorted(n_variants.keys())
    colors = {n: cmap(i / (len(n_keys) - 1)) for i, n in enumerate(n_keys)}

    r_values = {}
    for n, path in sorted(n_variants.items()):
        df = load_csv(path)
        if len(df) == 0:
            continue
        c, m, se = binned(df, n_bins=10)
        color = colors[n]
        ax.scatter(c, m, color=color, s=6, alpha=0.6, edgecolors="none")

        (b0, b1), _ = fit_logistic(df)
        ax.plot(theta_grid, logistic(theta_grid, b0, b1),
                color=color, linewidth=0.9, label=f"n = {n}")

        r_val = stats.pearsonr(df["theta"], df["join_fraction"])[0]
        r_values[n] = r_val

    r_text = "\n".join([f"n={n}: r = {r:.2f}" for n, r in sorted(r_values.items())])
    ax.text(0.03, 0.03, r_text, transform=ax.transAxes, fontsize=5.5,
            va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.8, edgecolor="#ccc", linewidth=0.4))

    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-3.5, 3.5)

    save(fig, "figA1_agent_count")


# ═══════════════════════════════════════════════════════════════════
# FIGURE A2: Network topology
# ═══════════════════════════════════════════════════════════════════

def figA2_network():
    pure = primary_data["pure"]
    comm = primary_data["comm"]
    net8 = load_all_csvs(ROOT / "network-k8")

    if len(net8) == 0:
        print("  SKIPPED figA2 — no network-k8 data")
        return

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    theta_grid = np.linspace(-3.5, 3.5, 200)

    datasets = [
        ("Pure (no network)", pure, C_PURE, "--", "o"),
        ("Comm k=4", comm, C_COMM, "-", "s"),
        ("Comm k=8", net8, C_NET, "-", "D"),
    ]

    slopes = {}
    for label, df, color, ls, marker in datasets:
        if len(df) == 0:
            continue
        c, m, se = binned(df, n_bins=12)
        ax.scatter(c, m, color=color, s=10, marker=marker, alpha=0.7,
                   zorder=3, edgecolors="none")
        ax.fill_between(c, m - 1.96 * se, m + 1.96 * se,
                        color=color, alpha=0.08, zorder=1)

        (b0, b1), _ = fit_logistic(df)
        ax.plot(theta_grid, logistic(theta_grid, b0, b1),
                color=color, linestyle=ls, linewidth=1.0,
                zorder=2, label=label)
        slopes[label] = b1

    slope_text = "Slope $\\beta_1$:\n" + "\n".join(
        [f"  {k}: {v:.2f}" for k, v in slopes.items()])
    ax.text(0.97, 0.97, slope_text, transform=ax.transAxes, fontsize=5.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.8, edgecolor="#ccc", linewidth=0.4))

    ax.legend(fontsize=5.5, loc="center right")
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-3.5, 3.5)

    save(fig, "figA2_network")


# ═══════════════════════════════════════════════════════════════════
# FIGURE A3: Bandwidth robustness
# ═══════════════════════════════════════════════════════════════════

def figA3_bandwidth():
    if len(info_all) == 0:
        print("  SKIPPED figA3 — no infodesign data")
        return

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    bw_datasets = [
        ("BW = 0.05", bw005, "#fdae61"),
        ("BW = 0.15 (default)", info_all, C_BASELINE),
        ("BW = 0.30", bw030, "#1a9641"),
    ]

    for label, df, color in bw_datasets:
        if len(df) == 0:
            continue
        if "design" not in df.columns:
            continue
        sub = df[df["design"] == "stability"]
        if len(sub) == 0:
            continue

        theta, mean, sem = design_curve(sub.assign(design="stability"), "stability")
        if len(theta) == 0:
            g = sub.groupby("theta")["join_fraction"].agg(
                ["mean", "sem"]).reset_index().sort_values("theta")
            theta, mean, sem = g["theta"].values, g["mean"].values, g["sem"].values

        ax.plot(theta, mean, color=color, linewidth=1.0, zorder=2, label=label)
        ax.scatter(theta, mean, color=color, s=10, zorder=3, edgecolors="none")
        ax.fill_between(theta, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=color, alpha=0.1, zorder=1)

    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel(r"$\theta$ (regime strength)")
    ax.set_ylabel("Join fraction (stability design)")
    ax.set_ylim(-0.03, 0.85)

    save(fig, "figA3_bandwidth")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating all figures...")
    print(f"  Data root: {ROOT}")
    print(f"  Output: {FIG_DIR}\n")

    fig01_sigmoid()
    fig02_cross_model()
    fig03_falsification()
    fig04_r_summary()
    fig05_communication()
    fig06_agent_threshold()
    fig07_all_designs()
    fig08_treatment_effect()
    fig09_censorship()
    fig10_infodesign_falsification()
    fig11_decomposition()
    fig12_surveillance()
    fig13_propaganda()
    fig14_cross_model_infodesign()
    figA1_agent_count()
    figA2_network()
    figA3_bandwidth()

    print(f"\nAll figures saved to {FIG_DIR}")
