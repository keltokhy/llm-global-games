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

  15. Surveillance mechanism tests (belief-behavior gap)
  16. First-order beliefs (calibration + treatment divergence)
  17. Second-order beliefs
  18. B/C sweep — fitted vs theoretical cutoff
  19. Nonparametric monotonicity — E[belief | z-score bin] vs theory

Appendix:
  A1. Agent count robustness
  A2. Network topology
  A3. Bandwidth robustness
  A4. Calibration convergence

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

from models import (
    MODEL_COLORS as _MODEL_COLORS,
    SHORT_NAMES as _SHORT_NAMES,
    EXCLUDE_MODELS as _EXCLUDE_MODELS,
)

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "output"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
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

# Model colors, short names, and exclusions — imported from models.py
MODEL_COLORS = _MODEL_COLORS
SHORT_NAMES = _SHORT_NAMES
EXCLUDE_MODELS = _EXCLUDE_MODELS


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
              if d.is_dir() and (d / "experiment_pure_summary.csv").exists()
              and d.name not in EXCLUDE_MODELS]
ALL_MODELS.sort()

# Cross-model comparison
comp = load_csv(ROOT / "comparison" / "model_comparison_summary.csv")
if comp is not None and "model" in comp.columns:
    comp = comp[~comp["model"].isin(EXCLUDE_MODELS)]

# Information design data — combine all per-design CSVs
_info_csvs = sorted((ROOT / PRIMARY).glob("experiment_infodesign_*_summary.csv"))
if _info_csvs:
    info_all = pd.concat([pd.read_csv(c) for c in _info_csvs], ignore_index=True)
    info_all = info_all.drop_duplicates(subset=["design", "theta", "rep"], keep="last")
else:
    info_all = pd.DataFrame()

# The B/C sweep overwrites experiment_infodesign_baseline_summary.csv with
# the last theta_star target (0.75), shifting the baseline theta grid to
# [0.45, 1.05].  Replace with the canonical theta_star=0.50 slice from the
# sweep so the baseline grid matches all other designs ([0.20, 0.80]).
_bc_sweep_path = ROOT / PRIMARY / "experiment_bc_sweep_summary.csv"
if _bc_sweep_path.exists() and len(info_all) > 0:
    _bc = pd.read_csv(_bc_sweep_path)
    if "theta_star_target" in _bc.columns:
        _bc_baseline = _bc[np.isclose(_bc["theta_star_target"].astype(float), 0.50)].copy()
        if len(_bc_baseline) > 0:
            _bc_baseline["design"] = "baseline"
            info_all = info_all[info_all["design"] != "baseline"]
            info_all = pd.concat([info_all, _bc_baseline], ignore_index=True)
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
                     if d.is_dir() and (d / "experiment_infodesign_all_summary.csv").exists()
                     and d.name not in EXCLUDE_MODELS]
INFODESIGN_MODELS.sort()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 01: Core sigmoid
# ═══════════════════════════════════════════════════════════════════

def fig01_sigmoid():
    pure = primary_data["pure"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    theta_grid = np.linspace(-3.5, 3.5, 200)

    (b0_p, b1_p), _ = fit_logistic(pure)

    cp, mp, sep = binned(pure, n_bins=15)

    ax.plot(theta_grid, logistic(theta_grid, b0_p, b1_p),
            color=C_PURE, linewidth=1.2, zorder=2, label="Fitted logistic")
    ax.scatter(cp, mp, color=C_PURE, s=12, alpha=0.8, zorder=3,
               edgecolors="none")
    ax.errorbar(cp, mp, yerr=sep * 1.96, fmt="none", ecolor=C_PURE,
                alpha=0.3, linewidth=0.5, zorder=1)

    # Benchmark attack mass A(θ) = Φ((x* - θ)/σ) where x* = θ* + σΦ⁻¹(θ*)
    sigma = 0.3
    theta_star = 1.0 / (1.0 + 1.0)  # B/(1+B) with B=1
    x_star = theta_star + sigma * stats.norm.ppf(np.clip(theta_star, 1e-6, 1 - 1e-6))
    am_vals = stats.norm.cdf((x_star - theta_grid) / sigma)
    ax.plot(theta_grid, am_vals, color="#d62728", linewidth=1.0, linestyle="--",
            zorder=1, alpha=0.7, label="Benchmark $A(\\theta)$ (B=C=1)")

    ts_p = -b0_p / b1_p
    ax.axvline(ts_p, color=C_PURE, linestyle=":", linewidth=0.5, alpha=0.5)

    r_p = stats.pearsonr(pure["theta"], pure["join_fraction"])[0]
    ax.text(0.03, 0.03,
            f"$r(\\theta,J)$ = {r_p:.2f}, $\\hat{{\\theta}}^*$ = {ts_p:.2f}",
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
# FIGURE 05: Communication effect — dumbbell + sigmoid
# ═══════════════════════════════════════════════════════════════════

def fig05_communication():
    pure = primary_data["pure"]
    comm = primary_data["comm"]

    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 3.0))
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
    ax.legend(fontsize=6, loc="upper right")

    plt.tight_layout()
    save(fig, "fig05_communication")


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

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.8))

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
            if row["slope"] >= 0:
                offset, ha, color = 0.015, "left", "#333"
            elif row["slope"] < -0.8:
                # Very long negative bar: place label inside
                offset, ha, color = 0.015, "left", "white"
            else:
                offset, ha, color = -0.015, "right", "#333"
            ax.text(row["slope"] + offset, i, f'{row["slope"]:.3f}',
                    fontsize=6, va="center", ha=ha, color=color)

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
    # Discover surveillance CSVs recursively (handles nested slug dirs)
    surv_csvs = sorted(surv_dir.rglob("experiment_comm_summary.csv"))
    surv_model_paths = {}  # slug → path to surveillance CSV
    for csv_path in surv_csvs:
        slug = csv_path.parent.name
        if "--" not in slug:
            continue
        surv_model_paths[slug] = csv_path
    surv_models = sorted(surv_model_paths.keys())

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 3.0),
                              gridspec_kw={"width_ratios": [1.3, 1]})
    theta_grid = np.linspace(-3.5, 3.5, 200)

    ax = axes[0]
    _fallback_colors = ["#2c7bb6", "#d7191c", "#1a9641", "#5e4fa2", "#fdae61"]
    deltas = []

    for i, model in enumerate(surv_models):
        color = MODEL_COLORS.get(model, _fallback_colors[i % len(_fallback_colors)])
        name = SHORT_NAMES.get(model, model[:15])

        comm_f = ROOT / model / "experiment_comm_summary.csv"
        if not comm_f.exists():
            continue
        comm = pd.read_csv(comm_f)

        surv_m = pd.read_csv(surv_model_paths[model])

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
        colors = _fallback_colors[:len(ddf)]
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
# FIGURE 16: Beliefs (first-order)
# ═══════════════════════════════════════════════════════════════════

def fig16_beliefs():
    import json as _json

    MISTRAL = ROOT / "mistralai--mistral-small-creative"
    PROP_DIR = MISTRAL / "_beliefs_propaganda_k5" / "mistralai--mistral-small-creative"

    C_BELIEF_PURE = "#636363"
    C_BELIEF_COMM = "#2166ac"
    C_BELIEF_SURV = "#7b3294"
    C_BELIEF_PROP = "#d6604d"

    def _load_belief_v2_agents(log_path, slice_range=None):
        """Load v2 belief agents (those with second_order_belief_raw)."""
        if not log_path.exists():
            return []
        with open(log_path) as f:
            periods = _json.load(f)
        if slice_range is not None:
            periods = periods[slice_range]
        rows = []
        sigma = 0.3
        for p in periods:
            theta_star = p["theta_star"]
            for a in p.get("agents", []):
                if a.get("belief") is None or a.get("api_error"):
                    continue
                if "second_order_belief_raw" not in a:
                    continue
                signal = a["signal"]
                belief = a["belief"] / 100.0
                decision = 1 if a["decision"] == "JOIN" else 0
                posterior = stats.norm.cdf((theta_star - signal) / sigma)
                rows.append({"belief": belief, "decision": decision, "posterior": posterior})
        return rows

    def _load_belief_agents(log_path):
        """Load all belief agents (fallback for propaganda data)."""
        if not log_path.exists():
            return []
        with open(log_path) as f:
            periods = _json.load(f)
        rows = []
        sigma = 0.3
        for p in periods:
            theta_star = p["theta_star"]
            for a in p.get("agents", []):
                if a.get("belief") is None or a.get("api_error"):
                    continue
                signal = a["signal"]
                belief = a["belief"] / 100.0
                decision = 1 if a["decision"] == "JOIN" else 0
                posterior = stats.norm.cdf((theta_star - signal) / sigma)
                rows.append({"belief": belief, "decision": decision, "posterior": posterior})
        return rows

    def _bin_data(x, y, edges):
        centers, means, ses, counts = [], [], [], []
        for i in range(len(edges) - 1):
            mask = (x >= edges[i]) & (x < edges[i + 1])
            n = mask.sum()
            if n < 5:
                continue
            centers.append((edges[i] + edges[i + 1]) / 2)
            means.append(y[mask].mean())
            ses.append(y[mask].std() / np.sqrt(n))
            counts.append(n)
        return np.array(centers), np.array(means), np.array(ses), np.array(counts)

    # Load v2 belief data from main experiment logs (matching verify_paper_stats.py)
    pure = _load_belief_v2_agents(MISTRAL / "experiment_pure_log.json")
    comm = _load_belief_v2_agents(MISTRAL / "experiment_comm_log.json",
                                  slice_range=slice(None, -200))
    surv = _load_belief_v2_agents(MISTRAL / "experiment_comm_log.json",
                                  slice_range=slice(-200, None))
    prop_path = PROP_DIR / "experiment_comm_log.json"
    prop = _load_belief_agents(prop_path) if prop_path.exists() else []
    has_prop = len(prop) > 50

    if not pure:
        print("  SKIPPED fig16 — no belief data")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_W, 2.6))

    # Panel (a): Stated belief vs Bayesian posterior
    posteriors = np.array([r["posterior"] for r in pure])
    beliefs = np.array([r["belief"] for r in pure])

    edges = np.linspace(0, 1, 21)
    bc, bm, bse, _ = _bin_data(posteriors, beliefs, edges)

    ax_a.plot([0, 1], [0, 1], color="#cccccc", linewidth=0.8, linestyle="--",
              zorder=1, label="Perfect calibration")
    ax_a.errorbar(bc, bm, yerr=1.96 * bse, fmt="o", color=C_BELIEF_PURE,
                  markersize=4, elinewidth=0.6, capsize=0, zorder=3)

    slope, intercept, r_val, _, _ = stats.linregress(posteriors, beliefs)
    x_fit = np.linspace(0, 1, 100)
    ax_a.plot(x_fit, intercept + slope * x_fit, color=C_BELIEF_PURE, linewidth=1.0,
              linestyle="-", zorder=2)

    ax_a.set_xlabel(r"Bayesian posterior $P(\mathrm{success} \mid x_i)$")
    ax_a.set_ylabel("Stated belief")
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_aspect("equal")
    ax_a.text(0.05, 0.92, f"$r = {r_val:+.2f}$\nslope $= {slope:.2f}$",
              transform=ax_a.transAxes, fontsize=7, va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))
    ax_a.set_title("(a) Beliefs track Bayesian posterior", fontsize=8, loc="left")

    # Panel (b): Join rate by belief bin
    bin_edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.01])
    bin_labels = ["0\u201320", "20\u201340", "40\u201360", "60\u201380", "80\u2013100"]

    def _get_arrays(rows):
        return np.array([r["belief"] for r in rows]), np.array([r["decision"] for r in rows])

    pb, pd_ = _get_arrays(pure)
    cb, cd = _get_arrays(comm) if comm else (np.array([]), np.array([]))
    sb, sd = _get_arrays(surv) if surv else (np.array([]), np.array([]))

    pc, pm, pse, _ = _bin_data(pb, pd_, bin_edges)

    n_groups = 4 if has_prop else (3 if len(surv) > 0 else 2)
    bar_w = 0.8 / n_groups
    x_pos = np.arange(len(pc))
    offsets = np.linspace(-0.4 + bar_w / 2, 0.4 - bar_w / 2, n_groups)

    ax_b.bar(x_pos + offsets[0], pm, width=bar_w, color=C_BELIEF_PURE, alpha=0.85,
             label="Pure", zorder=3, edgecolor="white", linewidth=0.3)
    ax_b.errorbar(x_pos + offsets[0], pm, yerr=1.96 * pse, fmt="none",
                  ecolor=C_BELIEF_PURE, elinewidth=0.6, capsize=2, zorder=4)

    if len(comm) > 0:
        cc, cm, cse, _ = _bin_data(cb, cd, bin_edges)
        ax_b.bar(x_pos + offsets[1], cm, width=bar_w, color=C_BELIEF_COMM, alpha=0.85,
                 label="Communication", zorder=3, edgecolor="white", linewidth=0.3)
        ax_b.errorbar(x_pos + offsets[1], cm, yerr=1.96 * cse, fmt="none",
                      ecolor=C_BELIEF_COMM, elinewidth=0.6, capsize=2, zorder=4)

    if len(surv) > 0:
        sc, sm, sse, _ = _bin_data(sb, sd, bin_edges)
        idx = 2
        ax_b.bar(x_pos + offsets[idx], sm, width=bar_w, color=C_BELIEF_SURV, alpha=0.85,
                 label="Surveillance", zorder=3, edgecolor="white", linewidth=0.3)
        ax_b.errorbar(x_pos + offsets[idx], sm, yerr=1.96 * sse, fmt="none",
                      ecolor=C_BELIEF_SURV, elinewidth=0.6, capsize=2, zorder=4)

    if has_prop:
        rpb, rpd = _get_arrays(prop)
        rc, rm, rse, _ = _bin_data(rpb, rpd, bin_edges)
        ax_b.bar(x_pos + offsets[-1], rm, width=bar_w, color=C_BELIEF_PROP, alpha=0.85,
                 label="Propaganda $k{=}5$", zorder=3, edgecolor="white", linewidth=0.3)
        ax_b.errorbar(x_pos + offsets[-1], rm, yerr=1.96 * rse, fmt="none",
                      ecolor=C_BELIEF_PROP, elinewidth=0.6, capsize=2, zorder=4)

    ax_b.set_xlabel("Stated belief (percent)")
    ax_b.set_ylabel("Join rate")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(bin_labels)
    ax_b.set_ylim(-0.05, 1.08)
    ax_b.legend(loc="upper left", framealpha=0.9, edgecolor="#ccc")
    ax_b.set_title("(b) Actions diverge from beliefs under treatment", fontsize=8, loc="left")

    plt.tight_layout()
    save(fig, "fig16_beliefs")


# ═══════════════════════════════════════════════════════════════════
# FIGURE A4: Calibration convergence
# ═══════════════════════════════════════════════════════════════════

def figA4_calibration():
    """Two-panel figure: (A) convergence of fitted_center across rounds,
    (B) bar chart of final cutoff_center per model."""
    import json as _json

    # ── Load autocalibrate_history.csv for each model ────────────
    histories = {}
    for model in ALL_MODELS:
        hp = ROOT / model / "autocalibrate_history.csv"
        if hp.exists():
            histories[model] = pd.read_csv(hp)

    # ── Load calibrated_params JSON for each model ───────────────
    cal_params = {}
    for model in ALL_MODELS:
        pp = ROOT / model / f"calibrated_params_{model}.json"
        if pp.exists():
            with open(pp) as _f:
                cal_params[model] = _json.load(_f)

    if not histories and not cal_params:
        print("  SKIPPED figA4 — no calibration data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.6))

    # ── Panel A: Convergence trajectories ────────────────────────
    ax = axes[0]

    if histories:
        # Pick models with history data; use model colors
        all_centers_hist = []
        for model, hist_df in sorted(histories.items()):
            color = MODEL_COLORS.get(model, "#636363")
            label = SHORT_NAMES.get(model, model)
            rounds = hist_df["round"].values
            centers = hist_df["fitted_center"].values
            all_centers_hist.extend(centers.tolist())
            ax.plot(rounds, centers, "o-", color=color, markersize=3,
                    linewidth=0.9, label=label, zorder=2)

        # Convergence band
        ax.axhspan(-0.15, 0.15, color="#d9f0d3", alpha=0.35, zorder=0,
                   label=r"$|c| < 0.15$ (converged)")
        ax.axhline(0, color="#636363", linewidth=0.5, linestyle=":", zorder=1)

        ax.set_xlabel("Calibration round")
        ax.set_ylabel("Fitted logistic center $c$")
        ax.set_title("(A) Convergence of fitted center", fontsize=8)
        ax.legend(fontsize=5, loc="upper right", ncol=2, framealpha=0.8)

        # Nice x-axis ticks (integer rounds only)
        max_round = max(h["round"].max() for h in histories.values())
        ax.set_xticks(range(1, int(max_round) + 1))
        ax.set_xlim(0.6, max_round + 0.4)

        # Auto-scale y-axis to show all trajectories with some padding
        if all_centers_hist:
            y_lo = min(all_centers_hist) - 0.3
            y_hi = max(all_centers_hist) + 0.3
            ax.set_ylim(y_lo, y_hi)
    else:
        # No history CSVs available — show note
        ax.text(0.5, 0.5, "No autocalibrate_history.csv\nfiles found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="#999")
        ax.set_title("(A) Convergence of fitted center", fontsize=8)
        ax.set_xlabel("Calibration round")
        ax.set_ylabel("Fitted logistic center $c$")

    # ── Panel B: Final cutoff_center bar chart ───────────────────
    ax = axes[1]

    if cal_params:
        # Sort by absolute cutoff_center for visual clarity
        items = sorted(cal_params.items(),
                       key=lambda kv: abs(kv[1].get("cutoff_center", 0)))
        names = [SHORT_NAMES.get(m, m) for m, _ in items]
        centers = [p.get("cutoff_center", 0) for _, p in items]
        colors = [MODEL_COLORS.get(m, "#636363") for m, _ in items]

        bars = ax.barh(range(len(names)), centers, color=colors,
                       edgecolor="white", linewidth=0.3, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.axvline(0, color="#636363", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Calibrated cutoff center")
        ax.set_title("(B) Per-model calibration shift", fontsize=8)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, centers)):
            ha = "left" if val >= 0 else "right"
            offset = 0.03 if val >= 0 else -0.03
            ax.text(val + offset, i, f"{val:.2f}", va="center", ha=ha,
                    fontsize=5.5, color="#333")
    else:
        ax.text(0.5, 0.5, "No calibrated params found",
                transform=ax.transAxes, ha="center", va="center", fontsize=7)

    fig.tight_layout(w_pad=1.5)
    save(fig, "figA4_calibration")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 17: Second-order beliefs
# ═══════════════════════════════════════════════════════════════════

def fig17_second_order_beliefs():
    """2-panel: (A) second-order belief vs theta, (B) second-order belief vs actual join."""
    import json as _json

    _MISTRAL_DIR = ROOT / "mistralai--mistral-small-creative"

    _BELIEF_V2_SOURCES = {
        "pure": _MISTRAL_DIR / "experiment_pure_log.json",
        "comm": _MISTRAL_DIR / "experiment_comm_log.json",
        "surveillance": _MISTRAL_DIR / "experiment_comm_log.json",
    }

    sigma = 0.3

    def _load_v2_agents(treatment):
        path = _BELIEF_V2_SOURCES.get(treatment)
        if path is None or not path.exists():
            return []
        with open(path) as f:
            periods = _json.load(f)
        if not periods:
            return []

        if treatment == "surveillance":
            candidates = periods[-200:]
        elif treatment == "comm":
            candidates = periods[:-200] if len(periods) > 200 else periods
        else:
            candidates = periods

        rows = []
        for p in candidates:
            theta = p["theta"]
            theta_star = p["theta_star"]
            agents = p.get("agents") or []
            if not agents or "second_order_belief_raw" not in agents[0]:
                continue
            real = [a for a in agents if not a.get("is_propaganda", False)]
            if not real:
                continue
            period_join = sum(1 for a in real if a.get("decision") == "JOIN") / len(real)
            for a in agents:
                if a.get("api_error") or a.get("is_propaganda", False):
                    continue
                sob = a.get("second_order_belief")
                if sob is None:
                    continue
                rows.append({
                    "theta": theta,
                    "second_order_belief": sob / 100.0,
                    "period_join": period_join,
                })
        return rows

    treatments = {
        "pure": (C_PURE, "Pure"),
        "comm": (C_COMM, "Communication"),
        "surveillance": (C_SURV, "Surveillance"),
    }

    all_data = {}
    any_data = False
    for t in treatments:
        rows = _load_v2_agents(t)
        all_data[t] = rows
        if rows:
            any_data = True

    if not any_data:
        print("  SKIPPED fig17 — no second-order belief data available")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_W, 2.6))

    # Panel A: Second-order belief vs theta
    for t, (color, label) in treatments.items():
        rows = all_data[t]
        if not rows:
            continue
        thetas = np.array([r["theta"] for r in rows])
        sobs = np.array([r["second_order_belief"] for r in rows])

        # Binned means
        d = pd.DataFrame({"theta": thetas, "sob": sobs})
        d["bin"] = pd.qcut(d["theta"], 12, duplicates="drop")
        g = d.groupby("bin", observed=True)
        centers = g["theta"].mean().values
        means = g["sob"].mean().values
        ses = g["sob"].sem().values
        order = np.argsort(centers)
        centers, means, ses = centers[order], means[order], ses[order]

        ax_a.scatter(centers, means, color=color, s=12, alpha=0.8,
                     edgecolors="none", zorder=3)
        ax_a.errorbar(centers, means, yerr=1.96 * ses, fmt="none",
                      ecolor=color, alpha=0.3, linewidth=0.5, zorder=1)

        # Regression line
        slope, intercept, r_val, _, _ = stats.linregress(thetas, sobs)
        x_fit = np.linspace(thetas.min(), thetas.max(), 100)
        ax_a.plot(x_fit, intercept + slope * x_fit, color=color,
                  linewidth=1.0, zorder=2, label=f"{label} ($r = {r_val:+.2f}$)")

    ax_a.set_xlabel(r"$\theta$ (regime strength)")
    ax_a.set_ylabel("Second-order belief")
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.legend(fontsize=6, loc="upper right")
    ax_a.set_title("(a) Second-order belief vs regime strength", fontsize=8, loc="left")

    # Panel B: Second-order belief vs actual join rate
    for t, (color, label) in treatments.items():
        rows = all_data[t]
        if not rows:
            continue
        pj = np.array([r["period_join"] for r in rows])
        sobs = np.array([r["second_order_belief"] for r in rows])

        # Binned means
        d = pd.DataFrame({"pj": pj, "sob": sobs})
        d["bin"] = pd.qcut(d["pj"], 8, duplicates="drop")
        g = d.groupby("bin", observed=True)
        centers = g["pj"].mean().values
        means = g["sob"].mean().values
        ses = g["sob"].sem().values
        order = np.argsort(centers)
        centers, means, ses = centers[order], means[order], ses[order]

        ax_b.scatter(centers, means, color=color, s=12, alpha=0.8,
                     edgecolors="none", zorder=3)
        ax_b.errorbar(centers, means, yerr=1.96 * ses, fmt="none",
                      ecolor=color, alpha=0.3, linewidth=0.5, zorder=1)

        r_val = stats.pearsonr(pj, sobs)[0]
        slope, intercept, _, _, _ = stats.linregress(pj, sobs)
        x_fit = np.linspace(pj.min(), pj.max(), 100)
        ax_b.plot(x_fit, intercept + slope * x_fit, color=color,
                  linewidth=1.0, zorder=2, label=f"{label} ($r = {r_val:+.2f}$)")

    # Perfect calibration line
    ax_b.plot([0, 1], [0, 1], color="#cccccc", linewidth=0.8, linestyle="--",
              zorder=1, label="Perfect calibration")

    ax_b.set_xlabel("Actual join rate (period level)")
    ax_b.set_ylabel("Second-order belief")
    ax_b.set_xlim(-0.03, 1.03)
    ax_b.set_ylim(-0.03, 1.03)
    ax_b.legend(fontsize=6, loc="upper left")
    ax_b.set_title("(b) Calibration: belief vs actual join", fontsize=8, loc="left")

    plt.tight_layout()
    save(fig, "fig17_second_order_beliefs")


# ═══════════════════════════════════════════════════════════════════
# FIG 18: B/C SWEEP — θ̂* vs θ*
# ═══════════════════════════════════════════════════════════════════

def fig18_bc_sweep():
    """Scatter of fitted cutoff θ̂* vs theoretical θ* across 7 B/C ratios."""
    bc_path = ROOT / PRIMARY / "experiment_bc_sweep_summary.csv"
    if not bc_path.exists():
        print("  SKIP fig18_bc_sweep: no data")
        return

    df = pd.read_csv(bc_path)

    def _logistic4(x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    targets, fitted = [], []
    for ts in sorted(df["theta_star_target"].unique()):
        sub = df[df["theta_star_target"] == ts]
        grouped = sub.groupby("theta")["join_fraction_valid"].mean()
        try:
            popt, _ = curve_fit(
                _logistic4, grouped.index.values, grouped.values,
                p0=[1.0, -10.0, 0.5, 0.0], maxfev=5000,
            )
            targets.append(ts)
            fitted.append(popt[2])
        except Exception:
            pass

    if len(targets) < 3:
        print("  SKIP fig18_bc_sweep: too few fits")
        return

    r_val, p_val = stats.pearsonr(targets, fitted)

    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    ax.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=0.8, zorder=1)
    ax.scatter(targets, fitted, s=40, color=C_PURE, edgecolors="k",
               linewidths=0.5, zorder=3)

    # Label each point
    for t, f in zip(targets, fitted):
        ax.annotate(f"{t:.2f}", (t, f), textcoords="offset points",
                    xytext=(5, -10), fontsize=6, color="#444")

    ax.set_xlabel(r"Theoretical $\theta^* = B/(B+C)$")
    ax.set_ylabel(r"Fitted cutoff $\hat{\theta}^*$")
    ax.set_title(f"$r = {r_val:.3f}$", fontsize=8, loc="right", style="italic")
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.15, 0.85)
    ax.set_aspect("equal")
    plt.tight_layout()
    save(fig, "fig18_bc_sweep")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 19: Nonparametric signal monotonicity
# ═══════════════════════════════════════════════════════════════════

def fig19_nonparametric_beliefs():
    """Binned-mean plot: E[stated belief | z-score bin] with theoretical overlay."""
    import json as _json

    MISTRAL = ROOT / "mistralai--mistral-small-creative"
    log_path = MISTRAL / "experiment_pure_log.json"
    if not log_path.exists():
        print("  SKIP fig19_nonparametric_beliefs: no data")
        return

    with open(log_path) as f:
        periods = _json.load(f)

    sigma = 0.3
    n_bins = 10

    # Extract agent-level z_score, belief_pre, and per-period theoretical posterior
    z_scores = []
    beliefs = []
    posteriors = []

    for p in periods:
        theta_star = p["theta_star"]
        z_center = p["z"]
        for a in p.get("agents", []):
            if a.get("api_error"):
                continue
            # Prefer belief_pre (pre-decision belief); fall back to belief
            b = a.get("belief_pre") if a.get("belief_pre") is not None else a.get("belief")
            if b is None:
                continue
            z = a["z_score"]
            signal = a["signal"]
            # Theoretical Bayesian posterior: P(success | x_i) = Phi((theta* - x_i) / sigma)
            posterior = stats.norm.cdf((theta_star - signal) / sigma)

            z_scores.append(z)
            beliefs.append(b / 100.0)  # Convert from percentage to [0,1]
            posteriors.append(posterior)

    if len(z_scores) < 50:
        print("  SKIP fig19_nonparametric_beliefs: insufficient belief data")
        return

    z_scores = np.array(z_scores)
    beliefs = np.array(beliefs)
    posteriors = np.array(posteriors)

    # Bin by z-score
    # Use percentile-based bins to ensure roughly equal counts
    bin_edges = np.percentile(z_scores, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 0.01  # include minimum
    bin_edges[-1] += 0.01  # include maximum

    bin_centers_b = []
    bin_means_b = []
    bin_sems_b = []
    bin_centers_t = []
    bin_means_t = []

    for i in range(n_bins):
        mask = (z_scores >= bin_edges[i]) & (z_scores < bin_edges[i + 1])
        n = mask.sum()
        if n < 5:
            continue
        c = z_scores[mask].mean()
        bin_centers_b.append(c)
        bin_means_b.append(beliefs[mask].mean())
        bin_sems_b.append(beliefs[mask].std() / np.sqrt(n))
        bin_centers_t.append(c)
        bin_means_t.append(posteriors[mask].mean())

    bin_centers_b = np.array(bin_centers_b)
    bin_means_b = np.array(bin_means_b)
    bin_sems_b = np.array(bin_sems_b)
    bin_centers_t = np.array(bin_centers_t)
    bin_means_t = np.array(bin_means_t)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    # Theoretical curve (smooth, using median theta_star and z_center)
    med_theta_star = np.median([p["theta_star"] for p in periods
                                if "belief_pre" in p["agents"][0] or "belief" in p["agents"][0]])
    med_z_center = np.median([p["z"] for p in periods
                              if "belief_pre" in p["agents"][0] or "belief" in p["agents"][0]])
    z_grid = np.linspace(z_scores.min(), z_scores.max(), 200)
    signal_grid = z_grid * sigma + med_z_center
    theory_curve = stats.norm.cdf((med_theta_star - signal_grid) / sigma)
    ax.plot(z_grid, theory_curve, color="#d62728", linewidth=1.0, linestyle="--",
            alpha=0.7, zorder=2,
            label=r"Theoretical $P(\mathrm{success} \mid z)$")

    # Binned theoretical posterior means (accounts for varying theta* and z across periods)
    ax.plot(bin_centers_t, bin_means_t, color="#d62728", marker="D", markersize=3,
            linewidth=0, alpha=0.5, zorder=2)

    # Empirical stated beliefs
    ax.errorbar(bin_centers_b, bin_means_b, yerr=1.96 * bin_sems_b,
                fmt="o", color=C_PURE, markersize=4, elinewidth=0.6,
                capsize=2, zorder=3, label="Mean stated belief")

    # Correlation annotation
    r_val, p_val = stats.pearsonr(z_scores, beliefs)
    ax.text(0.97, 0.97,
            f"$r = {r_val:+.2f}$\n$N = {len(z_scores):,}$",
            transform=ax.transAxes, fontsize=6, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.8, edgecolor="#ccc", linewidth=0.4))

    ax.set_xlabel("$z$-score (calibrated signal)")
    ax.set_ylabel("Belief / P(success)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Nonparametric monotonicity", fontsize=9, loc="left")
    ax.legend(loc="center left", fontsize=6, framealpha=0.9, edgecolor="#ccc")

    plt.tight_layout()
    save(fig, "fig19_nonparametric_beliefs")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 20: Cross-generator sigmoid overlay
# ═══════════════════════════════════════════════════════════════════

def fig20_cross_generator():
    """Sigmoid overlay for 3 language generators × 2 models."""
    cross_gen_base = ROOT / "cross-generator"
    if not cross_gen_base.exists():
        print("  Skipping fig20 (no cross-generator data)")
        return

    variant_map = {
        "Mistral Small Creative": {
            "baseline": "mistralai/mistral-small-creative_baseline",
            "cable": "mistralai/mistral-small-creative_cable",
            "journalistic": "mistralai/mistral-small-creative_journalistic",
        },
        "Llama 3.3 70B": {
            "baseline": "meta-llama/llama-3.3-70b-instruct_baseline",
            "cable": "meta-llama/llama-3.3-70b-instruct_cable",
            "journalistic": "meta-llama/llama-3.3-70b-instruct_journalistic",
        },
    }

    variant_colors = {"baseline": C_PURE, "cable": "#2c7bb6", "journalistic": "#d7191c"}
    variant_styles = {"baseline": "-", "cable": "--", "journalistic": ":"}

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.8), sharey=True)

    for ax, (model, variants) in zip(axes, variant_map.items()):
        for variant_name, rel_path in variants.items():
            csvs = list((cross_gen_base / rel_path).rglob("experiment_pure_summary.csv"))
            if not csvs:
                continue
            df = pd.read_csv(csvs[0])
            jcol = "join_fraction_valid" if "join_fraction_valid" in df.columns else "join_fraction"
            if len(df) == 0:
                continue

            # Binned data
            centers, means, ses = binned(df, join_col=jcol)

            # Fitted logistic
            popt, _ = fit_logistic(df, join_col=jcol)

            # Plot binned points
            ax.errorbar(centers, means, yerr=1.96 * ses,
                        fmt="o", markersize=3, elinewidth=0.5, capsize=1.5,
                        color=variant_colors[variant_name], alpha=0.7)

            # Plot fitted curve
            theta_grid = np.linspace(df["theta"].min(), df["theta"].max(), 200)
            ax.plot(theta_grid, logistic(theta_grid, *popt),
                    color=variant_colors[variant_name],
                    linestyle=variant_styles[variant_name],
                    linewidth=1.2,
                    label=f"{variant_name.capitalize()}")

            # Annotate r
            r_val = np.corrcoef(df["theta"].values, df[jcol].values)[0, 1]
            # Put r values in legend label
            ax.plot([], [], " ", label=f"  $r = {r_val:.3f}$")

        ax.set_xlabel(r"$\theta$ (regime strength)")
        ax.set_title(model, fontsize=9, loc="left")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9, edgecolor="#ccc")

    axes[0].set_ylabel("Join fraction")

    plt.tight_layout()
    save(fig, "fig20_cross_generator")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 21: Placebo calibration
# ═══════════════════════════════════════════════════════════════════

def fig21_placebo_calibration():
    """Show that r is unchanged under placebo calibration shifts."""
    placebo_base = ROOT / "placebo-calibration"
    if not placebo_base.exists():
        print("  Skipping fig21 (no placebo-calibration data)")
        return

    # Collect r-values for each condition
    conditions = []

    for model_slug, model_name in [
        ("mistralai--mistral-small-creative", "Mistral"),
        ("meta-llama--llama-3.3-70b-instruct", "Llama 70B"),
    ]:
        # Calibrated baseline from main output
        base_csv = ROOT / model_slug / "experiment_pure_summary.csv"
        if base_csv.exists():
            df = pd.read_csv(base_csv)
            jcol = "join_fraction_valid" if "join_fraction_valid" in df.columns else "join_fraction"
            r_val = np.corrcoef(df["theta"].values, df[jcol].dropna().values[:len(df["theta"])])[0, 1]
            conditions.append((model_name, "Calibrated", r_val, df[jcol].mean()))

        # Placebo shifts
        for shift, label in [("0p3", "+0.3"), ("neg0p3", "-0.3")]:
            short = model_slug.split("--")[1]
            rel = f"{model_slug.split('--')[0]}/{short}_shift_{shift}"
            csvs = list((placebo_base / rel).rglob("experiment_pure_summary.csv"))
            if not csvs:
                continue
            df = pd.read_csv(csvs[0])
            jcol = "join_fraction_valid" if "join_fraction_valid" in df.columns else "join_fraction"
            valid = df.dropna(subset=[jcol])
            if len(valid) < 3:
                continue
            r_val = np.corrcoef(valid["theta"].values, valid[jcol].values)[0, 1]
            conditions.append((model_name, f"$\\Delta c = {label}$", r_val, valid[jcol].mean()))

    if not conditions:
        print("  Skipping fig21 (no data loaded)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.5))

    # Panel A: r-values
    labels = [f"{m}\n{c}" for m, c, _, _ in conditions]
    r_vals = [r for _, _, r, _ in conditions]
    colors = []
    for _, cond, _, _ in conditions:
        if "Calibrated" in cond:
            colors.append(C_PURE)
        elif "+0.3" in cond:
            colors.append("#d7191c")
        else:
            colors.append("#2c7bb6")

    ax1.barh(range(len(conditions)), r_vals, color=colors, height=0.6, alpha=0.8)
    ax1.set_yticks(range(len(conditions)))
    ax1.set_yticklabels(labels, fontsize=6)
    ax1.set_xlabel("$r(\\theta, J)$")
    ax1.set_title("(a) Correlation unchanged", fontsize=9, loc="left")
    ax1.set_xlim(-1, 0)
    ax1.invert_yaxis()

    # Panel B: mean join shifts
    means = [mj for _, _, _, mj in conditions]
    ax2.barh(range(len(conditions)), means, color=colors, height=0.6, alpha=0.8)
    ax2.set_yticks(range(len(conditions)))
    ax2.set_yticklabels(labels, fontsize=6)
    ax2.set_xlabel("Mean join fraction")
    ax2.set_title("(b) Mean join shifts with center", fontsize=9, loc="left")
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()

    plt.tight_layout()
    save(fig, "fig21_placebo_calibration")


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
    fig05_communication()
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
    fig16_beliefs()
    figA4_calibration()
    fig17_second_order_beliefs()
    fig18_bc_sweep()
    fig19_nonparametric_beliefs()
    fig20_cross_generator()
    fig21_placebo_calibration()

    print(f"\nAll figures saved to {FIG_DIR}")
