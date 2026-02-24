"""Analyze Paper 2 batch results: provenance, rhetoric, z-centering, decomposition, censorship."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ── Style (matching make_figures.py) ─────────────────────────────────
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.family": "serif",
    "lines.linewidth": 1.0, "lines.markersize": 4,
})

COL_W, TEXT_W = 3.4, 7.0
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures" / "batch_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = SCRIPT_DIR / "output"

# ── Colors ───────────────────────────────────────────────────────────
C_BASELINE = "#636363"
C_STABILITY = "#2c7bb6"
C_INSTABILITY = "#d7191c"
C_CENS_UP = "#1a9641"
C_CENS_LO = "#e66101"
C_PUBLIC = "#7b3294"

DESIGN_COLORS = {
    "baseline": C_BASELINE,
    "stability": C_STABILITY, "instability": C_INSTABILITY,
    "censor_upper": C_CENS_UP, "censor_lower": C_CENS_LO,
    "public_signal": C_PUBLIC,
    "stability_clarity": "#abd9e9", "stability_direction": "#74add1",
    "stability_dissent": "#4575b4",
    "provenance_independent": "#1b9e77", "provenance_state": "#d95f02",
    "provenance_social": "#7570b3",
    "rhetoric_hot": "#e41a1c", "rhetoric_cold": "#377eb8",
    "scramble": "#fdae61", "flip": "#d7191c",
}

DESIGN_LABELS = {
    "baseline": "Baseline", "stability": "Stability", "instability": "Instability",
    "censor_upper": "Censor upper", "censor_lower": "Censor lower",
    "public_signal": "Public signal",
    "stability_clarity": "Clarity only", "stability_direction": "Direction only",
    "stability_dissent": "Dissent only",
    "provenance_independent": "Independent observers",
    "provenance_state": "State media", "provenance_social": "Social media",
    "rhetoric_hot": "Hot rhetoric", "rhetoric_cold": "Cold rhetoric",
    "scramble": "Scramble", "flip": "Flip",
}

MODEL_LABELS = {
    "meta-llama--llama-3.3-70b-instruct": "Llama 70B",
    "mistralai--mistral-small-creative": "Mistral Small",
    "qwen--qwen3-30b-a3b-instruct-2507": "Qwen3 30B",
    "allenai--olmo-3-7b-instruct": "OLMo 7B",
    "mistralai--ministral-3b-2512": "Ministral 3B",
}

MODEL_COLORS = {
    "meta-llama--llama-3.3-70b-instruct": "#2c7bb6",
    "mistralai--mistral-small-creative": "#1a9641",
    "qwen--qwen3-30b-a3b-instruct-2507": "#7b3294",
    "allenai--olmo-3-7b-instruct": "#d7191c",
    "mistralai--ministral-3b-2512": "#fdae61",
}


# ── Helpers ──────────────────────────────────────────────────────────
def logistic(x, b0, b1):
    return 1.0 / (1.0 + np.exp(b0 + b1 * x))

def fit_logistic(df, theta_col="theta_relative", join_col=None):
    if join_col is None:
        join_col = jcol(df)
    d = df.dropna(subset=[theta_col, join_col])
    x, y = d[theta_col].values, d[join_col].values
    try:
        popt, _ = curve_fit(logistic, x, y, p0=[0, 2], maxfev=10000)
        return popt
    except RuntimeError:
        return np.array([0.0, 0.0])

def jcol(df):
    if "join_fraction_valid" in df.columns and df["join_fraction_valid"].notna().sum() > 0:
        return "join_fraction_valid"
    return "join_fraction"

def binned(df, theta_col="theta_relative", join_col=None, n_bins=9):
    if join_col is None:
        join_col = jcol(df)
    d = df.dropna(subset=[theta_col, join_col]).copy()
    d["bin"] = pd.cut(d[theta_col], n_bins, duplicates="drop")
    g = d.groupby("bin", observed=True)
    centers = g[theta_col].mean().values
    means = g[join_col].mean().values
    ses = g[join_col].sem().values
    order = np.argsort(centers)
    return centers[order], means[order], ses[order]

def corr_r(df, theta_col="theta_relative", join_col=None):
    if join_col is None:
        join_col = jcol(df)
    d = df.dropna(subset=[theta_col, join_col])
    if len(d) < 5:
        return float("nan")
    r, _ = pearsonr(d[theta_col], d[join_col])
    return r

def plot_design(ax, df, design, theta_col="theta_relative", label=None, alpha=0.8):
    dd = df[df["design"] == design]
    if dd.empty:
        return
    color = DESIGN_COLORS.get(design, "#999999")
    lbl = label or DESIGN_LABELS.get(design, design)
    c, m, se = binned(dd, theta_col=theta_col)
    ax.errorbar(c, m, yerr=1.96*se, fmt="none", ecolor=color, alpha=0.3, linewidth=0.5, zorder=1)
    ax.scatter(c, m, color=color, s=14, alpha=alpha, zorder=3, edgecolors="none", label=lbl)
    # Fit
    b0, b1 = fit_logistic(dd, theta_col=theta_col)
    xg = np.linspace(dd[theta_col].min() - 0.05, dd[theta_col].max() + 0.05, 200)
    ax.plot(xg, logistic(xg, b0, b1), color=color, linewidth=1.2, zorder=2, alpha=0.7)

def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ── Load data ────────────────────────────────────────────────────────
def load_model(slug):
    p = OUT / slug / "experiment_infodesign_all_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

print("Loading data...")
llama = load_model("meta-llama--llama-3.3-70b-instruct")
mistral = load_model("mistralai--mistral-small-creative")
qwen = load_model("qwen--qwen3-30b-a3b-instruct-2507")
olmo = load_model("allenai--olmo-3-7b-instruct")
ministral = load_model("mistralai--ministral-3b-2512")

zc_path = OUT / "z-centered" / "meta-llama--llama-3.3-70b-instruct" / "experiment_infodesign_all_summary.csv"
llama_zc = pd.read_csv(zc_path) if zc_path.exists() else pd.DataFrame()

print(f"  Llama 70B: {len(llama)} rows, designs: {sorted(llama['design'].unique())}")
print(f"  Llama 70B z-centered: {len(llama_zc)} rows")
print(f"  Mistral: {len(mistral)} rows")
print(f"  Qwen: {len(qwen)} rows")
print(f"  OLMo: {len(olmo)} rows")
print(f"  Ministral: {len(ministral)} rows")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Provenance Treatment (Llama 70B)
# ══════════════════════════════════════════════════════════════════════
print("\nFig 1: Provenance treatment...")
fig, ax = plt.subplots(figsize=(COL_W, 2.8))
for d in ["baseline", "provenance_independent", "provenance_state", "provenance_social"]:
    plot_design(ax, llama, d)
ax.set_xlabel("$\\theta - \\theta^*$")
ax.set_ylabel("Join fraction")
ax.set_title("Provenance treatment (Llama 70B)")
ax.legend(fontsize=6, loc="upper right")
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.5, zorder=0)
ax.axvline(0.0, color="#ccc", linestyle=":", linewidth=0.5, zorder=0)
plt.tight_layout()
save(fig, "provenance_sigmoid")

# Stats
print("  Provenance r-values:")
for d in ["baseline", "provenance_independent", "provenance_state", "provenance_social"]:
    dd = llama[llama["design"] == d]
    r = corr_r(dd)
    mean_j = dd[jcol(dd)].mean()
    print(f"    {DESIGN_LABELS.get(d, d):>25s}: r={r:+.3f}, mean_join={mean_j:.3f}, n={len(dd)}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Rhetoric Treatment (Llama 70B)
# ══════════════════════════════════════════════════════════════════════
print("\nFig 2: Rhetoric treatment...")
fig, ax = plt.subplots(figsize=(COL_W, 2.8))
for d in ["baseline", "rhetoric_hot", "rhetoric_cold"]:
    plot_design(ax, llama, d)
ax.set_xlabel("$\\theta - \\theta^*$")
ax.set_ylabel("Join fraction")
ax.set_title("Hot/cold rhetoric (Llama 70B)")
ax.legend(fontsize=6, loc="upper right")
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.5, zorder=0)
ax.axvline(0.0, color="#ccc", linestyle=":", linewidth=0.5, zorder=0)
plt.tight_layout()
save(fig, "rhetoric_sigmoid")

print("  Rhetoric r-values:")
for d in ["baseline", "rhetoric_hot", "rhetoric_cold"]:
    dd = llama[llama["design"] == d]
    r = corr_r(dd)
    mean_j = dd[jcol(dd)].mean()
    print(f"    {DESIGN_LABELS.get(d, d):>15s}: r={r:+.3f}, mean_join={mean_j:.3f}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: z-centering comparison (Llama 70B)
# ══════════════════════════════════════════════════════════════════════
print("\nFig 3: z-centering comparison...")
fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.5), sharey=True)
for i, d in enumerate(["baseline", "stability", "instability"]):
    ax = axes[i]
    # Old (z=0) from main run
    dd_old = llama[llama["design"] == d]
    if not dd_old.empty:
        c, m, se = binned(dd_old)
        ax.scatter(c, m, color=DESIGN_COLORS.get(d, "#999"), s=12, alpha=0.5,
                   edgecolors="none", label=f"{DESIGN_LABELS.get(d,d)} (z=0)", marker="o")
        ax.errorbar(c, m, yerr=1.96*se, fmt="none", ecolor=DESIGN_COLORS.get(d, "#999"),
                    alpha=0.2, linewidth=0.5)
        b0, b1 = fit_logistic(dd_old)
        xg = np.linspace(-0.35, 0.35, 200)
        ax.plot(xg, logistic(xg, b0, b1), color=DESIGN_COLORS.get(d, "#999"),
                linewidth=1.0, alpha=0.5, linestyle="--")

    # New (z=θ*) from z-centered run
    dd_new = llama_zc[llama_zc["design"] == d] if not llama_zc.empty else pd.DataFrame()
    if not dd_new.empty:
        c2, m2, se2 = binned(dd_new)
        ax.scatter(c2, m2, color=DESIGN_COLORS.get(d, "#999"), s=14, alpha=0.9,
                   edgecolors="none", label=f"{DESIGN_LABELS.get(d,d)} (z=θ*)", marker="s")
        ax.errorbar(c2, m2, yerr=1.96*se2, fmt="none", ecolor=DESIGN_COLORS.get(d, "#999"),
                    alpha=0.3, linewidth=0.5)
        b0, b1 = fit_logistic(dd_new)
        ax.plot(xg, logistic(xg, b0, b1), color=DESIGN_COLORS.get(d, "#999"),
                linewidth=1.2, alpha=0.9)

    ax.set_title(DESIGN_LABELS.get(d, d))
    ax.set_xlabel("$\\theta - \\theta^*$")
    ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.5, zorder=0)
    ax.axvline(0.0, color="#ccc", linestyle=":", linewidth=0.5, zorder=0)
    ax.legend(fontsize=5.5, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
axes[0].set_ylabel("Join fraction")
fig.suptitle("z-centering: z=0 (dashed) vs z=θ* (solid) — Llama 70B", fontsize=9, y=1.02)
plt.tight_layout()
save(fig, "z_centering_comparison")

print("  z-centering r-values:")
for d in ["baseline", "stability", "instability"]:
    dd_old = llama[llama["design"] == d]
    dd_new = llama_zc[llama_zc["design"] == d] if not llama_zc.empty else pd.DataFrame()
    r_old = corr_r(dd_old) if not dd_old.empty else float("nan")
    r_new = corr_r(dd_new) if not dd_new.empty else float("nan")
    print(f"    {DESIGN_LABELS.get(d,d):>12s}: z=0 r={r_old:+.3f}, z=θ* r={r_new:+.3f}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Decomposition replication (Llama 70B)
# ══════════════════════════════════════════════════════════════════════
print("\nFig 4: Decomposition replication...")
fig, ax = plt.subplots(figsize=(COL_W, 2.8))
for d in ["baseline", "stability", "stability_clarity", "stability_direction", "stability_dissent"]:
    plot_design(ax, llama, d)
ax.set_xlabel("$\\theta - \\theta^*$")
ax.set_ylabel("Join fraction")
ax.set_title("Stability decomposition (Llama 70B)")
ax.legend(fontsize=5.5, loc="upper right")
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.5, zorder=0)
ax.axvline(0.0, color="#ccc", linestyle=":", linewidth=0.5, zorder=0)
plt.tight_layout()
save(fig, "decomposition_llama")

print("  Decomposition r-values (Llama 70B):")
for d in ["baseline", "stability", "stability_clarity", "stability_direction", "stability_dissent"]:
    dd = llama[llama["design"] == d]
    r = corr_r(dd)
    mean_j = dd[jcol(dd)].mean()
    print(f"    {DESIGN_LABELS.get(d, d):>18s}: r={r:+.3f}, mean_join={mean_j:.3f}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Censorship across models
# ══════════════════════════════════════════════════════════════════════
print("\nFig 5: Censorship across models...")
models_data = {
    "meta-llama--llama-3.3-70b-instruct": llama,
    "qwen--qwen3-30b-a3b-instruct-2507": qwen,
    "allenai--olmo-3-7b-instruct": olmo,
    "mistralai--ministral-3b-2512": ministral,
}

n_models = len(models_data)
fig, axes = plt.subplots(1, n_models, figsize=(TEXT_W, 2.5), sharey=True)
for i, (slug, mdf) in enumerate(models_data.items()):
    ax = axes[i]
    for d in ["baseline", "censor_upper", "censor_lower"]:
        dd = mdf[mdf["design"] == d]
        if dd.empty:
            continue
        color = DESIGN_COLORS.get(d, "#999")
        lbl = DESIGN_LABELS.get(d, d)
        c, m, se = binned(dd)
        ax.scatter(c, m, color=color, s=10, alpha=0.8, edgecolors="none", label=lbl, zorder=3)
        ax.errorbar(c, m, yerr=1.96*se, fmt="none", ecolor=color, alpha=0.25, linewidth=0.5, zorder=1)
        b0, b1 = fit_logistic(dd)
        xg = np.linspace(-0.35, 0.35, 200)
        ax.plot(xg, logistic(xg, b0, b1), color=color, linewidth=1.0, zorder=2, alpha=0.7)
    ax.set_title(MODEL_LABELS.get(slug, slug), fontsize=7)
    ax.set_xlabel("$\\theta - \\theta^*$")
    ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.5, zorder=0)
    ax.axvline(0.0, color="#ccc", linestyle=":", linewidth=0.5, zorder=0)
    ax.set_ylim(-0.05, 1.05)
    if i == 0:
        ax.set_ylabel("Join fraction")
    if i == n_models - 1:
        ax.legend(fontsize=5, loc="upper right")
fig.suptitle("Censorship treatment across models", fontsize=9, y=1.02)
plt.tight_layout()
save(fig, "censorship_cross_model")

print("  Censorship r-values by model:")
for slug, mdf in models_data.items():
    mname = MODEL_LABELS.get(slug, slug)
    for d in ["baseline", "censor_upper", "censor_lower"]:
        dd = mdf[mdf["design"] == d]
        if dd.empty:
            continue
        r = corr_r(dd)
        mean_j = dd[jcol(dd)].mean()
        print(f"    {mname:>14s} {DESIGN_LABELS.get(d,d):>14s}: r={r:+.3f}, mean_join={mean_j:.3f}, n={len(dd)}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Treatment effect summary (bar chart)
# ══════════════════════════════════════════════════════════════════════
print("\nFig 6: Treatment effect summary...")

# Compute mean join shift vs baseline at θ ≈ θ* (theta_relative near 0)
near_theta_star = llama[llama["theta_relative"].abs() < 0.10]
baseline_mean = near_theta_star[near_theta_star["design"] == "baseline"][jcol(llama)].mean()

all_designs_of_interest = [
    "stability", "instability", "stability_clarity", "stability_direction",
    "stability_dissent", "censor_upper", "censor_lower",
    "provenance_independent", "provenance_state", "provenance_social",
    "rhetoric_hot", "rhetoric_cold",
]

deltas = []
for d in all_designs_of_interest:
    dd = near_theta_star[near_theta_star["design"] == d]
    if dd.empty:
        continue
    d_mean = dd[jcol(dd)].mean()
    delta = d_mean - baseline_mean
    deltas.append((d, delta, d_mean))

deltas.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(COL_W, 3.5))
labels = [DESIGN_LABELS.get(d, d) for d, _, _ in deltas]
vals = [v for _, v, _ in deltas]
colors = [DESIGN_COLORS.get(d, "#999") for d, _, _ in deltas]
bars = ax.barh(range(len(deltas)), vals, color=colors, edgecolor="white", linewidth=0.3, height=0.7)
ax.set_yticks(range(len(deltas)))
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel("$\\Delta$ join rate vs baseline (at $\\theta \\approx \\theta^*$)")
ax.axvline(0, color="#999", linewidth=0.5)
ax.set_title("Treatment effects near threshold (Llama 70B)", fontsize=8)
for i, (d, v, m) in enumerate(deltas):
    ax.text(v + 0.005 * np.sign(v), i, f"{v:+.3f}", va="center", fontsize=5.5,
            ha="left" if v >= 0 else "right")
plt.tight_layout()
save(fig, "treatment_effects_bar")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Full panel — all new designs on Llama 70B
# ══════════════════════════════════════════════════════════════════════
print("\nFig 7: All designs panel...")
fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 5.0), sharey=True)

# Panel A: Core designs
ax = axes[0, 0]
for d in ["baseline", "stability", "instability"]:
    plot_design(ax, llama, d)
ax.set_title("Core designs")
ax.legend(fontsize=5, loc="upper right")
ax.set_ylabel("Join fraction")

# Panel B: Provenance
ax = axes[0, 1]
for d in ["baseline", "provenance_independent", "provenance_state", "provenance_social"]:
    plot_design(ax, llama, d)
ax.set_title("Provenance")
ax.legend(fontsize=5, loc="upper right")

# Panel C: Rhetoric
ax = axes[0, 2]
for d in ["baseline", "rhetoric_hot", "rhetoric_cold"]:
    plot_design(ax, llama, d)
ax.set_title("Rhetoric")
ax.legend(fontsize=5, loc="upper right")

# Panel D: Decomposition
ax = axes[1, 0]
for d in ["baseline", "stability_clarity", "stability_direction", "stability_dissent"]:
    plot_design(ax, llama, d)
ax.set_title("Stability decomposition")
ax.legend(fontsize=5, loc="upper right")
ax.set_ylabel("Join fraction")
ax.set_xlabel("$\\theta - \\theta^*$")

# Panel E: Censorship
ax = axes[1, 1]
for d in ["baseline", "censor_upper", "censor_lower"]:
    plot_design(ax, llama, d)
ax.set_title("Censorship")
ax.legend(fontsize=5, loc="upper right")
ax.set_xlabel("$\\theta - \\theta^*$")

# Panel F: Falsification
ax = axes[1, 2]
for d in ["baseline", "scramble", "flip"]:
    plot_design(ax, llama, d)
ax.set_title("Falsification")
ax.legend(fontsize=5, loc="upper right")
ax.set_xlabel("$\\theta - \\theta^*$")

for ax in axes.flat:
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.5, zorder=0)
    ax.axvline(0.0, color="#ccc", linestyle=":", linewidth=0.5, zorder=0)

fig.suptitle("Llama 70B: All information design treatments", fontsize=10, y=1.01)
plt.tight_layout()
save(fig, "all_designs_panel")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Cross-model baseline comparison
# ══════════════════════════════════════════════════════════════════════
print("\nFig 8: Cross-model baselines...")
fig, ax = plt.subplots(figsize=(COL_W, 2.8))
for slug, mdf in [("meta-llama--llama-3.3-70b-instruct", llama),
                   ("mistralai--mistral-small-creative", mistral),
                   ("qwen--qwen3-30b-a3b-instruct-2507", qwen),
                   ("allenai--olmo-3-7b-instruct", olmo),
                   ("mistralai--ministral-3b-2512", ministral)]:
    dd = mdf[mdf["design"] == "baseline"]
    if dd.empty:
        continue
    color = MODEL_COLORS.get(slug, "#999")
    lbl = MODEL_LABELS.get(slug, slug)
    c, m, se = binned(dd)
    ax.scatter(c, m, color=color, s=10, alpha=0.8, edgecolors="none", label=lbl, zorder=3)
    b0, b1 = fit_logistic(dd)
    xg = np.linspace(-0.35, 0.35, 200)
    ax.plot(xg, logistic(xg, b0, b1), color=color, linewidth=1.0, zorder=2, alpha=0.7)
ax.set_xlabel("$\\theta - \\theta^*$")
ax.set_ylabel("Join fraction")
ax.set_title("Baseline sigmoid by model")
ax.legend(fontsize=5.5, loc="upper right")
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.5, zorder=0)
ax.axvline(0.0, color="#ccc", linestyle=":", linewidth=0.5, zorder=0)
plt.tight_layout()
save(fig, "cross_model_baseline")

print("  Baseline r-values by model:")
for slug, mdf in [("meta-llama--llama-3.3-70b-instruct", llama),
                   ("mistralai--mistral-small-creative", mistral),
                   ("qwen--qwen3-30b-a3b-instruct-2507", qwen),
                   ("allenai--olmo-3-7b-instruct", olmo),
                   ("mistralai--ministral-3b-2512", ministral)]:
    dd = mdf[mdf["design"] == "baseline"]
    if dd.empty:
        continue
    r = corr_r(dd)
    mean_j = dd[jcol(dd)].mean()
    print(f"    {MODEL_LABELS.get(slug, slug):>18s}: r={r:+.3f}, mean_join={mean_j:.3f}, n={len(dd)}")


# ══════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY TABLE: All designs × Llama 70B")
print("="*70)
print(f"{'Design':>25s}  {'r':>7s}  {'mean_join':>9s}  {'n':>5s}  {'Δ_join@θ*':>10s}")
print("-"*70)
for d in sorted(llama["design"].unique()):
    dd = llama[llama["design"] == d]
    r = corr_r(dd)
    mean_j = dd[jcol(dd)].mean()
    # Delta at theta_star
    near = dd[dd["theta_relative"].abs() < 0.10]
    delta = near[jcol(near)].mean() - baseline_mean if not near.empty else float("nan")
    lbl = DESIGN_LABELS.get(d, d)
    print(f"{lbl:>25s}  {r:+7.3f}  {mean_j:9.3f}  {len(dd):5d}  {delta:+10.3f}")

print("\n" + "="*70)
print("CROSS-MODEL BASELINE r-values")
print("="*70)
for slug, mdf in [("meta-llama--llama-3.3-70b-instruct", llama),
                   ("mistralai--mistral-small-creative", mistral),
                   ("qwen--qwen3-30b-a3b-instruct-2507", qwen),
                   ("allenai--olmo-3-7b-instruct", olmo),
                   ("mistralai--ministral-3b-2512", ministral)]:
    dd = mdf[mdf["design"] == "baseline"]
    if dd.empty:
        continue
    r = corr_r(dd)
    print(f"  {MODEL_LABELS.get(slug, slug):>18s}: r = {r:+.3f} (n={len(dd)})")

print(f"\nFigures saved to: {FIG_DIR.resolve()}")
print("Done.")
