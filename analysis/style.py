"""
Shared figure style for all paper visualizations.

Single source of truth for matplotlib rcParams, color palettes,
layout dimensions, and common helpers used across make_figures.py,
construct_validity.py, make_diagrams.py, and any future figure scripts.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ── Two-column layout dimensions (inches) ─────────────────────────
COL_W = 3.4    # \columnwidth — single-column figure
TEXT_W = 7.0   # \textwidth — figure* spanning both columns


# ── rcParams (sized for 1:1 rendering in two-column layout) ──────
RCPARAMS = {
    "font.family":          "serif",
    "font.size":            8,
    "axes.titlesize":       9,
    "axes.labelsize":       8,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.6,
    "axes.grid":            False,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "legend.frameon":       False,
    "legend.handlelength":  1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "lines.linewidth":      1.0,
    "lines.markersize":     4,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
}


def apply_style():
    """Apply the paper's matplotlib style. Call once at module import."""
    matplotlib.use("Agg")
    plt.rcParams.update(RCPARAMS)


# ── Treatment colors ──────────────────────────────────────────────
C_PURE     = "#636363"
C_COMM     = "#2c7bb6"
C_FLIP     = "#d7191c"
C_SCRAMBLE = "#fdae61"
C_NET      = "#1a9641"
C_SURV     = "#7b3294"
C_PROP     = "#CC79A7"

# Information design colors
C_BASELINE    = "#636363"
C_STABILITY   = "#2c7bb6"
C_INSTABILITY = "#d7191c"
C_CENS_UP     = "#1a9641"
C_CENS_LO     = "#e66101"
C_PUBLIC      = "#7b3294"

# Construct validity colors
C_1FEAT = "#fdae61"
C_3FEAT = "#1a9641"

DESIGN_COLORS = {
    "baseline":             C_BASELINE,
    "stability":            C_STABILITY,
    "instability":          C_INSTABILITY,
    "censor_upper":         C_CENS_UP,
    "censor_lower":         C_CENS_LO,
    "public_signal":        C_PUBLIC,
    "scramble":             C_SCRAMBLE,
    "flip":                 C_FLIP,
    # Decomposition channels — distinct hues, not all-blue
    "stability_clarity":    "#e66101",   # orange
    "stability_direction":  "#1a9641",   # green
    "stability_dissent":    "#7b3294",   # purple
}

DESIGN_LABELS = {
    "baseline":             "Baseline",
    "stability":            "Stability",
    "instability":          "Instability",
    "censor_upper":         "Censor upper",
    "censor_lower":         "Censor lower",
    "public_signal":        "Public signal",
    "scramble":             "Scramble",
    "flip":                 "Flip",
    "stability_clarity":    "Clarity only",
    "stability_direction":  "Direction only",
    "stability_dissent":    "Dissent only",
}

DESIGN_MARKERS = {
    "baseline":             "o",
    "stability":            "s",
    "instability":          "D",
    "censor_upper":         "^",
    "censor_lower":         "v",
    "public_signal":        "P",
    "scramble":             "x",
    "flip":                 "+",
    "stability_clarity":    "<",
    "stability_direction":  ">",
    "stability_dissent":    "d",
}


# ── Helpers ───────────────────────────────────────────────────────

def join_col(df):
    """Prefer join_fraction_valid when available, fall back to join_fraction."""
    if "join_fraction_valid" in df.columns and df["join_fraction_valid"].notna().any():
        return "join_fraction_valid"
    return "join_fraction"


def logistic(x, b0, b1):
    """Standard logistic function: 1 / (1 + exp(b0 + b1*x))."""
    return 1.0 / (1.0 + np.exp(b0 + b1 * x))


def fit_logistic(df, theta_col="theta", jcol=None, join_col_name=None):
    """Fit a 2-parameter logistic to (theta, join_fraction).

    Returns (popt, pcov). On failure, returns (np.array([0, 0]), np.zeros((2,2))).
    """
    jcol = jcol or join_col_name or join_col(df)
    d = df.dropna(subset=[theta_col, jcol])
    x, y = d[theta_col].values, d[jcol].values
    try:
        popt, pcov = curve_fit(logistic, x, y, p0=[0.0, 2.0], maxfev=10000)
        return popt, pcov
    except (RuntimeError, ValueError):
        return np.array([0.0, 0.0]), np.zeros((2, 2))


def fitted_cutoff(popt):
    """Logistic midpoint: -b0/b1."""
    if popt is None:
        return float("nan")
    return -popt[0] / popt[1]


def attack_mass(theta, theta_star=0.50, sigma=0.30):
    """Theoretical attack mass A(theta) from Morris-Shin."""
    theta = np.asarray(theta, dtype=float)
    ts = float(np.clip(theta_star, 1e-8, 1 - 1e-8))
    x_star = ts + sigma * stats.norm.ppf(ts)
    return stats.norm.cdf((x_star - theta) / sigma)


def save(fig, name, fig_dir=None):
    """Save figure as both PDF and PNG."""
    fig_dir = fig_dir or FIG_DIR
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def add_hgrid(ax, alpha=0.3, linewidth=0.3):
    """Add subtle horizontal gridlines to bar/dot charts."""
    ax.yaxis.grid(True, linewidth=linewidth, alpha=alpha, color="#cccccc")
    ax.set_axisbelow(True)


def add_vgrid(ax, alpha=0.3, linewidth=0.3):
    """Add subtle vertical gridlines to horizontal bar charts."""
    ax.xaxis.grid(True, linewidth=linewidth, alpha=alpha, color="#cccccc")
    ax.set_axisbelow(True)


def panel_label(ax, label, x=-0.12, y=1.05):
    """Add a panel label (A., B., etc.) to a subplot."""
    ax.text(x, y, f"{label}.", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")
