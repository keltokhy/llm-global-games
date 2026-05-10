"""Shared publication style for all paper visualizations.

The paper's figures should read as one visual system: neutral empirical
baseline, blue information/communication treatments, red/orange destructive
falsifications, purple surveillance, and green stability/intervention designs.
This module is the single source of truth for rcParams, colors, dimensions,
and small drawing helpers used by every figure script.
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
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Two-column layout dimensions (inches) ─────────────────────────
COL_W = 3.4    # \columnwidth — single-column figure
TEXT_W = 7.0   # \textwidth — figure* spanning both columns


# ── rcParams (sized for 1:1 rendering in two-column layout) ──────
RCPARAMS = {
    "font.family":          "serif",
    "font.size":            8,
    "axes.titlesize":       8.5,
    "axes.labelsize":       8,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      6.5,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.55,
    "axes.edgecolor":       "#333333",
    "axes.labelcolor":      "#222222",
    "axes.titleweight":     "bold",
    "axes.grid":            False,
    "xtick.major.width":    0.55,
    "ytick.major.width":    0.55,
    "xtick.major.size":     2.8,
    "ytick.major.size":     2.8,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "legend.frameon":       False,
    "legend.handlelength":  1.5,
    "legend.handletextpad": 0.35,
    "legend.columnspacing": 0.8,
    "lines.linewidth":      1.15,
    "lines.markersize":     3.7,
    "figure.dpi":           150,
    "figure.facecolor":     "white",
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
}


def apply_style():
    """Apply the paper's matplotlib style. Call once at module import."""
    matplotlib.use("Agg")
    plt.rcParams.update(RCPARAMS)


# ── Treatment colors ──────────────────────────────────────────────
# Okabe-Ito/Tol-inspired palette with stable semantics across figures.
C_INK      = "#222222"
C_MUTED    = "#6f6f6f"
C_GRID     = "#d8d8d8"
C_LIGHT    = "#efefef"
C_PURE     = "#5f6368"  # neutral baseline
C_COMM     = "#0072B2"  # information / communication
C_FLIP     = "#D55E00"  # destructive falsification / reversal
C_SCRAMBLE = "#E69F00"  # signal destruction / placebo-ish warning
C_NET      = "#009E73"  # network / robustness
C_SURV     = "#7B3294"  # surveillance / monitoring
C_THEORY   = "#111111"

# Information design colors
C_BASELINE    = C_PURE
C_STABILITY   = "#009E73"
C_INSTABILITY = "#D55E00"
C_CENS_UP     = "#0072B2"
C_CENS_LO     = "#E69F00"
C_PUBLIC      = "#7B3294"

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
    "stability_clarity":    "#E69F00",
    "stability_direction":  "#009E73",
    "stability_dissent":    "#7B3294",
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
    for ax in fig.axes:
        polish_axes(ax)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def polish_axes(ax):
    """Apply final axis polish without changing data content."""
    ax.tick_params(colors=C_INK, labelcolor=C_INK)
    ax.xaxis.label.set_color(C_INK)
    ax.yaxis.label.set_color(C_INK)
    ax.title.set_color(C_INK)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def format_rate_axis(ax, y=True):
    """Standard [0,1] rate axis."""
    target = ax.yaxis if y else ax.xaxis
    target.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def shade_ci(ax, x, mean, se, color, alpha=0.11, zorder=1):
    """Draw a 95% CI ribbon where standard errors are available."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    se = np.asarray(se, dtype=float)
    if len(x) == 0 or np.all(~np.isfinite(se)):
        return
    ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                    color=color, alpha=alpha, linewidth=0, zorder=zorder)


def plot_curve_points(ax, x, mean, se=None, color=C_PURE, label=None,
                      marker="o", linestyle="-", linewidth=1.15, markersize=13,
                      alpha=0.95, ribbon=True, zorder=3):
    """Common line + point + optional confidence ribbon grammar."""
    if se is not None and ribbon:
        shade_ci(ax, x, mean, se, color)
    ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=linewidth,
            alpha=alpha, label=label, zorder=zorder - 1)
    if marker in {"x", "+", "1", "2", "3", "4", "|", "_"}:
        ax.scatter(x, mean, color=color, marker=marker, s=markersize,
                   alpha=alpha, linewidths=0.7, zorder=zorder)
    else:
        ax.scatter(x, mean, color=color, marker=marker, s=markersize,
                   alpha=alpha, edgecolors="white", linewidths=0.25, zorder=zorder)


def zero_line(ax, axis="y", color="#333333"):
    """Consistent zero-reference line."""
    if axis == "y":
        ax.axhline(0, color=color, linewidth=0.6, zorder=1)
    else:
        ax.axvline(0, color=color, linewidth=0.6, zorder=1)


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
