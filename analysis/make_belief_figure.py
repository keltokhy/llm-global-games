"""Generate belief elicitation figure for the paper.

Two-panel figure (figure*):
  (a) Stated belief vs Bayesian posterior — showing beliefs track strategic prediction
  (b) Join rate by belief bin, Pure vs Surveillance — showing preference falsification

Uses the same style as make_figures.py.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP = PROJECT_ROOT / "output" / "mistralai--mistral-small-creative" / "_overwrite_200period_backup"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style (match make_figures.py) ─────────────────────────────────
TEXT_W = 7.0
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

C_PURE = "#636363"
C_SURV = "#7b3294"


def load_agents(log_path):
    """Extract flat agent-level records from a log file."""
    with open(log_path) as f:
        periods = json.load(f)
    rows = []
    sigma = 0.3
    for p in periods:
        theta = p["theta"]
        theta_star = p["theta_star"]
        x_star = theta_star + sigma * stats.norm.ppf(theta_star)
        for a in p["agents"]:
            if a.get("belief") is None or a.get("api_error"):
                continue
            signal = a["signal"]
            belief = a["belief"] / 100.0
            decision = 1 if a["decision"] == "JOIN" else 0
            # P(success | x_i) = P(theta < theta* | x_i) = Phi((theta* - x_i) / sigma)
            posterior = stats.norm.cdf((theta_star - signal) / sigma)
            rows.append({
                "belief": belief,
                "decision": decision,
                "posterior": posterior,
                "signal": signal,
                "theta": theta,
                "theta_star": theta_star,
            })
    return rows


def bin_data(x, y, edges):
    """Bin y by x using given bin edges. Return centers, means, SEs, counts."""
    centers, means, ses, counts = [], [], [], []
    for i in range(len(edges) - 1):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        n = mask.sum()
        if n < 5:
            continue
        m = y[mask].mean()
        se = y[mask].std() / np.sqrt(n)
        centers.append((edges[i] + edges[i + 1]) / 2)
        means.append(m)
        ses.append(se)
        counts.append(n)
    return np.array(centers), np.array(means), np.array(ses), np.array(counts)


def main():
    pure = load_agents(BACKUP / "experiment_pure_beliefs_log.json")
    surv = load_agents(BACKUP / "experiment_surveillance_beliefs_log.json")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_W, 2.6))

    # ── Panel (a): Stated belief vs Bayesian posterior ────────────
    posteriors = np.array([r["posterior"] for r in pure])
    beliefs = np.array([r["belief"] for r in pure])

    # Bin posteriors into 20 equal-width bins
    edges = np.linspace(0, 1, 21)
    bc, bm, bse, bn = bin_data(posteriors, beliefs, edges)

    ax_a.plot([0, 1], [0, 1], color="#cccccc", linewidth=0.8, linestyle="--",
              zorder=1, label="Perfect calibration")
    ax_a.errorbar(bc, bm, yerr=1.96 * bse, fmt="o", color=C_PURE,
                  markersize=4, elinewidth=0.6, capsize=0, zorder=3)

    # OLS fit
    slope, intercept, r_val, _, _ = stats.linregress(posteriors, beliefs)
    x_fit = np.linspace(0, 1, 100)
    ax_a.plot(x_fit, intercept + slope * x_fit, color=C_PURE, linewidth=1.0,
              linestyle="-", zorder=2)

    ax_a.set_xlabel(r"Bayesian posterior $P(\mathrm{success} \mid x_i)$")
    ax_a.set_ylabel("Stated belief")
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_aspect("equal")
    ax_a.text(0.05, 0.92, f"$r = {r_val:+.2f}$\nslope $= {slope:.2f}$",
              transform=ax_a.transAxes, fontsize=7, va="top",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        alpha=0.8, edgecolor="#ccc", linewidth=0.4))
    ax_a.set_title("(a) Beliefs track Bayesian posterior", fontsize=8, loc="left")

    # ── Panel (b): Join rate by belief bin — Pure vs Surveillance ─
    pure_beliefs = np.array([r["belief"] for r in pure])
    pure_decisions = np.array([r["decision"] for r in pure])
    surv_beliefs = np.array([r["belief"] for r in surv])
    surv_decisions = np.array([r["decision"] for r in surv])

    bin_edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.01])
    bin_labels = ["0\u201320", "20\u201340", "40\u201360", "60\u201380", "80\u2013100"]
    pc, pm, pse, pn = bin_data(pure_beliefs, pure_decisions, bin_edges)
    sc, sm, sse, sn = bin_data(surv_beliefs, surv_decisions, bin_edges)

    x_pos = np.arange(len(pc))
    bar_w = 0.35
    ax_b.bar(x_pos - bar_w/2, pm, width=bar_w, color=C_PURE, alpha=0.85,
             label="Pure", zorder=3, edgecolor="white", linewidth=0.3)
    ax_b.bar(x_pos + bar_w/2, sm, width=bar_w, color=C_SURV, alpha=0.85,
             label="Surveillance", zorder=3, edgecolor="white", linewidth=0.3)

    # Error bars
    ax_b.errorbar(x_pos - bar_w/2, pm, yerr=1.96 * pse, fmt="none",
                  ecolor=C_PURE, elinewidth=0.6, capsize=2, zorder=4)
    ax_b.errorbar(x_pos + bar_w/2, sm, yerr=1.96 * sse, fmt="none",
                  ecolor=C_SURV, elinewidth=0.6, capsize=2, zorder=4)

    # Annotate the preference falsification gap at the 60-80% bin
    gap_idx = 3  # 60-80% bin
    if gap_idx < len(pm) and gap_idx < len(sm):
        gap_x = x_pos[gap_idx] + bar_w/2 + 0.15
        ax_b.annotate("", xy=(gap_x, sm[gap_idx] + 0.02),
                     xytext=(gap_x, pm[gap_idx] - 0.02),
                     arrowprops=dict(arrowstyle="<->", color="#333",
                                    linewidth=0.8))
        gap_pp = (pm[gap_idx] - sm[gap_idx]) * 100
        ax_b.text(gap_x + 0.08, (pm[gap_idx] + sm[gap_idx]) / 2,
                  f"{gap_pp:.0f} pp", fontsize=6.5, va="center")

    # Count annotations
    for i in range(len(pc)):
        y_max = max(pm[i], sm[i])
        ax_b.text(x_pos[i], -0.08, f"$n$={int(pn[i]+sn[i]):,}",
                  ha="center", va="top", fontsize=5.5, color="#999")

    ax_b.set_xlabel("Stated belief $P(\\mathrm{success})$, %")
    ax_b.set_ylabel("Join rate")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(bin_labels)
    ax_b.set_ylim(-0.12, 1.08)
    ax_b.legend(loc="upper left", framealpha=0.9, edgecolor="#ccc")
    ax_b.set_title("(b) Preference falsification under surveillance", fontsize=8, loc="left")

    plt.tight_layout()
    out = FIG_DIR / "fig16_beliefs.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    # Also save PNG for quick preview
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
