"""Generate belief elicitation figure for the paper.

Two-panel figure (figure*):
  (a) Stated belief vs Bayesian posterior — showing beliefs track strategic prediction
  (b) Join rate by belief bin, Pure vs Comm vs Surveillance — showing treatment effects

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
COMM_DIR = PROJECT_ROOT / "output" / "mistralai--mistral-small-creative" / "_beliefs_comm" / "mistralai--mistral-small-creative"
PROP_DIR = PROJECT_ROOT / "output" / "mistralai--mistral-small-creative" / "_beliefs_propaganda_k5" / "mistralai--mistral-small-creative"
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
C_COMM = "#2166ac"
C_SURV = "#7b3294"
C_PROP = "#d6604d"


def load_agents(log_path):
    """Extract flat agent-level records from a log file."""
    with open(log_path) as f:
        periods = json.load(f)
    rows = []
    sigma = 0.3
    for p in periods:
        theta = p["theta"]
        theta_star = p["theta_star"]
        for a in p["agents"]:
            if a.get("belief") is None or a.get("api_error"):
                continue
            signal = a["signal"]
            belief = a["belief"] / 100.0
            decision = 1 if a["decision"] == "JOIN" else 0
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
    comm = load_agents(COMM_DIR / "experiment_comm_log.json")
    surv = load_agents(BACKUP / "experiment_surveillance_beliefs_log.json")

    # Propaganda k=5 (optional — may not exist yet)
    prop_path = PROP_DIR / "experiment_comm_log.json"
    prop = load_agents(prop_path) if prop_path.exists() else []
    has_prop = len(prop) > 50

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_W, 2.6))

    # ── Panel (a): Stated belief vs Bayesian posterior ────────────
    posteriors = np.array([r["posterior"] for r in pure])
    beliefs = np.array([r["belief"] for r in pure])

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

    # ── Panel (b): Join rate by belief bin — treatments ────────────
    bin_edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.01])
    bin_labels = ["0\u201320", "20\u201340", "40\u201360", "60\u201380", "80\u2013100"]

    pure_beliefs = np.array([r["belief"] for r in pure])
    pure_decisions = np.array([r["decision"] for r in pure])
    comm_beliefs = np.array([r["belief"] for r in comm])
    comm_decisions = np.array([r["decision"] for r in comm])
    surv_beliefs = np.array([r["belief"] for r in surv])
    surv_decisions = np.array([r["decision"] for r in surv])

    pc, pm, pse, pn = bin_data(pure_beliefs, pure_decisions, bin_edges)
    cc, cm, cse, cn = bin_data(comm_beliefs, comm_decisions, bin_edges)
    sc, sm, sse, sn = bin_data(surv_beliefs, surv_decisions, bin_edges)

    n_groups = 4 if has_prop else 3
    bar_w = 0.8 / n_groups
    x_pos = np.arange(len(pc))

    offsets = np.linspace(-0.4 + bar_w / 2, 0.4 - bar_w / 2, n_groups)
    ax_b.bar(x_pos + offsets[0], pm, width=bar_w, color=C_PURE, alpha=0.85,
             label="Pure", zorder=3, edgecolor="white", linewidth=0.3)
    ax_b.bar(x_pos + offsets[1], cm, width=bar_w, color=C_COMM, alpha=0.85,
             label="Communication", zorder=3, edgecolor="white", linewidth=0.3)
    ax_b.bar(x_pos + offsets[2], sm, width=bar_w, color=C_SURV, alpha=0.85,
             label="Surveillance", zorder=3, edgecolor="white", linewidth=0.3)

    # Error bars
    ax_b.errorbar(x_pos + offsets[0], pm, yerr=1.96 * pse, fmt="none",
                  ecolor=C_PURE, elinewidth=0.6, capsize=2, zorder=4)
    ax_b.errorbar(x_pos + offsets[1], cm, yerr=1.96 * cse, fmt="none",
                  ecolor=C_COMM, elinewidth=0.6, capsize=2, zorder=4)
    ax_b.errorbar(x_pos + offsets[2], sm, yerr=1.96 * sse, fmt="none",
                  ecolor=C_SURV, elinewidth=0.6, capsize=2, zorder=4)

    if has_prop:
        prop_beliefs = np.array([r["belief"] for r in prop])
        prop_decisions = np.array([r["decision"] for r in prop])
        rc, rm, rse, rn = bin_data(prop_beliefs, prop_decisions, bin_edges)
        ax_b.bar(x_pos + offsets[3], rm, width=bar_w, color=C_PROP, alpha=0.85,
                 label="Propaganda $k{=}5$", zorder=3, edgecolor="white", linewidth=0.3)
        ax_b.errorbar(x_pos + offsets[3], rm, yerr=1.96 * rse, fmt="none",
                      ecolor=C_PROP, elinewidth=0.6, capsize=2, zorder=4)

    ax_b.set_xlabel("Stated belief (percent)")
    ax_b.set_ylabel("Join rate")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(bin_labels)
    ax_b.set_ylim(-0.05, 1.08)
    ax_b.legend(loc="upper left", framealpha=0.9, edgecolor="#ccc")
    ax_b.set_title("(b) Actions diverge from beliefs under treatment", fontsize=8, loc="left")

    plt.tight_layout()
    out = FIG_DIR / "fig16_beliefs.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
