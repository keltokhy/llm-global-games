"""The regime's problem with measured inputs: S(monitor) vs S(silent).

Implements equation (1) of the paper with every input measured:
  S(m; h) = 1 - mean_cells[ coup(m, cell) * (1 - h * qhat(m, cell)) ]
where coup(m, cell) is the realized uprising outcome in arm m (the citizen
side: chilling lowers coup rates in the surveillance arm on the same theta),
qhat(m, cell) is the regime's posterior that the uprising succeeds, proxied
by the mean FALL probability of the four common analyst models reading that
cell's intercepted traffic (the regime side: blinding flattens qhat in
crisis states under monitoring), and h in [0,1] is response capacity: the
probability that a fully anticipated uprising is defeated by preemption/
concession. h * qhat is the linear response rule.

Coverage: all 498 usable cells of the nested grid (150 main-sample + 348
held-out), both arms, same four-analyst roster throughout.

Outputs:
  Delta S(h) = S(1; h) - S(0; h): positive at h=0 (pure chilling), declining
  in h as the blinding cost binds; the crossing h* is the capacity level
  above which visible surveillance REDUCES regime survival. Repeated under
  crisis-exposure reweighting (mass on theta<0 scaled up) to trace the
  monitor-vs-silent frontier in (h, crisis exposure) space.

Usage: uv run python analysis/survival_exhibit.py
Writes: analysis/survival_exhibit.json, paper2/figures/fig_survival.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import C_PURE, C_SURV, apply_style  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT = PROJECT_ROOT / "output" / "analyst-pilot"
OUT_JSON = PROJECT_ROOT / "analysis" / "survival_exhibit.json"
OUT_FIG = PROJECT_ROOT / "paper2" / "figures" / "fig_survival.pdf"

ANALYSTS = [
    "deepseek--deepseek-v4-flash-20260423",
    "meta-llama--llama-4-maverick-17b-128e-instruct",
    "qwen--qwen3.7-plus",
    "meta-llama--llama-3.3-70b-instruct",
]
SAMPLES = ["nested", "nested-holdout"]


def load_cells() -> pd.DataFrame:
    """One row per cell: theta, coup and pooled-analyst qhat for each arm."""
    frames = []
    for slug in ANALYSTS:
        for label in SAMPLES:
            for arm in ("baseline", "surveillance"):
                p = PILOT / slug / f"experiment_analyst_{label}_{arm}_summary.csv"
                df = pd.read_csv(p)[["country", "period", "theta", "coup_success", "fall_est"]]
                df["arm"], df["analyst"] = arm, slug
                frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    pooled = (df.groupby(["country", "period", "arm"])
                .agg(theta=("theta", "first"),
                     coup=("coup_success", "first"),
                     qhat=("fall_est", lambda s: s.mean() / 100.0),
                     n_analysts=("fall_est", "size"))
                .reset_index())
    pooled = pooled[pooled.n_analysts == len(ANALYSTS)]
    wide = pooled.pivot_table(index=["country", "period", "theta"],
                              columns="arm", values=["coup", "qhat"]).reset_index()
    wide.columns = ["country", "period", "theta",
                    "coup_base", "coup_surv", "qhat_base", "qhat_surv"]
    return wide.dropna()


def delta_S(cells: pd.DataFrame, h: float, weights: np.ndarray) -> float:
    s1 = 1 - np.average(cells.coup_surv * (1 - np.minimum(1.0, h * cells.qhat_surv)),
                        weights=weights)
    s0 = 1 - np.average(cells.coup_base * (1 - np.minimum(1.0, h * cells.qhat_base)),
                        weights=weights)
    return float(s1 - s0)


def crisis_weights(cells: pd.DataFrame, w_crisis: float) -> np.ndarray:
    """Reweight crisis cells (theta<0) by factor w_crisis (1 = empirical grid)."""
    w = np.where(cells.theta < 0, w_crisis, 1.0)
    return w / w.sum()


def main() -> None:
    cells = load_cells()
    print(f"[survival] {len(cells)} cells with both arms x {len(ANALYSTS)} analysts")
    print(f"[survival] coup rates: baseline {cells.coup_base.mean():.3f} "
          f"surveillance {cells.coup_surv.mean():.3f}")
    crisis = cells.theta < 0
    print(f"[survival] qhat in crisis cells where coup occurs: "
          f"baseline {cells[crisis & (cells.coup_base==1)].qhat_base.mean():.3f}  "
          f"surveillance {cells[crisis & (cells.coup_surv==1)].qhat_surv.mean():.3f}")

    hs = np.linspace(0, 1, 101)
    results = {"n_cells": int(len(cells)), "h_grid": hs.tolist(), "curves": {}}
    crossings = {}
    for w_crisis in (1.0, 2.0, 4.0):
        w = crisis_weights(cells, w_crisis)
        curve = [delta_S(cells, h, w) for h in hs]
        results["curves"][str(w_crisis)] = curve
        sign_change = np.where(np.diff(np.sign(curve)) < 0)[0]
        crossings[w_crisis] = float(hs[sign_change[0] + 1]) if len(sign_change) else None
    results["h_star"] = {str(k): v for k, v in crossings.items()}
    OUT_JSON.write_text(json.dumps(results, indent=1))

    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    styles = {"1.0": ("-", C_SURV, "empirical grid"),
              "2.0": ("--", "#9b59b6", r"crisis mass $\times 2$"),
              "4.0": (":", "#c39bd3", r"crisis mass $\times 4$")}
    for k, curve in results["curves"].items():
        ls, color, lab = styles[k]
        ax.plot(hs, np.array(curve) * 100, ls, color=color, lw=1.5, label=lab)
        hstar = crossings[float(k)]
        if hstar is not None:
            ax.plot([hstar], [0], "o", color=color, ms=4)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel(r"response capacity $h$ (P(defeat a fully anticipated uprising))",
                  fontsize=8)
    ax.set_ylabel(r"$S(\mathrm{monitor}) - S(\mathrm{silent})$, pp", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Visible surveillance helps weak responders\nand hurts capable ones",
                 fontsize=9)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    print(f"[survival] h* (crossing): {crossings}")
    print(f"[survival] -> {OUT_JSON}\n[survival] -> {OUT_FIG}")


if __name__ == "__main__":
    main()
