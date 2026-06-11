"""Publication figures for "The Intelligence Cost of Surveillance" (paper 2).

Figure 1  fig_crisis_blinding.pdf
  (a) Analyst Brier on regime survival by theta bin, baseline vs surveillance,
      pooled over the four preregistered held-out analysts (348 fresh cells):
      the gap opens only where the regime is weak.
  (b) Per-analyst crisis vs calm Brier deltas on the main 150-cell sample,
      ordered by analyst capability tier: the frontier model loses as much
      in crisis as cheap models.

Figure 2  fig_dose_response.pdf
  (a) Chilling curve: mean join fraction at four monitoring doses.
  (b) Blinding curve: crisis delta-Brier at three doses, per analyst.
  Both step-like: the mild cue does most of the work.

Figure 3  fig_lemma1_text.pdf
  Message-feature divergence between arms by sender signal and by theta:
  near-universal style shift with a monotone selection gradient.

Usage: uv run python analysis/paper2_figures.py
Writes PDFs to paper2/figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import C_COMM, C_PURE, C_SURV, apply_style  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT = PROJECT_ROOT / "output" / "analyst-pilot"
FIG_DIR = PROJECT_ROOT / "paper2" / "figures"

HOLDOUT_ANALYSTS = {
    "deepseek--deepseek-v4-flash-20260423": "DeepSeek v4",
    "meta-llama--llama-4-maverick-17b-128e-instruct": "Llama 4 Mav.",
    "qwen--qwen3.7-plus": "Qwen 3.7",
    "meta-llama--llama-3.3-70b-instruct": "Llama 3.3",
}
# Main-sample analysts ordered by capability tier (2025 -> 2026 open -> frontier)
TIERED_ANALYSTS = [
    ("meta-llama--llama-3.3-70b-instruct", "Llama 3.3\n(2025)"),
    ("deepseek--deepseek-v4-flash-20260423", "DeepSeek v4\nflash"),
    ("meta-llama--llama-4-maverick-17b-128e-instruct", "Llama 4\nMaverick"),
    ("qwen--qwen3.7-plus", "Qwen 3.7\nPlus"),
    ("z-ai--glm-5.1-20260406", "GLM 5.1"),
    ("anthropic--claude-opus-4.8", "Opus 4.8\n(frontier)"),
]
DOSE_ANALYSTS = {
    "deepseek--deepseek-v4-flash-20260423": "DeepSeek v4",
    "meta-llama--llama-4-maverick-17b-128e-instruct": "Llama 4 Mav.",
    "meta-llama--llama-3.3-70b-instruct": "Llama 3.3",
}


def load_pair(slug: str, label: str) -> pd.DataFrame:
    root = PILOT / slug
    b = pd.read_csv(root / f"experiment_analyst_{label}_baseline_summary.csv")
    s = pd.read_csv(root / f"experiment_analyst_{label}_surveillance_summary.csv")
    m = b.merge(s, on=["country", "period"], suffixes=("_b", "_s"))
    m["theta"] = m["theta_b"]
    for arm in ("b", "s"):
        m[f"brier_{arm}"] = (m[f"fall_est_{arm}"] / 100.0
                             - m[f"coup_success_{arm}"].astype(float)) ** 2
    return m


def paired_ci(d: np.ndarray) -> tuple[float, float]:
    d = d[~np.isnan(d)]
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(1.96 * se)


# ── Figure 1 ──────────────────────────────────────────────────────────


def fig_crisis_blinding() -> None:
    import matplotlib.pyplot as plt

    pooled = pd.concat(
        [load_pair(slug, "nested-holdout").assign(analyst=name)
         for slug, name in HOLDOUT_ANALYSTS.items()],
        ignore_index=True,
    )
    edges = np.quantile(pooled["theta"].unique(), np.linspace(0, 1, 9))
    edges[0], edges[-1] = pooled.theta.min() - 1e-9, pooled.theta.max() + 1e-9
    pooled["bin"] = pd.cut(pooled["theta"], edges, labels=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (a) Brier by theta bin, pooled holdout analysts
    ax = axes[0]
    for arm, color, label in (("b", C_PURE, "baseline messages"),
                              ("s", C_SURV, "surveilled messages")):
        means, los, his, centers = [], [], [], []
        for k, g in pooled.groupby("bin"):
            cells = g.groupby(["country", "period"])[f"brier_{arm}"].mean()
            means.append(cells.mean())
            ci = 1.96 * cells.std(ddof=1) / np.sqrt(len(cells))
            los.append(cells.mean() - ci)
            his.append(cells.mean() + ci)
            centers.append(g["theta"].median())
        ax.plot(centers, means, "o-", color=color, lw=1.4, ms=3.5, label=label)
        ax.fill_between(centers, los, his, color=color, alpha=0.18, lw=0)
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.text(0.04, 0.93, "regime weak $\\leftarrow$", transform=ax.transAxes,
            fontsize=7, color="#555555")
    ax.text(0.96, 0.93, "$\\rightarrow$ regime strong", transform=ax.transAxes,
            fontsize=7, color="#555555", ha="right")
    ax.set_xlabel(r"regime strength $\theta$ (bin medians)", fontsize=8)
    ax.set_ylabel("analyst Brier error\n(regime-survival judgment)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_title("(a) Held-out cells: the gap opens only in crisis", fontsize=8.5)

    # (b) Crisis vs calm deltas by analyst tier (main 150-cell sample)
    ax = axes[1]
    xs = np.arange(len(TIERED_ANALYSTS))
    for j, (slug, name) in enumerate(TIERED_ANALYSTS):
        m = load_pair(slug, "nested")
        d = (m["brier_b"] - m["brier_s"]).values
        for sel, color, marker, off in ((m.theta < 0, C_SURV, "o", -0.08),
                                        (m.theta >= 0, "#9aa0a6", "s", 0.08)):
            mean, ci = paired_ci(d[sel.values])
            ax.errorbar(j + off, mean, yerr=ci, fmt=marker, color=color,
                        ms=4, capsize=2, lw=1.1)
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in TIERED_ANALYSTS], fontsize=6.3)
    ax.set_ylabel(r"$\Delta$ Brier (baseline $-$ surveilled)", fontsize=8)
    ax.errorbar([], [], fmt="o", color=C_SURV, label=r"crisis cells ($\theta<0$)")
    ax.errorbar([], [], fmt="s", color="#9aa0a6", label=r"calm cells ($\theta\geq 0$)")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("(b) Capability does not protect the crisis margin", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_crisis_blinding.pdf")
    plt.close(fig)


# ── Figure 2 ──────────────────────────────────────────────────────────

DOSES = [
    ("none", "output/revision-nested-comm/meta-llama--llama-3.3-70b-instruct/experiment_comm_summary.csv"),
    ("mild", "output/revision-nested-surv-mild/meta-llama--llama-3.3-70b-instruct/experiment_comm_summary.csv"),
    ("full", "output/revision-nested-surv/meta-llama--llama-3.3-70b-instruct/experiment_comm_summary.csv"),
    ("severe", "output/revision-nested-surv-severe/meta-llama--llama-3.3-70b-instruct/experiment_comm_summary.csv"),
]
DOSE_LABELS = {"none": "nested", "mild": "nested-mild", "full": "nested", "severe": "nested-severe"}


def fig_dose_response() -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (a) chilling
    ax = axes[0]
    xs, means, cis = [], [], []
    for k, (dose, path) in enumerate(DOSES):
        df = pd.read_csv(PROJECT_ROOT / path)
        jf = df["join_fraction_valid"].values
        xs.append(k)
        means.append(jf.mean())
        cis.append(1.96 * jf.std(ddof=1) / np.sqrt(len(jf)))
    ax.errorbar(xs, means, yerr=cis, fmt="o-", color=C_COMM, lw=1.4, ms=4, capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([d for d, _ in DOSES], fontsize=8)
    ax.set_xlabel("monitoring salience (warning dose)", fontsize=8)
    ax.set_ylabel("mean join fraction\n(500 matched cells)", fontsize=8)
    ax.set_title("(a) Chilling: most of it at the mildest cue", fontsize=8.5)
    ax.annotate("no consequence\nlanguage", (1, means[1]), textcoords="offset points",
                xytext=(8, 12), fontsize=6.5, color="#555555")

    # (b) blinding: crisis delta-Brier by dose, per analyst
    ax = axes[1]
    dose_order = ["mild", "full", "severe"]
    for slug, name in DOSE_ANALYSTS.items():
        ys, errs = [], []
        for dose in dose_order:
            m = load_pair(slug, DOSE_LABELS[dose] if dose != "full" else "nested")
            d = (m["brier_b"] - m["brier_s"])[m.theta < 0].values
            mean, ci = paired_ci(d)
            ys.append(mean)
            errs.append(ci)
        ax.errorbar(range(len(dose_order)), ys, yerr=errs, fmt="o-", lw=1.2, ms=3.5,
                    capsize=2, label=name)
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    ax.set_xticks(range(len(dose_order)))
    ax.set_xticklabels(dose_order, fontsize=8)
    ax.set_xlabel("monitoring salience (warning dose)", fontsize=8)
    ax.set_ylabel(r"crisis $\Delta$ Brier" + "\n(baseline $-$ dose arm)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("(b) Blinding: already near-maximal at mild", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dose_response.pdf")
    plt.close(fig)


# ── Figure 3 ──────────────────────────────────────────────────────────


def fig_lemma1() -> None:
    import matplotlib.pyplot as plt

    res = json.loads((PROJECT_ROOT / "analysis" / "lemma1_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, key, xlabel, title, mono in (
        (axes[0], "by_sender_z", "sender signal $z$ (anti-regime $\\to$ pro-regime)",
         "(a) By sender signal", res["monotonicity_z"]),
        (axes[1], "by_theta", r"regime strength $\theta$",
         "(b) By state", res["monotonicity_theta"]),
    ):
        b = pd.DataFrame(res[key])
        ci = np.array(b["smd_ci"].tolist())
        ax.errorbar(b["bin_center"], b["mean_abs_smd"],
                    yerr=[b["mean_abs_smd"] - ci[:, 0], ci[:, 1] - b["mean_abs_smd"]],
                    fmt="o-", color=C_SURV, lw=1.2, ms=3.5, capsize=2,
                    label="feature divergence (|SMD|)")
        ax2 = ax.twinx()
        ax2.plot(b["bin_center"], b["classifier_auc"], "s--", color=C_PURE,
                 lw=1.0, ms=3, alpha=0.75, label="arm-classifier AUC")
        ax2.set_ylim(0.45, 1.02)
        ax2.axhline(0.5, color="#bbbbbb", lw=0.6, ls=":")
        ax2.set_ylabel("arm-classifier AUC", fontsize=7.5, color=C_PURE)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("mean |SMD| across\ntext features", fontsize=8, color=C_SURV)
        ax.set_title(f"{title}  (Spearman $\\rho$={mono['spearman_rho']:.2f}, "
                     f"$p$={mono['p']:.3f})", fontsize=8.5)
    fig.suptitle("Style shift is near-universal; the selection gradient is monotone",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_lemma1_text.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_crisis_blinding()
    fig_dose_response()
    fig_lemma1()
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"[figures] {f}")


if __name__ == "__main__":
    main()
