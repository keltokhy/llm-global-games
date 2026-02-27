#!/usr/bin/env python3
"""
Construct validity test: are LLMs playing a strategic game, or just doing
sentiment classification on briefing text?

Tests:
  1. Latent Feature Baseline — fit logistic regression from (direction,
     clarity, coordination) to decision.  Compare 3-feature vs 1-feature
     (direction only) accuracy across all 9 models.
  2. Cross-Treatment Departure — train the latent-feature classifier on
     PURE data, then predict on communication and surveillance data.
     Systematic residuals would mean LLMs go beyond text classification.
  3. Figure — two-panel figure saved to paper/figures/.

Usage: uv run python analysis/construct_validity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_PATH = Path(__file__).resolve().parent / "construct_validity_results.json"

# ── Models ────────────────────────────────────────────────────────
PART1_MODELS = [
    "mistralai--mistral-small-creative",
    "meta-llama--llama-3.3-70b-instruct",
    "mistralai--ministral-3b-2512",
    "qwen--qwen3-30b-a3b-instruct-2507",
    "openai--gpt-oss-120b",
    "qwen--qwen3-235b-a22b-2507",
    "arcee-ai--trinity-large-preview_free",
    "minimax--minimax-m2-her",
]

SHORT_NAMES = {
    "mistralai--mistral-small-creative": "Mistral-Small",
    "meta-llama--llama-3.3-70b-instruct": "Llama-3.3-70B",
    "mistralai--ministral-3b-2512": "Ministral-3B",
    "qwen--qwen3-30b-a3b-instruct-2507": "Qwen3-30B",
    "openai--gpt-oss-120b": "GPT-OSS-120B",
    "qwen--qwen3-235b-a22b-2507": "Qwen3-235B",
    "arcee-ai--trinity-large-preview_free": "Trinity",
    "minimax--minimax-m2-her": "MiniMax-M2",
}

# ── Figure styling (matches make_figures.py) ──────────────────────
COL_W = 3.4
TEXT_W = 7.0

plt.rcParams.update({
    "font.family":          "serif",
    "font.size":            8,
    "axes.titlesize":       9,
    "axes.labelsize":       8,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      6.5,
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
})

# Colors
C_PURE     = "#636363"
C_COMM     = "#2c7bb6"
C_SURV     = "#7b3294"
C_1FEAT    = "#fdae61"
C_3FEAT    = "#1a9641"

LW_REF = 0.6
ANNOT_BOX = dict(boxstyle="round,pad=0.3", facecolor="white",
                 alpha=0.92, edgecolor="#d0d0d0", linewidth=0.5)


# ── Data loading ──────────────────────────────────────────────────

def load_agent_data(log_path: Path) -> pd.DataFrame | None:
    """Load a log JSON and return a flat DataFrame of agent-level observations.

    Returns columns: direction, clarity, coordination, decision (0/1).
    Drops agents with api_error or unparseable decisions.
    """
    if not log_path.exists():
        return None

    with open(log_path, encoding="utf-8") as f:
        periods = json.load(f)

    rows = []
    for period in periods:
        theta = period.get("theta")
        for agent in period.get("agents", []):
            if agent.get("api_error"):
                continue
            decision_str = agent.get("decision")
            if decision_str not in ("JOIN", "STAY"):
                continue
            rows.append({
                "theta": theta,
                "direction": agent["direction"],
                "clarity": agent["clarity"],
                "coordination": agent["coordination"],
                "decision": 1 if decision_str == "JOIN" else 0,
            })

    if not rows:
        return None
    return pd.DataFrame(rows)


# ── Analysis helpers ──────────────────────────────────────────────

def fit_and_evaluate(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """Fit LogisticRegression with CV and return accuracy + pseudo-R2."""
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    # Get cross-validated predicted probabilities for pseudo-R2
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    # McFadden pseudo-R2: 1 - LL_model / LL_null
    ll_model = -log_loss(y, proba, normalize=False)
    p_bar = y.mean()
    ll_null = len(y) * (p_bar * np.log(p_bar + 1e-15) +
                        (1 - p_bar) * np.log(1 - p_bar + 1e-15))

    pseudo_r2 = 1.0 - ll_model / ll_null if ll_null != 0 else 0.0

    # Fit on full data for coefficients
    clf.fit(X, y)

    return {
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std": float(np.std(acc_scores)),
        "pseudo_r2": float(pseudo_r2),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "n_obs": int(len(y)),
    }


# ── Part 1: Latent Feature Baseline ──────────────────────────────

def run_latent_feature_baseline() -> list[dict]:
    """For each model, compare 3-feature vs 1-feature logistic regression."""
    results = []

    for slug in PART1_MODELS:
        name = SHORT_NAMES.get(slug, slug)
        log_path = OUTPUT_DIR / slug / "experiment_pure_log.json"
        df = load_agent_data(log_path)
        if df is None:
            print(f"  SKIP {name}: no pure log")
            continue

        X_3 = df[["direction", "clarity", "coordination"]].values
        X_1 = df[["direction"]].values
        y = df["decision"].values

        # Check for degenerate cases (all same class)
        if len(np.unique(y)) < 2:
            print(f"  SKIP {name}: degenerate labels (all same decision)")
            continue

        res_3 = fit_and_evaluate(X_3, y)
        res_1 = fit_and_evaluate(X_1, y)

        row = {
            "model_slug": slug,
            "model_name": name,
            "n_obs": res_3["n_obs"],
            "acc_3feat": res_3["cv_accuracy_mean"],
            "acc_3feat_std": res_3["cv_accuracy_std"],
            "r2_3feat": res_3["pseudo_r2"],
            "coef_direction": res_3["coef"][0],
            "coef_clarity": res_3["coef"][1],
            "coef_coordination": res_3["coef"][2],
            "intercept_3feat": res_3["intercept"],
            "acc_1feat": res_1["cv_accuracy_mean"],
            "acc_1feat_std": res_1["cv_accuracy_std"],
            "r2_1feat": res_1["pseudo_r2"],
            "coef_direction_only": res_1["coef"][0],
            "intercept_1feat": res_1["intercept"],
            "acc_gain": res_3["cv_accuracy_mean"] - res_1["cv_accuracy_mean"],
            "r2_gain": res_3["pseudo_r2"] - res_1["pseudo_r2"],
        }
        results.append(row)
        print(f"  {name:20s}  3-feat acc={res_3['cv_accuracy_mean']:.3f}  "
              f"1-feat acc={res_1['cv_accuracy_mean']:.3f}  "
              f"gain={row['acc_gain']:+.3f}  "
              f"R2_3={res_3['pseudo_r2']:.3f}  R2_1={res_1['pseudo_r2']:.3f}")

    return results


# ── Part 2: Cross-Treatment Departure ────────────────────────────

def run_cross_treatment_departure() -> list[dict]:
    """Train on PURE, predict on comm/surveillance. Report residuals."""
    results = []

    for slug in PART1_MODELS:
        name = SHORT_NAMES.get(slug, slug)

        # Train on pure
        pure_path = OUTPUT_DIR / slug / "experiment_pure_log.json"
        df_pure = load_agent_data(pure_path)
        if df_pure is None:
            continue
        if len(np.unique(df_pure["decision"].values)) < 2:
            continue

        X_train = df_pure[["direction", "clarity", "coordination"]].values
        y_train = df_pure["decision"].values

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train)

        # Predict on pure (self-check)
        p_pure = clf.predict_proba(X_train)[:, 1]
        resid_pure = y_train - p_pure
        results.append({
            "model_slug": slug,
            "model_name": name,
            "treatment": "pure",
            "mean_residual": float(np.mean(resid_pure)),
            "std_residual": float(np.std(resid_pure)),
            "n_obs": int(len(y_train)),
            "mean_actual": float(np.mean(y_train)),
            "mean_predicted": float(np.mean(p_pure)),
        })

        # Predict on comm
        comm_path = OUTPUT_DIR / slug / "experiment_comm_log.json"
        df_comm = load_agent_data(comm_path)
        if df_comm is not None and len(df_comm) > 0:
            X_comm = df_comm[["direction", "clarity", "coordination"]].values
            y_comm = df_comm["decision"].values
            p_comm = clf.predict_proba(X_comm)[:, 1]
            resid_comm = y_comm - p_comm
            results.append({
                "model_slug": slug,
                "model_name": name,
                "treatment": "comm",
                "mean_residual": float(np.mean(resid_comm)),
                "std_residual": float(np.std(resid_comm)),
                "n_obs": int(len(y_comm)),
                "mean_actual": float(np.mean(y_comm)),
                "mean_predicted": float(np.mean(p_comm)),
            })

        # Predict on surveillance
        surv_path = OUTPUT_DIR / "surveillance" / slug / "experiment_comm_log.json"
        df_surv = load_agent_data(surv_path)
        if df_surv is not None and len(df_surv) > 0:
            X_surv = df_surv[["direction", "clarity", "coordination"]].values
            y_surv = df_surv["decision"].values
            p_surv = clf.predict_proba(X_surv)[:, 1]
            resid_surv = y_surv - p_surv
            results.append({
                "model_slug": slug,
                "model_name": name,
                "treatment": "surveillance",
                "mean_residual": float(np.mean(resid_surv)),
                "std_residual": float(np.std(resid_surv)),
                "n_obs": int(len(y_surv)),
                "mean_actual": float(np.mean(y_surv)),
                "mean_predicted": float(np.mean(p_surv)),
            })

    return results


# ── Figure ────────────────────────────────────────────────────────

def make_figure(baseline_results: list[dict], departure_results: list[dict]):
    """Two-panel figure: (a) baseline accuracy comparison, (b) treatment residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 3.0))

    # ── Panel A: Latent feature baseline accuracy ─────────────────
    ax = axes[0]
    if baseline_results:
        bdf = pd.DataFrame(baseline_results).sort_values("acc_3feat", ascending=True)
        y_pos = np.arange(len(bdf))
        bar_h = 0.35

        ax.barh(y_pos - bar_h / 2, bdf["acc_1feat"], height=bar_h,
                color=C_1FEAT, edgecolor="none", label="Direction only")
        ax.barh(y_pos + bar_h / 2, bdf["acc_3feat"], height=bar_h,
                color=C_3FEAT, edgecolor="none", label="3-feature model")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(bdf["model_name"], fontsize=7)
        ax.set_xlabel("Cross-validated accuracy")
        ax.set_title("A.  Latent feature baseline", loc="left", fontsize=9)
        ax.legend(loc="lower right", fontsize=6)
        ax.set_xlim(0.5, 1.0)
        ax.xaxis.grid(True, linewidth=0.3, alpha=0.3, color="#cccccc")
        ax.set_axisbelow(True)

        # Annotate mean gain
        mean_gain = bdf["acc_gain"].mean()
        ax.text(0.97, 0.15,
                f"Mean gain: {mean_gain:+.1%}",
                transform=ax.transAxes, fontsize=6.5,
                va="bottom", ha="right", bbox=ANNOT_BOX)

    # ── Panel B: Cross-treatment departure ────────────────────────
    ax = axes[1]
    if departure_results:
        ddf = pd.DataFrame(departure_results)

        # Aggregate across models per treatment
        treatments = ["pure", "comm", "surveillance"]
        treatment_colors = {"pure": C_PURE, "comm": C_COMM, "surveillance": C_SURV}
        treatment_labels = {"pure": "Pure (self-check)",
                            "comm": "Communication",
                            "surveillance": "Surveillance"}

        # Group by model + treatment for a grouped bar chart
        models_with_departure = []
        for slug in PART1_MODELS:
            sub = ddf[ddf["model_slug"] == slug]
            if len(sub) > 1:  # Must have at least pure + one other
                models_with_departure.append(slug)

        if models_with_departure:
            y_pos = np.arange(len(models_with_departure))
            bar_h = 0.25
            offsets = {"pure": -bar_h, "comm": 0, "surveillance": bar_h}

            for treatment in treatments:
                vals = []
                positions = []
                for i, slug in enumerate(models_with_departure):
                    row = ddf[(ddf["model_slug"] == slug) & (ddf["treatment"] == treatment)]
                    if len(row) > 0:
                        vals.append(row.iloc[0]["mean_residual"])
                        positions.append(i + offsets[treatment])

                if vals:
                    ax.barh(positions, vals, height=bar_h,
                            color=treatment_colors[treatment], edgecolor="none",
                            label=treatment_labels[treatment])

            ax.axvline(0, color="#333", linewidth=LW_REF)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([SHORT_NAMES.get(s, s) for s in models_with_departure],
                               fontsize=7)
            ax.set_xlabel("Mean residual (actual $-$ predicted)")
            ax.set_title("B.  Cross-treatment departure", loc="left", fontsize=9)
            ax.legend(loc="lower right", fontsize=6)
            ax.xaxis.grid(True, linewidth=0.3, alpha=0.3, color="#cccccc")
            ax.set_axisbelow(True)

            # Annotate aggregate stats
            comm_resid = ddf[ddf["treatment"] == "comm"]["mean_residual"]
            surv_resid = ddf[ddf["treatment"] == "surveillance"]["mean_residual"]
            text_parts = []
            if len(comm_resid) > 0:
                text_parts.append(f"Comm mean: {comm_resid.mean():+.3f}")
            if len(surv_resid) > 0:
                text_parts.append(f"Surv mean: {surv_resid.mean():+.3f}")
            if text_parts:
                ax.text(0.97, 0.97, "\n".join(text_parts),
                        transform=ax.transAxes, fontsize=6.5,
                        va="top", ha="right", bbox=ANNOT_BOX)

    fig.align_labels()
    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / "fig_construct_validity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_construct_validity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved fig_construct_validity to {FIG_DIR}")


# ── Summary table ─────────────────────────────────────────────────

def print_summary(baseline_results: list[dict], departure_results: list[dict]):
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("CONSTRUCT VALIDITY: LATENT FEATURE BASELINE")
    print("=" * 80)

    if baseline_results:
        print(f"\n{'Model':>20s}  {'N':>6s}  {'Acc(3f)':>8s}  {'Acc(1f)':>8s}  "
              f"{'Gain':>7s}  {'R2(3f)':>7s}  {'R2(1f)':>7s}  {'R2 gain':>7s}")
        print("-" * 80)
        for r in sorted(baseline_results, key=lambda x: -x["acc_3feat"]):
            print(f"{r['model_name']:>20s}  {r['n_obs']:6d}  "
                  f"{r['acc_3feat']:8.3f}  {r['acc_1feat']:8.3f}  "
                  f"{r['acc_gain']:+7.3f}  "
                  f"{r['r2_3feat']:7.3f}  {r['r2_1feat']:7.3f}  "
                  f"{r['r2_gain']:+7.3f}")

        # Aggregate row
        mean_acc3 = np.mean([r["acc_3feat"] for r in baseline_results])
        mean_acc1 = np.mean([r["acc_1feat"] for r in baseline_results])
        mean_r2_3 = np.mean([r["r2_3feat"] for r in baseline_results])
        mean_r2_1 = np.mean([r["r2_1feat"] for r in baseline_results])
        print("-" * 80)
        print(f"{'MEAN':>20s}  {'':>6s}  {mean_acc3:8.3f}  {mean_acc1:8.3f}  "
              f"{mean_acc3 - mean_acc1:+7.3f}  "
              f"{mean_r2_3:7.3f}  {mean_r2_1:7.3f}  "
              f"{mean_r2_3 - mean_r2_1:+7.3f}")

        # Coefficient summary
        print(f"\n{'Model':>20s}  {'b(dir)':>8s}  {'b(clar)':>8s}  {'b(coord)':>9s}  "
              f"{'intercept':>9s}")
        print("-" * 60)
        for r in sorted(baseline_results, key=lambda x: -x["acc_3feat"]):
            print(f"{r['model_name']:>20s}  {r['coef_direction']:8.3f}  "
                  f"{r['coef_clarity']:8.3f}  {r['coef_coordination']:9.3f}  "
                  f"{r['intercept_3feat']:9.3f}")

    print("\n" + "=" * 80)
    print("CONSTRUCT VALIDITY: CROSS-TREATMENT DEPARTURE")
    print("=" * 80)

    if departure_results:
        ddf = pd.DataFrame(departure_results)
        print(f"\n{'Model':>20s}  {'Treatment':>14s}  {'N':>6s}  "
              f"{'Actual':>7s}  {'Predicted':>9s}  {'Residual':>9s}")
        print("-" * 75)
        for _, row in ddf.iterrows():
            print(f"{row['model_name']:>20s}  {row['treatment']:>14s}  "
                  f"{row['n_obs']:6d}  "
                  f"{row['mean_actual']:7.3f}  {row['mean_predicted']:9.3f}  "
                  f"{row['mean_residual']:+9.3f}")

        # Aggregate by treatment
        print("-" * 75)
        for treatment in ["pure", "comm", "surveillance"]:
            sub = ddf[ddf["treatment"] == treatment]
            if len(sub) > 0:
                print(f"{'MEAN':>20s}  {treatment:>14s}  "
                      f"{'':>6s}  "
                      f"{sub['mean_actual'].mean():7.3f}  "
                      f"{sub['mean_predicted'].mean():9.3f}  "
                      f"{sub['mean_residual'].mean():+9.3f}")

    print()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Construct validity analysis")
    print(f"  Data root: {OUTPUT_DIR}")
    print(f"  Output: {FIG_DIR}\n")

    print("Part 1: Latent Feature Baseline (3-feat vs 1-feat logistic)")
    print("-" * 60)
    baseline_results = run_latent_feature_baseline()

    print(f"\nPart 2: Cross-Treatment Departure")
    print("-" * 60)
    departure_results = run_cross_treatment_departure()

    # Summary
    print_summary(baseline_results, departure_results)

    # Figure
    make_figure(baseline_results, departure_results)

    # Save results JSON
    output = {
        "latent_feature_baseline": baseline_results,
        "cross_treatment_departure": departure_results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {RESULTS_PATH}")
