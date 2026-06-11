"""Scoring and paired stats for the regime-analyst pilot (paper 2, Phase A).

Reads output/analyst-pilot/<analyst_slug>/experiment_analyst_*_{summary.csv,log.json}
and computes, per analyst model:

  Nested corpus (baseline vs surveillance, matched cells):
    - strength:   Brier((FALL/100 - coup_success)^2), AUC(FALL -> coup),
                  |Spearman rho(FALL, theta)|
    - join frac:  MAE(|JOIN_PERCENT/100 - jf_true_shown|), calibration slope
    - per-sender: accuracy, pooled AUC of p_join
    - targeting:  precision@5 LIFT = prec@5 - jf_true_shown

  Paired contrasts: per-cell Delta(baseline - surveillance) with paired t-test
  and a 10,000-draw sign-flip permutation test (seed 42) for decomposable
  metrics; paired cluster bootstrap over cells (10,000 resamples) for pooled
  AUC and Spearman rho deltas. Heterogeneity split theta<0 vs theta>=0.

  Coded corpus (direct vs coded): theta-recovery only (no decision truth).

Go/no-go rule (preregistered in proposals/analyst_pilot_prereg.md): GO if
surveillance degrades per-sender AUC AND precision@5 lift in the predicted
direction with permutation/bootstrap p < 0.05 for >= 2 of 3+ analyst models
on the nested corpus.

Usage: uv run python analysis/analyst_scoring.py
Writes: analysis/analyst_results.json, output/analyst-pilot/figures/fig_analyst_pilot.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import C_PURE, C_SURV, apply_style  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = PROJECT_ROOT / "output" / "analyst-pilot"
RESULTS_PATH = PROJECT_ROOT / "analysis" / "analyst_results.json"
FIG_DIR = PILOT_DIR / "figures"

N_PERM = 10_000
N_BOOT = 10_000
PERM_SEED = 42

ARM_PAIRS = {"nested": ("baseline", "surveillance"), "coded_pairs": ("direct", "coded")}


# ── Loading ───────────────────────────────────────────────────────────


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cell-level summary df, sender-level df) across all analysts/arms."""
    summaries, senders = [], []
    for model_dir in sorted(PILOT_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "figures":
            continue
        for log_path in sorted(model_dir.glob("experiment_analyst_*_log.json")):
            label = log_path.stem.replace("experiment_analyst_", "").replace("_log", "")
            corpus, arm = label.rsplit("_", 1)
            with open(log_path) as f:
                entries = json.load(f)
            for e in entries:
                row = {k: v for k, v in e.items() if k not in ("senders", "response_text")}
                row["corpus"], row["arm"] = corpus, arm
                summaries.append(row)
                for s in e.get("senders", []):
                    if s.get("parse_ok") and s.get("true_decision") in ("JOIN", "STAY"):
                        senders.append({
                            "analyst_model": e["analyst_model"],
                            "corpus": corpus, "arm": arm,
                            "country": e["country"], "period": e["period"],
                            "theta": e["theta"],
                            "p_join": s["p_join"],
                            "joined": int(s["true_decision"] == "JOIN"),
                        })
    return pd.DataFrame(summaries), pd.DataFrame(senders)


# ── Stats helpers ─────────────────────────────────────────────────────


def pooled_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann-Whitney AUC; None if labels are one-class."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = stats.rankdata(scores)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def paired_perm_test(diffs: np.ndarray, seed: int = PERM_SEED) -> dict:
    """Mean paired delta with paired t and sign-flip permutation p-values."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return {"n": int(len(d)), "mean": None, "p_t": None, "p_perm": None}
    mean = float(d.mean())
    t_p = float(stats.ttest_1samp(d, 0.0).pvalue)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    perm_means = (flips * d).mean(axis=1)
    p_perm = float((np.abs(perm_means) >= abs(mean)).mean())
    return {"n": int(len(d)), "mean": mean,
            "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "p_t": t_p, "p_perm": p_perm}


def bootstrap_pooled_delta(
    sender_df: pd.DataFrame, arm_a: str, arm_b: str, seed: int = PERM_SEED
) -> dict:
    """Paired cluster bootstrap (over cells) of pooled sender-AUC delta."""
    cells = sorted(set(map(tuple, sender_df[["country", "period"]].values)))
    if len(cells) < 3:
        return {"delta": None, "ci": None, "p_boot": None}
    by_cell = {
        (arm, c): g
        for (arm, *c_), g in sender_df.groupby(["arm", "country", "period"])
        for c in [tuple(c_)]
    }

    def _auc(arm: str, cell_sample: list) -> float | None:
        frames = [by_cell[(arm, c)] for c in cell_sample if (arm, c) in by_cell]
        if not frames:
            return None
        df = pd.concat(frames)
        return pooled_auc(df["joined"].values, df["p_join"].values)

    auc_a, auc_b = _auc(arm_a, cells), _auc(arm_b, cells)
    if auc_a is None or auc_b is None:
        return {"delta": None, "ci": None, "p_boot": None}
    delta = auc_a - auc_b
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        sample = [cells[i] for i in rng.integers(0, len(cells), len(cells))]
        a, b = _auc(arm_a, sample), _auc(arm_b, sample)
        if a is not None and b is not None:
            boots.append(a - b)
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p: share of resampled deltas crossing zero
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"auc_a": auc_a, "auc_b": auc_b, "delta": float(delta),
            "ci": [float(lo), float(hi)], "p_boot": p, "n_cells": len(cells)}


def bootstrap_spearman_delta(
    paired: pd.DataFrame, col: str = "fall_est", seed: int = PERM_SEED
) -> dict:
    """Paired cell bootstrap of Delta |Spearman rho(fall, theta)| (A - B arms)."""
    a = paired[f"{col}_a"].values.astype(float)
    b = paired[f"{col}_b"].values.astype(float)
    th = paired["theta"].values.astype(float)
    ok = ~(np.isnan(a) | np.isnan(b) | np.isnan(th))
    a, b, th = a[ok], b[ok], th[ok]
    if len(a) < 5:
        return {"delta": None}

    def _absrho(x, t):
        if len(set(x)) < 2:
            return 0.0
        return abs(stats.spearmanr(x, t).statistic)

    delta = _absrho(a, th) - _absrho(b, th)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(a), len(a))
        boots.append(_absrho(a[idx], th[idx]) - _absrho(b[idx], th[idx]))
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"rho_a": _absrho(a, th), "rho_b": _absrho(b, th),
            "delta": float(delta), "ci": [float(lo), float(hi)],
            "p_boot": p, "n_cells": int(len(a))}


# ── Per-analyst scoring ───────────────────────────────────────────────


def pair_cells(df: pd.DataFrame, arm_a: str, arm_b: str) -> pd.DataFrame:
    """Merge the two arms on (country, period); suffix _a/_b; drop error rows pairwise."""
    a = df[(df.arm == arm_a) & (~df.api_error.astype(bool))]
    b = df[(df.arm == arm_b) & (~df.api_error.astype(bool))]
    merged = a.merge(b, on=["country", "period"], suffixes=("_a", "_b"))
    merged["theta"] = merged["theta_a"]
    return merged


def _cell_columns(paired: pd.DataFrame, suffix: str) -> dict[str, np.ndarray]:
    fall = paired[f"fall_est{suffix}"].values.astype(float) / 100.0
    coup = paired[f"coup_success{suffix}"].astype(float).values
    jf_est = paired[f"join_pct_est{suffix}"].values.astype(float) / 100.0
    jf_true = paired[f"jf_true_shown{suffix}"].values.astype(float)
    prec5 = paired[f"n_top5_join_true{suffix}"].values.astype(float) / 5.0
    return {
        "brier": (fall - coup) ** 2,
        "jf_abs_err": np.abs(jf_est - jf_true),
        "sender_acc": paired[f"sender_accuracy{suffix}"].values.astype(float),
        "prec5_lift": prec5 - jf_true,
    }


METRIC_SIGNS = {
    # Predicted direction of Delta = metric(baseline) - metric(surveillance)
    # under H-A1 (analysts do BETTER on baseline messages).
    "brier": -1,        # error metric: baseline lower -> Delta negative
    "jf_abs_err": -1,   # error metric
    "sender_acc": +1,   # accuracy: baseline higher -> Delta positive
    "prec5_lift": +1,   # lift: baseline higher -> Delta positive
}


def score_analyst(
    summary: pd.DataFrame, senders: pd.DataFrame, model: str, corpus: str
) -> dict:
    arm_a, arm_b = ARM_PAIRS[corpus]
    df = summary[(summary.analyst_model == model) & (summary.corpus == corpus)]
    if df.empty:
        return {}
    paired = pair_cells(df, arm_a, arm_b)
    out: dict = {"arms": [arm_a, arm_b], "n_paired_cells": int(len(paired))}

    if corpus == "nested":
        cols_a, cols_b = _cell_columns(paired, "_a"), _cell_columns(paired, "_b")
        for metric in METRIC_SIGNS:
            res = paired_perm_test(cols_a[metric] - cols_b[metric])
            res["arm_means"] = {
                arm_a: float(np.nanmean(cols_a[metric])),
                arm_b: float(np.nanmean(cols_b[metric])),
            }
            res["predicted_sign"] = METRIC_SIGNS[metric]
            # Heterogeneity: weak-regime vs strong-regime cells (H-A2)
            weak = paired["theta"].values < 0
            res["delta_theta_lt0"] = paired_perm_test(
                (cols_a[metric] - cols_b[metric])[weak])
            res["delta_theta_ge0"] = paired_perm_test(
                (cols_a[metric] - cols_b[metric])[~weak])
            out[metric] = res
        # Pooled sender AUC (cluster bootstrap)
        sd = senders[(senders.analyst_model == model) & (senders.corpus == corpus)]
        out["sender_auc"] = bootstrap_pooled_delta(sd, arm_a, arm_b)
        # AUC(FALL -> coup) per arm (descriptive)
        out["fall_auc"] = {
            arm: pooled_auc(
                paired[f"coup_success_{s}"].astype(float).values,
                paired[f"fall_est_{s}"].values.astype(float),
            )
            for arm, s in ((arm_a, "a"), (arm_b, "b"))
        }
        # Calibration slope of jf_true on jf_est, per arm
        for arm, s in ((arm_a, "a"), (arm_b, "b")):
            x = paired[f"join_pct_est_{s}"].values.astype(float) / 100.0
            y = paired[f"jf_true_shown_{s}"].values.astype(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            out.setdefault("jf_calibration_slope", {})[arm] = (
                float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() > 5 and np.std(x[ok]) > 0
                else None
            )
        # Robustness: Delta sender accuracy vs Delta mean message length is
        # computed downstream from the logs if needed; lengths not in summary.
    else:  # coded corpus: theta recovery only
        out["abs_spearman_fall_theta"] = bootstrap_spearman_delta(paired)
        med = float(np.median(paired["theta"].values))
        # Higher FALL should predict the weaker-regime half (theta < median).
        out["theta_direction_auc"] = {
            arm: pooled_auc(
                (paired["theta"].values < med).astype(float),
                paired[f"fall_est_{s}"].values.astype(float),
            )
            for arm, s in ((arm_a, "a"), (arm_b, "b"))
        }
    return out


# ── Go/no-go and reporting ────────────────────────────────────────────


def evaluate_go_nogo(results: dict) -> dict:
    """Preregistered rule: sender AUC and prec@5 lift degrade (predicted sign,
    p<0.05) for >= 2 analyst models on the nested corpus."""
    votes = {}
    for model, res in results.items():
        nested = res.get("nested", {})
        acc = nested.get("prec5_lift", {})
        auc = nested.get("sender_auc", {})
        acc_ok = (
            acc.get("mean") is not None and acc["mean"] > 0
            and acc.get("p_perm") is not None and acc["p_perm"] < 0.05
        )
        auc_ok = (
            auc.get("delta") is not None and auc["delta"] > 0
            and auc.get("p_boot") is not None and auc["p_boot"] < 0.05
        )
        votes[model] = {"prec5_lift_pass": bool(acc_ok), "sender_auc_pass": bool(auc_ok),
                        "both": bool(acc_ok and auc_ok)}
    n_pass = sum(v["both"] for v in votes.values())
    return {"votes": votes, "n_models_passing": n_pass, "go": bool(n_pass >= 2)}


def print_table(results: dict) -> None:
    print(f"\n{'analyst':<42} {'metric':<14} {'baseline':>9} {'surveil':>9} "
          f"{'delta':>8} {'p_perm':>8}")
    print("-" * 95)
    for model, res in sorted(results.items()):
        nested = res.get("nested", {})
        for metric in ("brier", "jf_abs_err", "sender_acc", "prec5_lift"):
            m = nested.get(metric)
            if not m or m.get("mean") is None:
                continue
            arms = m["arm_means"]
            print(f"{model:<42} {metric:<14} {arms['baseline']:>9.3f} "
                  f"{arms['surveillance']:>9.3f} {m['mean']:>8.3f} {m['p_perm']:>8.4f}")
        auc = nested.get("sender_auc", {})
        if auc.get("delta") is not None:
            print(f"{model:<42} {'sender_auc':<14} {auc['auc_a']:>9.3f} "
                  f"{auc['auc_b']:>9.3f} {auc['delta']:>8.3f} {auc['p_boot']:>8.4f}")
        rho = res.get("coded_pairs", {}).get("abs_spearman_fall_theta", {})
        if rho.get("delta") is not None:
            print(f"{model:<42} {'|rho| dir/cod':<14} {rho['rho_a']:>9.3f} "
                  f"{rho['rho_b']:>9.3f} {rho['delta']:>8.3f} {rho['p_boot']:>8.4f}")


def make_figure(results: dict) -> Path:
    apply_style()
    import matplotlib.pyplot as plt

    panels = [
        ("sender_auc", "Per-sender AUC", "pooled"),
        ("prec5_lift", "Precision@5 lift", "cell"),
        ("sender_acc", "Per-sender accuracy", "cell"),
        ("jf_abs_err", "Join-fraction MAE", "cell"),
    ]
    models = [m for m in sorted(results) if results[m].get("nested")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    for ax, (key, title, kind) in zip(axes.flat, panels):
        xs = np.arange(len(models))
        width = 0.36
        for off, arm, color in ((-width / 2, "baseline", C_PURE),
                                (width / 2, "surveillance", C_SURV)):
            vals, errs = [], []
            for m in models:
                nested = results[m]["nested"]
                if kind == "pooled":
                    d = nested.get(key, {})
                    v = d.get("auc_a" if arm == "baseline" else "auc_b")
                    vals.append(v if v is not None else np.nan)
                    errs.append(np.nan)
                else:
                    d = nested.get(key, {})
                    vals.append(d.get("arm_means", {}).get(arm, np.nan))
                    errs.append(1.96 * d["se"] / np.sqrt(2) if d.get("se") else np.nan)
            ax.bar(xs + off, vals, width, color=color, label=arm)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([m.split("/")[-1][:18] for m in models],
                           rotation=30, ha="right", fontsize=6)
        # annotate p-values
        for i, m in enumerate(models):
            d = results[m]["nested"].get(key, {})
            p = d.get("p_boot") if kind == "pooled" else d.get("p_perm")
            if p is not None:
                ax.annotate(f"p={p:.3f}", (i, ax.get_ylim()[1] * 0.97),
                            ha="center", va="top", fontsize=5.5)
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Regime-analyst pilot: baseline vs surveillance messages",
                 fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_analyst_pilot.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    summary, senders = load_runs()
    if summary.empty:
        raise SystemExit(f"no analyst outputs found under {PILOT_DIR}")
    results: dict = {}
    for model in sorted(summary.analyst_model.unique()):
        results[model] = {}
        for corpus in summary[summary.analyst_model == model].corpus.unique():
            scored = score_analyst(summary, senders, model, corpus)
            if scored:
                results[model][corpus] = scored
    verdict = evaluate_go_nogo(results)
    payload = {"results": results, "go_nogo": verdict,
               "n_perm": N_PERM, "n_boot": N_BOOT, "seed": PERM_SEED}
    RESULTS_PATH.write_text(json.dumps(payload, indent=1, default=float))
    print_table(results)
    fig = make_figure(results)
    print(f"\n[scoring] results -> {RESULTS_PATH}")
    print(f"[scoring] figure  -> {fig}")
    print(f"[scoring] GO/NO-GO: {'GO' if verdict['go'] else 'NO-GO'} "
          f"({verdict['n_models_passing']} models pass both prereg criteria)")


if __name__ == "__main__":
    main()
