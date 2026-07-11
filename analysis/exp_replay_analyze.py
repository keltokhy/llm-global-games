#!/usr/bin/env python3
r"""Matched-cell, country-clustered analysis of replay arms (Experiments A/B).

Each arm is an experiment_comm_log.json produced by replaying a fixed-message
log to Llama receivers on the nested 10x50 grid. For a contrast (treat vs ref)
we collapse to (country, period) cell join fractions, pair on the cell, cluster
on the 10 countries, and report a cluster-robust t (G-1 dof) plus an exact
restricted wild-cluster bootstrap (full 2^G enumeration -- deterministic).

Usage:
  uv run python analysis/exp_replay_analyze.py LABEL=path/to/experiment_comm_log.json ...
  # then it prints cell means for each arm and a set of named contrasts passed via --contrast TREAT:REF
  uv run python analysis/exp_replay_analyze.py \
    --csv analysis/exp_ab_cell_joins.csv \
    --stats-out paper/tables/stats_replay.tex \
    --table-out paper/tables/tab_replay_mechanisms.tex
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


CSV_ARMS = [
    "B_comm",
    "B_surv",
    "B_risk_stripped",
    "B_risk_only",
    "A_w1_direct",
    "A_w1_coded",
    "A_w0_direct",
    "A_w0_coded",
]

CSV_CONTRASTS = [
    ("B_surv", "B_comm"),
    ("B_risk_stripped", "B_comm"),
    ("B_risk_stripped", "B_surv"),
    ("B_risk_only", "B_comm"),
    ("B_risk_only", "B_surv"),
    ("A_w1_direct", "A_w0_direct"),
    ("A_w1_coded", "A_w0_coded"),
    ("A_w1_direct", "A_w1_coded"),
    ("A_w0_direct", "A_w0_coded"),
    ("A_w1_direct", "B_comm"),
    ("A_w1_coded", "B_comm"),
    ("A_w0_direct", "B_comm"),
    ("A_w0_coded", "B_comm"),
]

CSV_AGENTS_PER_CELL = 25


def load_cells(path: Path) -> dict[tuple, float]:
    """(country, period) -> valid join fraction."""
    entries = json.loads(Path(path).read_text())
    out = {}
    for e in entries:
        agents = e.get("agents", [])
        dec = [a.get("decision") for a in agents]
        valid = [d for d in dec if d in ("JOIN", "STAY")]
        if not valid:
            continue
        out[(e["country"], e["period"])] = sum(1 for d in valid if d == "JOIN") / len(valid)
    return out


def mean_join(path: Path) -> tuple[float, int]:
    cells = load_cells(path)
    v = np.array(list(cells.values()))
    return float(v.mean()) * 100, len(v)


def _beta_se(d: np.ndarray, gi: np.ndarray, G: int):
    N = len(d)
    beta = float(d.mean())
    e = d - beta
    sg = np.array([e[gi == k].sum() for k in range(G)])
    meat = float((sg ** 2).sum())
    corr = G / (G - 1)  # K=1, (N-1)/(N-1)
    var = corr * meat / (N ** 2)
    se = float(np.sqrt(var)) if var > 0 else float("nan")
    return beta, se


def contrast(treat: Path, ref: Path, scale: float = 100.0) -> dict:
    ct = load_cells(treat)
    cr = load_cells(ref)
    keys = sorted(set(ct) & set(cr))
    if not keys:
        return {"error": "no matched cells"}
    d = np.array([ct[k] - cr[k] for k in keys]) * scale
    countries = [k[0] for k in keys]
    groups = sorted(set(countries))
    G = len(groups)
    gidx = {c: i for i, c in enumerate(groups)}
    gi = np.array([gidx[c] for c in countries])
    beta, se = _beta_se(d, gi, G)
    t = beta / se if se else float("nan")
    dof = G - 1
    p_t = float(2 * stats.t.sf(abs(t), dof)) if not np.isnan(t) else float("nan")
    abs_t = abs(t)
    count = total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=G):
        w = np.array(signs)[gi]
        b_s, se_s = _beta_se(w * d, gi, G)
        t_s = b_s / se_s if se_s else 0.0
        total += 1
        if abs(t_s) >= abs_t - 1e-12:
            count += 1
    return {
        "delta_pp": round(beta, 2), "se": round(se, 3), "t": round(t, 2),
        "dof": dof, "p_cluster_t": round(p_t, 4),
        "p_wild_exact": round(count / total, 4),
        "n_cells": len(keys), "n_clusters": G,
    }


def load_cells_csv(path: Path, arm: str) -> dict[tuple, float]:
    """(country, period) -> join_valid for one arm, from the compact CSV."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["arm"] == arm:
                out[(int(r["country"]), int(r["period"]))] = float(r["join_valid"])
    return out


def contrast_cells(ct: dict, cr: dict, scale: float = 100.0) -> dict:
    keys = sorted(set(ct) & set(cr))
    if not keys:
        return {"error": "no matched cells"}
    d = np.array([ct[k] - cr[k] for k in keys]) * scale
    countries = [k[0] for k in keys]
    groups = sorted(set(countries))
    G = len(groups)
    gidx = {c: i for i, c in enumerate(groups)}
    gi = np.array([gidx[c] for c in countries])
    beta, se = _beta_se(d, gi, G)
    t = beta / se if se else float("nan")
    abs_t = abs(t)
    count = total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=G):
        w = np.array(signs)[gi]
        b_s, se_s = _beta_se(w * d, gi, G)
        t_s = b_s / se_s if se_s else 0.0
        total += 1
        if abs(t_s) >= abs_t - 1e-12:
            count += 1
    return {"delta_pp": round(beta, 2), "t": round(t, 2),
            "p_wild_exact": round(count / total, 4), "n_cells": len(keys), "n_clusters": G}


def _csv_response_diagnostics(csv_path: Path) -> dict[str, dict[str, float]]:
    summaries = {
        arm: {"valid": 0.0, "join": 0.0, "total": 0.0}
        for arm in CSV_ARMS
    }
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            arm = row["arm"]
            if arm not in summaries:
                continue
            valid = int(row["n_valid"])
            summaries[arm]["valid"] += valid
            summaries[arm]["join"] += float(row["join_valid"]) * valid
            summaries[arm]["total"] += CSV_AGENTS_PER_CELL

    for values in summaries.values():
        valid = values["valid"]
        total = values["total"]
        joins = values["join"]
        values["valid_pct"] = 100 * valid / total
        values["all_stay_mean"] = 100 * joins / total
        values["all_join_mean"] = 100 * (joins + total - valid) / total
    return summaries


def analyze_csv(csv_path: Path) -> tuple[dict, dict, dict, dict]:
    """Return per-arm cells, means, contrasts, and response diagnostics."""
    cells = {arm: load_cells_csv(csv_path, arm) for arm in CSV_ARMS}
    means = {
        arm: float(np.mean(list(arm_cells.values()))) * 100
        for arm, arm_cells in cells.items()
        if arm_cells
    }
    results = {
        (treat, ref): contrast_cells(cells[treat], cells[ref])
        for treat, ref in CSV_CONTRASTS
        if cells[treat] and cells[ref]
    }
    diagnostics = _csv_response_diagnostics(csv_path)
    return cells, means, results, diagnostics


def _p_text(value: float) -> str:
    return f"= {value:.3f}"


def _write_stats_tex(path: Path, means: dict, results: dict, diagnostics: dict) -> None:
    def mean(arm: str) -> str:
        return f"{means[arm]:.1f}\\%"

    def result(treat: str, ref: str) -> dict:
        return results[(treat, ref)]

    def delta(treat: str, ref: str, *, absolute: bool = False) -> str:
        value = result(treat, ref)["delta_pp"]
        if absolute:
            value = abs(value)
        return f"{value:+.2f}" if not absolute else f"{value:.2f}"

    def p_text(treat: str, ref: str) -> str:
        return _p_text(result(treat, ref)["p_wild_exact"])

    def valid_pct(arm: str) -> str:
        return f"{diagnostics[arm]['valid_pct']:.1f}\\%"

    def imputed_delta(treat: str, ref: str, outcome: str) -> str:
        key = f"all_{outcome}_mean"
        value = diagnostics[treat][key] - diagnostics[ref][key]
        return f"{value:+.2f}"

    n_cells = result("B_surv", "B_comm")["n_cells"]
    n_clusters = result("B_surv", "B_comm")["n_clusters"]
    lines = [
        "% Auto-generated by analysis/exp_replay_analyze.py -- do not edit.",
        "% Source: analysis/exp_ab_cell_joins.csv.",
        f"\\providecommand{{\\ReplayN}}{{{n_cells}}}",
        f"\\providecommand{{\\ReplayG}}{{{n_clusters}}}",
        f"\\providecommand{{\\ReplayBaselineMeanJoinPct}}{{{mean('B_comm')}}}",
        f"\\providecommand{{\\ReplaySurvMeanJoinPct}}{{{mean('B_surv')}}}",
        f"\\providecommand{{\\ReplayRiskStrippedMeanJoinPct}}{{{mean('B_risk_stripped')}}}",
        f"\\providecommand{{\\ReplayRiskOnlyMeanJoinPct}}{{{mean('B_risk_only')}}}",
        f"\\providecommand{{\\ReplayWillingDirectMeanJoinPct}}{{{mean('A_w1_direct')}}}",
        f"\\providecommand{{\\ReplayWillingCodedMeanJoinPct}}{{{mean('A_w1_coded')}}}",
        f"\\providecommand{{\\ReplayNoWillingDirectMeanJoinPct}}{{{mean('A_w0_direct')}}}",
        f"\\providecommand{{\\ReplayNoWillingCodedMeanJoinPct}}{{{mean('A_w0_coded')}}}",
        f"\\providecommand{{\\ReplayBaselineValidPct}}{{{valid_pct('B_comm')}}}",
        f"\\providecommand{{\\ReplaySurvValidPct}}{{{valid_pct('B_surv')}}}",
        f"\\providecommand{{\\ReplayRiskStrippedValidPct}}{{{valid_pct('B_risk_stripped')}}}",
        f"\\providecommand{{\\ReplayRiskOnlyValidPct}}{{{valid_pct('B_risk_only')}}}",
        f"\\providecommand{{\\ReplayWillingDirectValidPct}}{{{valid_pct('A_w1_direct')}}}",
        f"\\providecommand{{\\ReplayWillingCodedValidPct}}{{{valid_pct('A_w1_coded')}}}",
        f"\\providecommand{{\\ReplayNoWillingDirectValidPct}}{{{valid_pct('A_w0_direct')}}}",
        f"\\providecommand{{\\ReplayNoWillingCodedValidPct}}{{{valid_pct('A_w0_coded')}}}",
        f"\\providecommand{{\\ReplaySurvDeltaPP}}{{{delta('B_surv', 'B_comm')}}}",
        f"\\providecommand{{\\ReplaySurvDeltaAbsPP}}{{{delta('B_surv', 'B_comm', absolute=True)}}}",
        f"\\providecommand{{\\ReplaySurvPText}}{{{p_text('B_surv', 'B_comm')}}}",
        f"\\providecommand{{\\ReplayRiskStrippedVsBaseDeltaPP}}{{{delta('B_risk_stripped', 'B_comm')}}}",
        f"\\providecommand{{\\ReplayRiskStrippedVsBasePText}}{{{p_text('B_risk_stripped', 'B_comm')}}}",
        f"\\providecommand{{\\ReplayRiskStrippedVsSurvDeltaPP}}{{{delta('B_risk_stripped', 'B_surv')}}}",
        f"\\providecommand{{\\ReplayRiskStrippedVsSurvPText}}{{{p_text('B_risk_stripped', 'B_surv')}}}",
        f"\\providecommand{{\\ReplayRiskOnlyDeltaPP}}{{{delta('B_risk_only', 'B_comm')}}}",
        f"\\providecommand{{\\ReplayRiskOnlyDeltaAbsPP}}{{{delta('B_risk_only', 'B_comm', absolute=True)}}}",
        f"\\providecommand{{\\ReplayRiskOnlyPText}}{{{p_text('B_risk_only', 'B_comm')}}}",
        f"\\providecommand{{\\ReplayWillingDirectDeltaPP}}{{{delta('A_w1_direct', 'A_w0_direct')}}}",
        f"\\providecommand{{\\ReplayWillingDirectPText}}{{{p_text('A_w1_direct', 'A_w0_direct')}}}",
        f"\\providecommand{{\\ReplayWillingCodedDeltaPP}}{{{delta('A_w1_coded', 'A_w0_coded')}}}",
        f"\\providecommand{{\\ReplayWillingCodedPText}}{{{p_text('A_w1_coded', 'A_w0_coded')}}}",
        f"\\providecommand{{\\ReplayDirectWithWillingDeltaPP}}{{{delta('A_w1_direct', 'A_w1_coded')}}}",
        f"\\providecommand{{\\ReplayDirectWithWillingPText}}{{{p_text('A_w1_direct', 'A_w1_coded')}}}",
        f"\\providecommand{{\\ReplayDirectNoWillingDeltaPP}}{{{delta('A_w0_direct', 'A_w0_coded')}}}",
        f"\\providecommand{{\\ReplayDirectNoWillingPText}}{{{p_text('A_w0_direct', 'A_w0_coded')}}}",
        f"\\providecommand{{\\ReplaySurvAllStayDeltaPP}}{{{imputed_delta('B_surv', 'B_comm', 'stay')}}}",
        f"\\providecommand{{\\ReplaySurvAllJoinDeltaPP}}{{{imputed_delta('B_surv', 'B_comm', 'join')}}}",
        f"\\providecommand{{\\ReplayRiskOnlyAllStayDeltaPP}}{{{imputed_delta('B_risk_only', 'B_comm', 'stay')}}}",
        f"\\providecommand{{\\ReplayRiskOnlyAllJoinDeltaPP}}{{{imputed_delta('B_risk_only', 'B_comm', 'join')}}}",
        f"\\providecommand{{\\ReplayWillingDirectAllStayDeltaPP}}{{{imputed_delta('A_w1_direct', 'A_w0_direct', 'stay')}}}",
        f"\\providecommand{{\\ReplayWillingDirectAllJoinDeltaPP}}{{{imputed_delta('A_w1_direct', 'A_w0_direct', 'join')}}}",
        f"\\providecommand{{\\ReplayWillingCodedAllStayDeltaPP}}{{{imputed_delta('A_w1_coded', 'A_w0_coded', 'stay')}}}",
        f"\\providecommand{{\\ReplayWillingCodedAllJoinDeltaPP}}{{{imputed_delta('A_w1_coded', 'A_w0_coded', 'join')}}}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table_tex(path: Path) -> None:
    tex = r"""\begin{table}[t]
\centering
\caption{Fixed-message interventions on willingness, style, and repression-risk content. Every row compares parseable decisions from fresh Llama 3.3 70B receivers on the same 500 country--period cells.}
\label{tab:replay_mechanisms}
\scriptsize
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrl}
\toprule
Contrast & Reference join & Treatment join & $\Delta$ (pp) & Exact wild-cluster $p$ \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: willingness $\times$ style interventions}} \\
Willingness stated, direct form & \ReplayNoWillingDirectMeanJoinPct & \ReplayWillingDirectMeanJoinPct & \ReplayWillingDirectDeltaPP & $p \ReplayWillingDirectPText$ \\
Willingness stated, coded form & \ReplayNoWillingCodedMeanJoinPct & \ReplayWillingCodedMeanJoinPct & \ReplayWillingCodedDeltaPP & $p \ReplayWillingCodedPText$ \\
Direct rather than coded, willingness stated & \ReplayWillingCodedMeanJoinPct & \ReplayWillingDirectMeanJoinPct & \ReplayDirectWithWillingDeltaPP & $p \ReplayDirectWithWillingPText$ \\
Direct rather than coded, willingness omitted & \ReplayNoWillingCodedMeanJoinPct & \ReplayNoWillingDirectMeanJoinPct & \ReplayDirectNoWillingDeltaPP & $p \ReplayDirectNoWillingPText$ \\
\addlinespace
\multicolumn{5}{l}{\textit{Panel B: repression-risk interventions}} \\
Surveillance replay vs. baseline & \ReplayBaselineMeanJoinPct & \ReplaySurvMeanJoinPct & \ReplaySurvDeltaPP & $p \ReplaySurvPText$ \\
Risk clause added to baseline messages & \ReplayBaselineMeanJoinPct & \ReplayRiskOnlyMeanJoinPct & \ReplayRiskOnlyDeltaPP & $p \ReplayRiskOnlyPText$ \\
Risk language stripped from surveilled messages & \ReplayBaselineMeanJoinPct & \ReplayRiskStrippedMeanJoinPct & \ReplayRiskStrippedVsBaseDeltaPP & $p \ReplayRiskStrippedVsBasePText$ \\
Risk-stripped vs. surveillance replay & \ReplaySurvMeanJoinPct & \ReplayRiskStrippedMeanJoinPct & \ReplayRiskStrippedVsSurvDeltaPP & $p \ReplayRiskStrippedVsSurvPText$ \\
\bottomrule
\end{tabular}
}
\begin{tablenotes}
\footnotesize\emph{Notes:} Each arm contains $N=\ReplayN$ matched cells in $G=\ReplayG$ country clusters; $p$-values enumerate all $2^G$ sign assignments in an exact restricted wild-cluster bootstrap. Qwen3 30B rewrote messages while holding the factual regime-strength assessment fixed; fresh Llama 3.3 70B receivers then saw the rewritten messages with a surveillance-free decision prompt. Panel A crosses an explicit claim that others are willing to act with direct or coded form. Valid-decision rates are tightly balanced in Panel A (\ReplayWillingDirectValidPct--\ReplayNoWillingCodedValidPct); coding every unparseable response uniformly as STAY or uniformly as JOIN leaves the willingness contrasts positive (direct: \ReplayWillingDirectAllStayDeltaPP/\ReplayWillingDirectAllJoinDeltaPP~pp; coded: \ReplayWillingCodedAllStayDeltaPP/\ReplayWillingCodedAllJoinDeltaPP~pp). Panel B adds a single risk clause to baseline messages or removes risk and guarded phrasing from surveilled messages. Its valid-decision rates differ (baseline \ReplayBaselineValidPct, surveillance \ReplaySurvValidPct, risk-stripped \ReplayRiskStrippedValidPct, risk-only \ReplayRiskOnlyValidPct). The surveillance replay remains negative under the same two uniform recodings (\ReplaySurvAllStayDeltaPP/\ReplaySurvAllJoinDeltaPP~pp), but the risk-only contrast does not (\ReplayRiskOnlyAllStayDeltaPP/\ReplayRiskOnlyAllJoinDeltaPP~pp). The risk-stripped rewrite also sharpens actionability. Panels A and B therefore triangulate mechanisms rather than form an additive decomposition, and the risk-only estimate is conditional on valid decisions.
\end{tablenotes}
\end{table}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")


def run_from_csv(
    csv_path: Path,
    *,
    stats_out: Path | None = None,
    table_out: Path | None = None,
) -> None:
    """Reproduce all A/B contrasts from the committed compact CSV."""
    cells, means, results, diagnostics = analyze_csv(csv_path)
    print("=== Arm means (valid join %) ===")
    for arm in CSV_ARMS:
        if arm in means:
            print(f"  {arm:18s} {means[arm]:6.2f}%  (n={len(cells[arm])})")
    print("\n=== Contrasts (country-clustered, exact wild-cluster bootstrap) ===")
    for tr, rf in CSV_CONTRASTS:
        if (tr, rf) in results:
            r = results[(tr, rf)]
            print(f"  {tr:16s} vs {rf:16s}: Δ={r['delta_pp']:+6.2f}pp  t={r['t']:+6.2f}  "
                  f"p_wild={r['p_wild_exact']:.4f}  (n={r['n_cells']}, G={r['n_clusters']})")

    if stats_out is not None:
        _write_stats_tex(stats_out, means, results, diagnostics)
    if table_out is not None:
        _write_table_tex(table_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", help="LABEL=path pairs")
    ap.add_argument("--contrast", action="append", default=[], help="TREAT:REF")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Reproduce all A/B contrasts from the compact cell-joins CSV")
    ap.add_argument("--stats-out", type=Path, default=None,
                    help="Write generated LaTeX macros for the compact CSV")
    ap.add_argument("--table-out", type=Path, default=None,
                    help="Write a generated LaTeX table for the compact CSV")
    args = ap.parse_args()

    if args.csv is not None:
        run_from_csv(args.csv, stats_out=args.stats_out, table_out=args.table_out)
        return

    arms = {}
    for a in args.arms:
        label, path = a.split("=", 1)
        arms[label] = Path(path)

    print("=== Arm cell means (valid join %, n cells) ===")
    for label, path in arms.items():
        if path.exists():
            m, n = mean_join(path)
            print(f"  {label:18s} {m:6.2f}%   (n={n})")
        else:
            print(f"  {label:18s} MISSING: {path}")

    print("\n=== Contrasts (country-clustered, exact wild-cluster bootstrap) ===")
    for c in args.contrast:
        treat, ref = c.split(":", 1)
        if treat not in arms or ref not in arms:
            print(f"  {c}: unknown arm"); continue
        if not arms[treat].exists() or not arms[ref].exists():
            print(f"  {c}: missing log"); continue
        r = contrast(arms[treat], arms[ref])
        print(f"  {treat:16s} vs {ref:16s}: Δ={r['delta_pp']:+6.2f}pp  "
              f"t={r['t']:+6.2f}  p_wild={r['p_wild_exact']:.4f}  "
              f"(n={r['n_cells']} cells, G={r['n_clusters']})")


if __name__ == "__main__":
    main()
