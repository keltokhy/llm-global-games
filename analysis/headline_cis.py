#!/usr/bin/env python3
r"""Country-clustered 95% CIs for the two full-scale surveillance estimates.

Issue #105: surface country-clustered confidence intervals in the main text for
both full-scale surveillance contrasts.

  - Primary (Mistral Small Creative): matched-cell delta, clustered on the 5
    countries. The point estimate, cluster-robust t, and G are already in
    analysis/verified_stats.json (prompt_isolation.surveillance.matched.
    country_clustered); the CI is the Wald cluster-robust interval
    delta +/- t_{0.975, G-1} * SE, with SE = |delta / t|.

  - Confirmatory (Llama 3.3 70B nested 500-cell): the headline NestedSurv delta
    (-8.0 pp) is reported with a paired t only, so this script recomputes the
    country-clustered SE from the raw nested comm/surv cell logs (clustering on
    the 10 countries) using the same estimator as exp_replay_analyze.py, and
    forms the Wald cluster-robust interval. It also reports the exact restricted
    wild-cluster bootstrap p (2^10 enumeration) as corroboration.

Validation gates (abort if violated):
  - recomputed Llama nested delta rounds to the published \NestedSurvDeltaPP.
  - recomputed Mistral SE reproduces the published cluster-robust t.

Writes paper/tables/stats_headline_cis.tex.

Usage: uv run python analysis/headline_cis.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = "meta-llama--llama-3.3-70b-instruct"
COMM = ROOT / "output" / "revision-nested-comm" / MODEL_DIR / "experiment_comm_log.json"
SURV = ROOT / "output" / "revision-nested-surv" / MODEL_DIR / "experiment_comm_log.json"
VERIFIED = ROOT / "analysis" / "verified_stats.json"
OUT = ROOT / "paper" / "tables" / "stats_headline_cis.tex"

# Published headline values these CIs attach to (sanity targets).
NESTED_DELTA_TARGET = -8.0   # \NestedSurvDeltaPP
MISTRAL_T_TARGET = -13.90    # \PromptIsoMistralCountryClustT


def load_cells(path: Path) -> dict[tuple, float]:
    """(country, period) -> valid join fraction, matching exp_replay_analyze."""
    out = {}
    for e in json.loads(Path(path).read_text()):
        dec = [a.get("decision") for a in e.get("agents", [])]
        valid = [d for d in dec if d in ("JOIN", "STAY")]
        if valid:
            out[(e["country"], e["period"])] = sum(d == "JOIN" for d in valid) / len(valid)
    return out


def _beta_se(d: np.ndarray, gi: np.ndarray, G: int):
    """Cluster-robust mean and SE (K=1), identical to exp_replay_analyze._beta_se."""
    N = len(d)
    beta = float(d.mean())
    e = d - beta
    sg = np.array([e[gi == k].sum() for k in range(G)])
    var = (G / (G - 1)) * float((sg ** 2).sum()) / (N ** 2)
    return beta, (float(np.sqrt(var)) if var > 0 else float("nan"))


def cluster_contrast(treat: dict, ref: dict) -> dict:
    keys = sorted(set(treat) & set(ref))
    d = np.array([treat[k] - ref[k] for k in keys]) * 100.0
    countries = [k[0] for k in keys]
    groups = sorted(set(countries))
    G = len(groups)
    gidx = {c: i for i, c in enumerate(groups)}
    gi = np.array([gidx[c] for c in countries])
    beta, se = _beta_se(d, gi, G)
    t = beta / se
    dof = G - 1
    crit = float(stats.t.ppf(0.975, dof))
    # exact restricted wild-cluster bootstrap p (full 2^G sign enumeration)
    abs_t, count, total = abs(t), 0, 0
    for signs in itertools.product((-1.0, 1.0), repeat=G):
        w = np.array(signs)[gi]
        b_s, se_s = _beta_se(w * d, gi, G)
        total += 1
        if se_s and abs(b_s / se_s) >= abs_t - 1e-12:
            count += 1
    return {
        "delta": beta, "se": se, "t": t, "dof": dof, "G": G, "n_cells": len(keys),
        "ci_lo": beta - crit * se, "ci_hi": beta + crit * se,
        "p_wild_exact": count / total,
    }


def main() -> None:
    # ---- Llama nested (confirmatory), recomputed from raw cells ----
    comm, surv = load_cells(COMM), load_cells(SURV)
    nested = cluster_contrast(surv, comm)
    assert round(nested["delta"], 1) == NESTED_DELTA_TARGET, (
        f"nested delta {nested['delta']:.2f} != published {NESTED_DELTA_TARGET}")

    # ---- Mistral (primary/discovery), from stored country-clustered stats ----
    vs = json.loads(VERIFIED.read_text())
    cc = vs["prompt_isolation"]["surveillance"]["Mistral Small Creative"]["matched"]["country_clustered"]
    m_delta, m_t, m_G = cc["delta_pp"], cc["t_stat"], cc["n_countries"]
    assert round(m_t, 2) == MISTRAL_T_TARGET, f"Mistral t {m_t} != published {MISTRAL_T_TARGET}"
    m_se = abs(m_delta / m_t)
    m_crit = float(stats.t.ppf(0.975, m_G - 1))
    m_lo, m_hi = m_delta - m_crit * m_se, m_delta + m_crit * m_se

    print("=== Validation / CIs ===")
    print(f"Mistral primary: delta={m_delta:.2f} t={m_t:.2f} G={m_G} SE={m_se:.3f} "
          f"-> 95% CI [{m_lo:.1f}, {m_hi:.1f}] pp")
    print(f"Llama nested:    delta={nested['delta']:.2f} t_cc={nested['t']:.2f} "
          f"G={nested['G']} SE={nested['se']:.3f} dof={nested['dof']} "
          f"p_wild={nested['p_wild_exact']:.4f} -> 95% CI "
          f"[{nested['ci_lo']:.1f}, {nested['ci_hi']:.1f}] pp  (n={nested['n_cells']})")

    def pp(x):
        return f"{x:.1f}"

    lines = [
        "% Auto-generated by analysis/headline_cis.py (Issue #105). Do not edit by hand.",
        "% Country-clustered 95% Wald CIs for the two full-scale surveillance estimates.",
        f"\\providecommand{{\\PromptIsoMistralCCCILoPP}}{{{pp(m_lo)}}}",
        f"\\providecommand{{\\PromptIsoMistralCCCIHiPP}}{{{pp(m_hi)}}}",
        f"\\providecommand{{\\NestedSurvCCTStat}}{{{nested['t']:.2f}}}",
        f"\\providecommand{{\\NestedSurvCCDof}}{{{nested['dof']}}}",
        f"\\providecommand{{\\NestedSurvCCG}}{{{nested['G']}}}",
        f"\\providecommand{{\\NestedSurvCCPwildText}}{{{('<0.01' if nested['p_wild_exact'] < 0.01 else f'{nested['p_wild_exact']:.3f}')}}}",
        f"\\providecommand{{\\NestedSurvCILoPP}}{{{pp(nested['ci_lo'])}}}",
        f"\\providecommand{{\\NestedSurvCIHiPP}}{{{pp(nested['ci_hi'])}}}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
