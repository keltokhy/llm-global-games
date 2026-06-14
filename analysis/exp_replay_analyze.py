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
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="LABEL=path pairs")
    ap.add_argument("--contrast", action="append", default=[], help="TREAT:REF")
    args = ap.parse_args()

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
