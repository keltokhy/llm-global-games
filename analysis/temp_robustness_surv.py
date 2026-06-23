#!/usr/bin/env python3
r"""Issue #109: robustness of the surveillance effect to (A) decoding temperature
and (B) a structurally different monitoring framing.

Part A -- decoding temperature. For temperatures 0.3 and 1.0, contrast the matched
comm baseline and clean surveillance arms (mode=full) and report the surveillance
delta with a cluster-robust t, exact restricted wild-cluster bootstrap p, and a
95% Wald CI (same estimator as analysis/headline_cis.py). Temperature 0.7 is the
published nested headline (-8.0 pp), shown for reference.

Part B -- structural framing. The surveillance arm run with --surveillance-mode
structural (monitoring woven into the scenario instead of an appended note) is
contrasted against the nested 500-cell communication baseline (temp 0.7, seed
5150), directly comparable to the appended-note "full" headline of -8.0 pp.

For a lean, reproducible repo, the temperature and structural arms are stored as
compact per-cell CSVs (country, period, join_valid); the full message logs are
regenerable via the documented run commands. The nested comm baseline uses its
committed full log.

Writes paper/tables/stats_temprobust.tex (macros) when all arms are present.
Usage: uv run python analysis/temp_robustness_surv.py
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SLUG = "meta-llama--llama-3.3-70b-instruct"
TR = ROOT / "output" / "temp-robustness"
NESTED_COMM = ROOT / "output" / "revision-nested-comm" / SLUG / "experiment_comm_log.json"
STRUCT_SURV = ROOT / "output" / "structural-framing" / "surv-structural" / "cells.csv"
OUT = ROOT / "paper" / "tables" / "stats_temprobust.tex"


def load_cells(path: Path) -> dict[tuple, float]:
    """(country, period) -> valid join fraction, from a cells.csv or a full log JSON."""
    path = Path(path)
    if path.suffix == ".csv":
        out = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[(int(r["country"]), int(r["period"]))] = float(r["join_valid"])
        return out
    out = {}
    for e in json.loads(path.read_text()):
        dec = [a.get("decision") for a in e.get("agents", [])]
        valid = [d for d in dec if d in ("JOIN", "STAY")]
        if valid:
            out[(e["country"], e["period"])] = sum(d == "JOIN" for d in valid) / len(valid)
    return out


def _beta_se(d, gi, G):
    N = len(d); beta = float(d.mean()); e = d - beta
    sg = np.array([e[gi == k].sum() for k in range(G)])
    var = (G / (G - 1)) * float((sg ** 2).sum()) / (N ** 2)
    return beta, (float(np.sqrt(var)) if var > 0 else float("nan"))


def contrast(treat, ref):
    keys = sorted(set(treat) & set(ref))
    d = np.array([treat[k] - ref[k] for k in keys]) * 100.0
    countries = sorted({k[0] for k in keys})
    G = len(countries); gidx = {c: i for i, c in enumerate(countries)}
    gi = np.array([gidx[k[0]] for k in keys])
    beta, se = _beta_se(d, gi, G); t = beta / se
    crit = float(stats.t.ppf(0.975, G - 1))
    abs_t, count = abs(t), 0
    for signs in itertools.product((-1.0, 1.0), repeat=G):
        w = np.array(signs)[gi]
        b_s, se_s = _beta_se(w * d, gi, G)
        if se_s and abs(b_s / se_s) >= abs_t - 1e-12:
            count += 1
    return {"delta": beta, "se": se, "t": t, "G": G, "n": len(keys),
            "ci_lo": beta - crit * se, "ci_hi": beta + crit * se, "p_wild": count / 2 ** G,
            "comm_mean": float(np.mean(list(ref.values()))) * 100,
            "surv_mean": float(np.mean(list(treat.values()))) * 100}


def fmt(label, r):
    ci = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
    pw = "<0.01" if r["p_wild"] < 0.01 else f"{r['p_wild']:.3f}"
    print(f"{label:>10} {r['comm_mean']:>7.1f} {r['surv_mean']:>7.1f} {r['delta']:>9.2f} "
          f"{ci:>16} {r['t']:>8.2f} {r['G']:>3} {r['n']:>4} {pw:>8}")


def main():
    print(f"{'arm':>10} {'comm%':>7} {'surv%':>7} {'delta_pp':>9} {'95% CI':>16} "
          f"{'t_cc':>8} {'G':>3} {'N':>4} {'p_wild':>8}")
    print(f"{'0.7 full':>10} {'47.6':>7} {'39.6':>7} {'-8.0':>9} {'[-8.7, -7.4]':>16} "
          f"{'-26.91':>8} {'10':>3} {'500':>4} {'<0.01':>8}  published headline")

    macros = {}
    for temp, lo_hi in (("0.3", "Lo"), ("1.0", "Hi")):
        tag = "t" + temp.replace(".", "")[:2]
        cp = TR / f"{tag}-comm" / "cells.csv"
        sp = TR / f"{tag}-surv" / "cells.csv"
        if not (cp.exists() and sp.exists()):
            print(f"{temp:>10}  (missing arms)"); continue
        r = contrast(load_cells(sp), load_cells(cp)); fmt(f"{temp} full", r)
        macros[f"TR{lo_hi}Temp"] = temp
        macros[f"TR{lo_hi}TempDeltaPP"] = f"{r['delta']:.1f}"
        macros[f"TR{lo_hi}TempCILoPP"] = f"{r['ci_lo']:.1f}"
        macros[f"TR{lo_hi}TempCIHiPP"] = f"{r['ci_hi']:.1f}"
        macros[f"TR{lo_hi}TempN"] = str(r["n"])
        macros[f"TR{lo_hi}TempPwildText"] = "<0.01" if r["p_wild"] < 0.01 else f"{r['p_wild']:.3f}"

    if STRUCT_SURV.exists() and NESTED_COMM.exists():
        r = contrast(load_cells(STRUCT_SURV), load_cells(NESTED_COMM)); fmt("0.7 struct", r)
        macros["TRStructDeltaPP"] = f"{r['delta']:.1f}"
        macros["TRStructCILoPP"] = f"{r['ci_lo']:.1f}"
        macros["TRStructCIHiPP"] = f"{r['ci_hi']:.1f}"
        macros["TRStructTStat"] = f"{r['t']:.2f}"
        macros["TRStructN"] = str(r["n"])
        macros["TRStructPwildText"] = "<0.01" if r["p_wild"] < 0.01 else f"{r['p_wild']:.3f}"
    else:
        print(f"{'0.7 struct':>10}  (structural arm not present)")

    if {"TRLoTempDeltaPP", "TRHiTempDeltaPP", "TRStructDeltaPP"} <= macros.keys():
        lines = ["% Auto-generated by analysis/temp_robustness_surv.py (Issue #109). Do not edit by hand."]
        lines += [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
        OUT.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)} ({len(macros)} macros)")
    else:
        print("\n(not all arms present; macro file not written)")


if __name__ == "__main__":
    main()
