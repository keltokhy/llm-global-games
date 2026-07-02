#!/usr/bin/env python3
r"""
Analysis of the nested Llama 3.3 70B revision campaign:

  output/revision-nested-comm   - communication baseline (+pre-decision beliefs,
                                  messages excluded from the belief prompt)
  output/revision-nested-surv   - clean surveillance (+same belief elicitation)
  output/revision-nested-style  - codedness-induction control (style instruction,
                                  no monitoring framing)
  output/revision-nested-decoded-replay - decoded surveilled messages replayed
  output/revision-nested-raw-replay     - raw surveilled messages replayed
  output/revision-nested-surv-mild      - mild warning (routine monitoring,
                                          no consequence language)
  output/revision-nested-surv-severe    - severe warning (explicit tracing and
                                          arrest consequences)

All arms share grid and seed (10 countries x 50 periods, n=25, seed 5150), so
cells are nested by construction. Writes paper/tables/stats_nested.tex.

Usage: uv run python analysis/nested_campaign_checks.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = "meta-llama--llama-3.3-70b-instruct"
OUT = ROOT / "paper" / "tables" / "stats_nested.tex"

ARMS = {
    "comm": ROOT / "output" / "revision-nested-comm" / MODEL_DIR / "experiment_comm_log.json",
    "surv": ROOT / "output" / "revision-nested-surv" / MODEL_DIR / "experiment_comm_log.json",
    "style": ROOT / "output" / "revision-nested-style" / MODEL_DIR / "experiment_comm_log.json",
    "decoded": ROOT / "output" / "revision-nested-decoded-replay" / MODEL_DIR / "experiment_comm_log.json",
    "rawreplay": ROOT / "output" / "revision-nested-raw-replay" / MODEL_DIR / "experiment_comm_log.json",
    "mild": ROOT / "output" / "revision-nested-surv-mild" / MODEL_DIR / "experiment_comm_log.json",
    "severe": ROOT / "output" / "revision-nested-surv-severe" / MODEL_DIR / "experiment_comm_log.json",
}

CODED_PAT = re.compile(
    r"walls? (?:are|is) crack|ground (?:is )?shift|heads? down|feels? different|"
    r"weather|wind|storm|season|harvest|garden|seeds?\b|roots?\b|tide|current\b",
    re.I,
)
DIRECT_PAT = re.compile(r"regime|security forces|police|military|street|protest|uprising", re.I)

KEY = ["country", "period", "theta", "z", "benefit", "theta_star"]


def load(path: Path):
    if not path.exists():
        return None, None, None
    data = json.loads(path.read_text())
    rows, msgs, beliefs = [], [], []
    for e in data:
        rows.append(
            {
                "country": e.get("country"),
                "period": e.get("period"),
                "theta": round(float(e["theta"]), 6),
                "z": round(float(e.get("z", np.nan)), 6),
                "benefit": e.get("benefit"),
                "theta_star": e.get("theta_star"),
                "jf": e.get("join_fraction_valid", e.get("join_fraction")),
            }
        )
        for ag in e.get("agents", []):
            if ag.get("api_error"):
                continue
            msg = ag.get("message_sent") or ""
            if msg.strip():
                msgs.append(msg)
            bp = ag.get("belief_pre")
            sp = ag.get("second_order_belief_pre")
            if bp is not None or sp is not None:
                beliefs.append({"belief_pre": bp, "sob_pre": sp})
    cells = pd.DataFrame(rows).dropna(subset=["jf"]).groupby(KEY, as_index=False)["jf"].mean()
    return cells, msgs, pd.DataFrame(beliefs)


def paired(a: pd.DataFrame, b: pd.DataFrame):
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    d = m["jf_b"] - m["jf_a"]
    t, p = stats.ttest_1samp(d, 0.0)
    return len(m), float(d.mean()) * 100, float(t), float(p)


def msg_classifier_acc(a_msgs, b_msgs, *, length_only=False):
    """Held-out message-classifier accuracy using the paper's primary recipe
    (verify_paper_stats._classifier_summary): balanced subsample (seed 42), 70/30
    stratified split, tf-idf uni+bi-grams (max 5000 features, min_df 10), logistic
    regression. length_only replaces the text features with a single word-count
    feature, a control for whether separation is a message-length artifact."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    a_msgs, b_msgs = list(a_msgs), list(b_msgs)
    rng = np.random.default_rng(42)
    n = min(len(a_msgs), len(b_msgs))
    a_fit = [a_msgs[i] for i in rng.choice(len(a_msgs), n, replace=False)]
    b_fit = [b_msgs[i] for i in rng.choice(len(b_msgs), n, replace=False)]
    texts = a_fit + b_fit
    labels = [0] * n + [1] * n
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    if length_only:
        Xtr = np.array([[len(t.split())] for t in Xtr_txt], dtype=float)
        Xte = np.array([[len(t.split())] for t in Xte_txt], dtype=float)
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=10)
        Xtr = vec.fit_transform(Xtr_txt)
        Xte = vec.transform(Xte_txt)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return 100.0 * float(accuracy_score(yte, clf.predict(Xte)))


def fmt_p(p: float) -> str:
    # Bare "<" because these macros are used inside math mode ("$p \XPText$");
    # a "$<$" form would toggle out of math and render "<" as an inverted
    # exclamation mark under OT1 text encoding.
    return "<0.001" if p < 0.001 else f"= {p:.3f}"


def main() -> None:
    arms = {k: load(v) for k, v in ARMS.items()}
    comm_cells, comm_msgs, comm_bel = arms["comm"]
    if comm_cells is None:
        print("comm arm log not found; nothing to do")
        return
    lines = ["% Auto-generated by analysis/nested_campaign_checks.py -- do not edit."]

    def emit(name: str, val: str) -> None:
        lines.append(f"\\providecommand{{\\{name}}}{{{val}}}")

    emit("NestedCommMeanJoinPct", f"{comm_cells['jf'].mean()*100:.1f}\\%")

    for arm, prefix in (("surv", "NestedSurv"), ("style", "NestedStyle"),
                        ("decoded", "NestedDecoded"), ("rawreplay", "NestedRawReplay"),
                        ("mild", "NestedMild"), ("severe", "NestedSevere")):
        cells = arms[arm][0]
        if cells is None:
            print(f"  [skip] {arm}: no log yet")
            continue
        n, dpp, t, p = paired(comm_cells, cells)
        emit(f"{prefix}N", str(n))
        emit(f"{prefix}DeltaPP", f"{dpp:+.1f}")
        emit(f"{prefix}DeltaAbsPP", f"{abs(dpp):.1f}")
        emit(f"{prefix}T", f"{t:+.2f}")
        emit(f"{prefix}PText", fmt_p(p))
        emit(f"{prefix}MeanJoinPct", f"{cells['jf'].mean()*100:.1f}\\%")
        print(f"  {arm}: N={n} delta={dpp:+.1f}pp t={t:+.2f} p={p:.2g}")

    # decoded vs raw replay (within fixed-messages machinery)
    if arms["decoded"][0] is not None and arms["rawreplay"][0] is not None:
        n, dpp, t, p = paired(arms["rawreplay"][0], arms["decoded"][0])
        emit("NestedDecodeVsRawN", str(n))
        emit("NestedDecodeVsRawDeltaPP", f"{dpp:+.1f}")
        emit("NestedDecodeVsRawPText", fmt_p(p))
        print(f"  decoded-vs-raw: N={n} delta={dpp:+.1f}pp p={p:.2g}")

    # surveillance vs style (does monitoring framing add anything beyond style?)
    if arms["surv"][0] is not None and arms["style"][0] is not None:
        n, dpp, t, p = paired(arms["style"][0], arms["surv"][0])
        emit("NestedSurvVsStyleN", str(n))
        emit("NestedSurvVsStyleDeltaPP", f"{dpp:+.1f}")
        emit("NestedSurvVsStylePText", fmt_p(p))
        print(f"  surv-vs-style: N={n} delta={dpp:+.1f}pp p={p:.2g}")

    # surveillance vs style, TEXTUAL: can a classifier tell the two message
    # distributions apart? (surveillance-induced coding vs instructed codedness,
    # same model + grid + in-character mode). Length-only control rules out a
    # message-length artifact.
    if arms["surv"][1] and arms["style"][1]:
        clf_acc = msg_classifier_acc(arms["surv"][1], arms["style"][1])
        clf_len = msg_classifier_acc(arms["surv"][1], arms["style"][1], length_only=True)
        emit("NestedSurvVsStyleClassifierAcc", f"{clf_acc:.1f}")
        emit("NestedSurvVsStyleClassifierLenAcc", f"{clf_len:.1f}")
        print(f"  surv-vs-style classifier: acc={clf_acc:.1f}% (length-only {clf_len:.1f}%)")

    # warning dose gradient: mild share of the full-warning effect, severe vs mild
    if arms["mild"][0] is not None and arms["surv"][0] is not None:
        _, d_mild, _, _ = paired(comm_cells, arms["mild"][0])
        _, d_full, _, _ = paired(comm_cells, arms["surv"][0])
        if d_full != 0:
            emit("NestedMildShareOfFullPct", f"{100*d_mild/d_full:.0f}\\%")
            print(f"  mild share of full effect: {100*d_mild/d_full:.0f}%")
    if arms["mild"][0] is not None and arms["severe"][0] is not None:
        n, dpp, t, p = paired(arms["mild"][0], arms["severe"][0])
        emit("NestedSevereVsMildN", str(n))
        emit("NestedSevereVsMildDeltaPP", f"{dpp:+.1f}")
        emit("NestedSevereVsMildPText", fmt_p(p))
        print(f"  severe-vs-mild: N={n} delta={dpp:+.1f}pp p={p:.2g}")

    # message manipulation check: coded/direct rates per arm
    for arm, prefix in (("comm", "NestedComm"), ("surv", "NestedSurv"), ("style", "NestedStyle")):
        msgs = arms[arm][1]
        if not msgs:
            continue
        emit(f"{prefix}CodedPct", f"{100*sum(bool(CODED_PAT.search(t)) for t in msgs)/len(msgs):.1f}\\%")
        emit(f"{prefix}DirectPct", f"{100*sum(bool(DIRECT_PAT.search(t)) for t in msgs)/len(msgs):.1f}\\%")

    # pre-decision, messages-excluded beliefs: comm vs surv
    bs = arms["surv"][2]
    if comm_bel is not None and bs is not None and len(comm_bel) and len(bs):
        for col, tag in (("belief_pre", "NestedBel"), ("sob_pre", "NestedSOB")):
            a = comm_bel[col].dropna().astype(float)
            b = bs[col].dropna().astype(float)
            if len(a) and len(b):
                t, p = stats.ttest_ind(b, a, equal_var=False)
                emit(f"{tag}CommMean", f"{a.mean():.1f}")
                emit(f"{tag}SurvMean", f"{b.mean():.1f}")
                emit(f"{tag}DeltaPP", f"{b.mean()-a.mean():+.1f}")
                emit(f"{tag}PText", fmt_p(p))
                print(f"  {tag}: comm {a.mean():.1f} surv {b.mean():.1f} p={p:.2g}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
