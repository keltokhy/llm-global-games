"""Paper 2 deterministic analysis (no API): reproduces the locked empirics.

  1. Replay effect: surveillance messages -> naive receivers suppress joining.
  2. Illegibility refutation: text monitors separate conditions at AUC ~ 1.0.
  3. Lexicon mediation: a regime-beatability proxy mediates; hedging does not.
  4. Data-named mechanism: terms most associated with (not) joining.

Run: uv run python -m paper2.mechanism_analysis
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
LLAMA = "meta-llama--llama-3.3-70b-instruct"
PREFIX = re.compile(r"^\s*(trusted contact|neighbor|colleague|friend|associate)\s*:\s*", re.I)

# A-priori lexicons (transparent proxies; validated against an LLM judge elsewhere).
LEX = {
    # confident assertion that the regime is weak / falling (the mechanism proxy)
    "beatable": re.compile(r"\b(crack|cracks|cracking|crumbl|collaps|falling|fall|fragile|"
                           r"weak|weaken|unravel|faltering|losing control|back foot|perfect storm|"
                           r"will fall|inevitable|tipping point)\b", re.I),
    # regime is holding / still in control (suppressive)
    "holding":  re.compile(r"\b(still (?:firmly )?in control|firmly|holding|maintain|despite|"
                           r"underlying|subtle|stable|secure|grip|contained)\b", re.I),
    "hedge":    re.compile(r"\b(if|once|unless|should|depends|wait|waiting|ready to|hold back|"
                           r"see what|follow|cautious|careful|those who|if others|when others)\b", re.I),
}


def clean(m: str) -> str:
    return PREFIX.sub("", m.strip().strip('"').strip()).strip().strip('"')


def load_cells(path: Path, field: str) -> pd.DataFrame:
    """field='message_sent' (live) or 'messages_received' (replay)."""
    d = json.loads(Path(path).read_text())
    rows = []
    for i, c in enumerate(d):
        msgs: list[str] = []
        for ag in c.get("agents", []):
            if field == "message_sent":
                m = ag.get("message_sent")
                if isinstance(m, str) and clean(m):
                    msgs.append(clean(m))
            else:
                for m in ag.get("messages_received", []) or []:
                    if isinstance(m, str) and clean(m) and clean(m) not in msgs:
                        msgs.append(clean(m))
        if not msgs:
            continue
        row = {"cell": i, "theta": c["theta"], "join": c["join_fraction"],
               "text": " \n ".join(msgs), "n_msg": len(msgs)}
        for name, rx in LEX.items():
            row[name] = float(np.mean([len(rx.findall(m)) for m in msgs]))
        rows.append(row)
    return pd.DataFrame(rows)


def p(path: str) -> Path:
    return ROOT / "output" / path


# ---- 1. Replay effect -------------------------------------------------------
def replay_effect() -> None:
    print("\n## 1. Replay: surveillance messages -> naive receivers")
    surv = load_cells(p(f"expB-surv-replay/{LLAMA}/experiment_comm_log.json"), "messages_received")
    comm = load_cells(p(f"expB-comm-replay/{LLAMA}/experiment_comm_log.json"), "messages_received")
    m = surv.merge(comm, on="cell", suffixes=("_s", "_c"))
    js, jc = m.join_s.values, m.join_c.values
    t, pv = stats.ttest_rel(js, jc)
    d = js - jc
    ci = 1.96 * d.std() / np.sqrt(len(d))
    print(f"   join: surv={js.mean():.3f}  comm={jc.mean():.3f}  diff={d.mean():+.3f}pp-frac  "
          f"paired t={t:.1f} p={pv:.1e}  95%CI[{d.mean()-ci:+.3f},{d.mean()+ci:+.3f}]  n={len(d)}")


# ---- 2. Illegibility refutation --------------------------------------------
def _msgs(path: str, field: str) -> list[str]:
    df = load_cells(p(path), field)
    out: list[str] = []
    for t in df.text:
        out += [s for s in t.split(" \n ") if s]
    return out


def illegibility() -> None:
    print("\n## 2. Illegibility refutation: text-monitor separability (5-fold AUC; 0.5=illegible)")
    surv = _msgs(f"revision-nested-surv-severe/{LLAMA}/experiment_comm_log.json", "message_sent")
    style = _msgs(f"revision-nested-style/{LLAMA}/experiment_comm_log.json", "message_sent")
    w1 = _msgs("expA-w1-coded/w1_coded_log.json", "message_sent")
    w0 = _msgs("expA-w0-coded/w0_coded_log.json", "message_sent")
    for a, b, lab in [(surv, style, "surveillance vs codedness-control"),
                      (w1, w0, "coded+surv vs coded+no-surv (apples-to-apples)")]:
        X, y = a + b, np.r_[np.zeros(len(a)), np.ones(len(b))]
        pipe = Pipeline([("t", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True)),
                         ("c", LogisticRegression(max_iter=2000, C=4.0))])
        auc = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
        print(f"   {lab:48s} AUC={auc:.3f}")


# ---- 3. Lexicon mediation (within matched cells) ----------------------------
def lexicon_mediation() -> None:
    print("\n## 3. Lexicon mediation within matched cells (Delta-join ~ Delta-feature)")
    surv = load_cells(p(f"expB-surv-replay/{LLAMA}/experiment_comm_log.json"), "messages_received")
    comm = load_cells(p(f"expB-comm-replay/{LLAMA}/experiment_comm_log.json"), "messages_received")
    m = surv.merge(comm, on="cell", suffixes=("_s", "_c"))
    m["dj"] = m.join_s - m.join_c
    print(f"   (n={len(m)} matched cells; surveillance lowers join by {m.dj.mean():+.3f})")
    for f in ["beatable", "holding", "hedge"]:
        m["df"] = m[f"{f}_s"] - m[f"{f}_c"]
        fit = smf.ols("dj ~ df", m).fit()
        print(f"   {f:10s} d(surv-comm)={m['df'].mean():+.3f}  slope={fit.params['df']:+.4f}  p={fit.pvalues['df']:.1e}")


# ---- 4. Data-named mechanism -----------------------------------------------
def data_named() -> None:
    print("\n## 4. Data-named mechanism: message text -> theta-residualized join")
    surv = load_cells(p(f"expB-surv-replay/{LLAMA}/experiment_comm_log.json"), "messages_received")
    comm = load_cells(p(f"expB-comm-replay/{LLAMA}/experiment_comm_log.json"), "messages_received")
    pool = pd.concat([surv, comm], ignore_index=True)
    base = smf.ols("join ~ theta + I(theta**2)", pool).fit()
    resid = (pool["join"] - base.predict(pool)).values
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000, sublinear_tf=True)
    X = vec.fit_transform(pool["text"])
    pred = cross_val_predict(Ridge(alpha=2.0), X, resid, cv=KFold(5, shuffle=True, random_state=0))
    print(f"   theta-only R2={base.rsquared:.3f}; text->residual join out-of-sample R2={r2_score(resid, pred):.3f}")
    names = np.array(vec.get_feature_names_out())
    coef = Ridge(alpha=2.0).fit(X, resid).coef_
    print("   mobilizing (->join):", [names[i] for i in np.argsort(coef)[-12:][::-1]])
    print("   suppressive (->stay):", [names[i] for i in np.argsort(coef)[:12]])


if __name__ == "__main__":
    replay_effect()
    illegibility()
    lexicon_mediation()
    data_named()
