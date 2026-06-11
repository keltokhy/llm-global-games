"""Direct text test of Lemma 1 (cutoff coding) for "The Intelligence Cost of Surveillance".

Lemma 1 says coding is selected on the sender's private signal: under
surveillance, only citizens with incriminating things to say change how they
write. Two testable implications in the message text itself:

  (T1) Across-arm divergence of message features is concentrated among
       LOW-signal senders (anti-regime z-scores) and nearly absent among
       high-signal senders — divergence by sender z bin, the sharpest test.
  (T2) Aggregated to cells, divergence is decreasing in theta — the
       crisis-concentration corollary.

Features (simple, replicable lexicons; counts per 100 words):
  action cues, direct regime terms, hedges, first-person commitment,
  plus message length. Divergence per bin = |standardized mean difference|
  averaged over features, and a two-sample text-feature classifier AUC
  (logistic on the feature vector; baseline vs surveillance).

Usage: uv run python analysis/lemma1_text_divergence.py
Writes: analysis/lemma1_results.json, output/analyst-pilot/figures/fig_lemma1_divergence.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import C_PURE, C_SURV, apply_style  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_LOG = PROJECT_ROOT / "output/revision-nested-comm/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json"
SURV_LOG = PROJECT_ROOT / "output/revision-nested-surv/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json"
RESULTS_PATH = PROJECT_ROOT / "analysis" / "lemma1_results.json"
FIG_PATH = PROJECT_ROOT / "output" / "analyst-pilot" / "figures" / "fig_lemma1_divergence.png"

N_BINS = 8
N_BOOT = 2000
SEED = 42

LEXICONS = {
    "action_cues": [
        "join", "act", "move", "rise", "uprising", "fight", "protest", "streets",
        "ready", "prepared", "mobilize", "organize", "tonight", "now is the time",
        "take action", "stand up", "march",
    ],
    "regime_direct": [
        "regime", "government", "fall", "falls", "falling", "collapse", "collapsing",
        "overthrow", "topple", "crumble", "crumbling", "dictator", "security forces",
        "military", "loyalists", "crackdown",
    ],
    "hedges": [
        "maybe", "perhaps", "might", "could be", "unclear", "uncertain", "hard to say",
        "not sure", "possibly", "seems", "appears", "i think", "i guess", "who knows",
        "we'll see", "time will tell",
    ],
    "commitment": [
        "i will", "i'll", "i'm ready", "i am ready", "count me", "i'm in", "i am in",
        "we will", "we'll", "let's", "we should", "we must", "i plan", "we need to",
    ],
}


def _features(msg: str) -> dict:
    text = re.sub(r"\s+", " ", str(msg or "")).strip().lower()
    n_words = max(1, len(text.split()))
    out = {"length_words": float(len(text.split()))}
    for name, terms in LEXICONS.items():
        count = sum(text.count(t) for t in terms)
        out[name] = 100.0 * count / n_words
    return out


def _usable(agent: dict) -> bool:
    msg = str(agent.get("message_sent") or "")
    if agent.get("api_error") or agent.get("decision") not in ("JOIN", "STAY") or not msg.strip():
        return False
    if len(msg) > 1500:
        return False
    return sum(c.isalpha() or c.isspace() for c in msg) / max(1, len(msg)) >= 0.75


def load_messages() -> pd.DataFrame:
    rows = []
    for arm, path in (("baseline", BASE_LOG), ("surveillance", SURV_LOG)):
        with open(path) as f:
            entries = json.load(f)
        for e in entries:
            for a in e["agents"]:
                if not _usable(a):
                    continue
                rows.append({
                    "arm": arm, "country": e["country"], "period": e["period"],
                    "theta": e["theta"], "agent_id": a["id"],
                    "z_score": a.get("z_score"), "decision": a["decision"],
                    **_features(a["message_sent"]),
                })
    df = pd.DataFrame(rows)
    # Keep only sender-cells present in BOTH arms (matched senders).
    key_cols = ["country", "period", "agent_id"]
    both = df.groupby(key_cols)["arm"].nunique()
    matched = both[both == 2].index
    df = df.set_index(key_cols).loc[matched].reset_index()
    return df


FEATURES = ["action_cues", "regime_direct", "hedges", "commitment", "length_words"]


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (pooled sd)."""
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def _classifier_auc(xa: np.ndarray, xb: np.ndarray, seed: int = SEED) -> float:
    """AUC of a logistic classifier separating the arms on the feature vector
    (5-fold cross-validated). 0.5 = arms indistinguishable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([xa, xb])
    y = np.concatenate([np.zeros(len(xa)), np.ones(len(xb))])
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, preds))


def divergence_by_bin(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    """Per-bin divergence between arms: mean |SMD| over features + classifier AUC."""
    vals = df[bin_col].values
    edges = np.quantile(vals, np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    df = df.assign(_bin=pd.cut(df[bin_col], edges, labels=False))
    rows = []
    rng = np.random.default_rng(SEED)
    for b, g in df.groupby("_bin"):
        ga = g[g.arm == "baseline"]
        gb = g[g.arm == "surveillance"]
        if len(ga) < 30 or len(gb) < 30:
            continue
        smds = {f: _smd(ga[f].values, gb[f].values) for f in FEATURES}
        mean_abs_smd = float(np.mean([abs(v) for v in smds.values()]))
        # bootstrap CI on mean |SMD|
        boots = []
        for _ in range(N_BOOT):
            ia = rng.integers(0, len(ga), len(ga))
            ib = rng.integers(0, len(gb), len(gb))
            boots.append(np.mean([
                abs(_smd(ga[f].values[ia], gb[f].values[ib])) for f in FEATURES
            ]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "bin": int(b),
            "bin_center": float(g[bin_col].median()),
            "n_per_arm": int(min(len(ga), len(gb))),
            "mean_abs_smd": mean_abs_smd,
            "smd_ci": [float(lo), float(hi)],
            "classifier_auc": _classifier_auc(ga[FEATURES].values, gb[FEATURES].values),
            **{f"smd_{k}": v for k, v in smds.items()},
        })
    return pd.DataFrame(rows)


def monotonicity_test(binned: pd.DataFrame, x: str = "bin_center",
                      y: str = "mean_abs_smd") -> dict:
    """Spearman correlation of divergence with the bin variable (predicted negative:
    divergence decreasing in z / theta)."""
    from scipy import stats
    rho = stats.spearmanr(binned[x], binned[y])
    return {"spearman_rho": float(rho.statistic), "p": float(rho.pvalue)}


def main() -> None:
    df = load_messages()
    print(f"[lemma1] {len(df):,} matched messages "
          f"({(df.arm == 'baseline').sum():,} baseline / {(df.arm == 'surveillance').sum():,} surveillance)")

    by_z = divergence_by_bin(df, "z_score")
    by_theta = divergence_by_bin(df, "theta")
    res = {
        "n_messages": int(len(df)),
        "features": FEATURES,
        "by_sender_z": by_z.to_dict(orient="records"),
        "by_theta": by_theta.to_dict(orient="records"),
        "monotonicity_z": monotonicity_test(by_z),
        "monotonicity_theta": monotonicity_test(by_theta),
    }
    RESULTS_PATH.write_text(json.dumps(res, indent=1))

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, binned, xlabel, title in (
        (axes[0], by_z, "sender signal z (anti-regime → pro-regime)", "By sender signal (T1)"),
        (axes[1], by_theta, r"regime strength $\theta$", "By state (T2)"),
    ):
        ci = np.array([c for c in binned["smd_ci"]])
        ax.errorbar(binned["bin_center"], binned["mean_abs_smd"],
                    yerr=[binned["mean_abs_smd"] - ci[:, 0], ci[:, 1] - binned["mean_abs_smd"]],
                    fmt="o-", color=C_SURV, lw=1.2, ms=3.5, capsize=2)
        ax2 = ax.twinx()
        ax2.plot(binned["bin_center"], binned["classifier_auc"], "s--",
                 color=C_PURE, lw=1.0, ms=3, alpha=0.7)
        ax2.set_ylabel("arm-classifier AUC", fontsize=7, color=C_PURE)
        ax2.set_ylim(0.45, 1.0)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("mean |SMD| across features", fontsize=8, color=C_SURV)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Message divergence (baseline vs surveillance) is selected on the sender",
                 fontsize=9.5)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)

    print(f"[lemma1] monotonicity in sender z: rho={res['monotonicity_z']['spearman_rho']:.3f} "
          f"p={res['monotonicity_z']['p']:.4f}")
    print(f"[lemma1] monotonicity in theta:    rho={res['monotonicity_theta']['spearman_rho']:.3f} "
          f"p={res['monotonicity_theta']['p']:.4f}")
    print(f"[lemma1] results -> {RESULTS_PATH}\n[lemma1] figure  -> {FIG_PATH}")


if __name__ == "__main__":
    main()
