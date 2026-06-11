"""Model-free analyst baseline: TF-IDF + logistic regression as the "regime analyst".

Makes the blinding result independent of any LLM reader: a statistical
classifier trained to extract (a) the cell's coup outcome and (b) each
sender's JOIN/STAY from message text, evaluated on matched baseline vs
surveillance arms with cell-level cross-fitting (no cell appears in its own
training fold). Two training regimes:

  - in-domain:   train on baseline, test on held-out baseline vs held-out
                 surveillance cells (the regime's classifier was built on
                 yesterday's free speech)
  - matched:     train and test within the same arm (upper bound: the regime
                 retrains on surveilled traffic)

Predictions: surveillance degrades both tasks in-domain, concentrated at
theta < 0; retraining ("matched") recovers little in crisis states.

Usage: uv run python analysis/analyst_classifier_baseline.py
Writes: analysis/classifier_analyst_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_LOG = PROJECT_ROOT / "output/revision-nested-comm/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json"
SURV_LOG = PROJECT_ROOT / "output/revision-nested-surv/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json"
RESULTS_PATH = PROJECT_ROOT / "analysis" / "classifier_analyst_results.json"

SEED = 42
N_FOLDS = 5


def _usable(agent: dict) -> bool:
    msg = str(agent.get("message_sent") or "")
    if agent.get("api_error") or agent.get("decision") not in ("JOIN", "STAY") or not msg.strip():
        return False
    if len(msg) > 1500:
        return False
    return sum(c.isalpha() or c.isspace() for c in msg) / max(1, len(msg)) >= 0.75


def load() -> pd.DataFrame:
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
                    "cell": f"{e['country']}_{e['period']}",
                    "theta": e["theta"], "coup": int(bool(e.get("coup_success"))),
                    "agent_id": a["id"], "joined": int(a["decision"] == "JOIN"),
                    "text": str(a["message_sent"]),
                })
    df = pd.DataFrame(rows)
    key = ["country", "period", "agent_id"]
    both = df.groupby(key)["arm"].nunique()
    df = df.set_index(key).loc[both[both == 2].index].reset_index()
    return df


def _pipe() -> object:
    return make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3),
        LogisticRegression(max_iter=2000, C=1.0),
    )


def crossfit_scores(train_df: pd.DataFrame, test_dfs: dict[str, pd.DataFrame],
                    label_col: str) -> dict[str, pd.DataFrame]:
    """Cell-grouped cross-fitting: folds defined on cells; each test row is
    scored by the model whose training fold excluded its cell."""
    cells = train_df["cell"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_of_cell: dict[str, int] = {}
    for k, (_, te) in enumerate(gkf.split(train_df, groups=cells)):
        for c in set(cells[te]):
            fold_of_cell[c] = k
    out = {}
    for name, test_df in test_dfs.items():
        preds = np.full(len(test_df), np.nan)
        for k in range(N_FOLDS):
            tr_mask = np.array([fold_of_cell.get(c, -1) != k for c in train_df["cell"]])
            te_mask = np.array([fold_of_cell.get(c, k) == k for c in test_df["cell"]])
            if te_mask.sum() == 0:
                continue
            model = _pipe().fit(train_df["text"].values[tr_mask],
                                train_df[label_col].values[tr_mask])
            preds[te_mask] = model.predict_proba(test_df["text"].values[te_mask])[:, 1]
        out[name] = test_df.assign(pred=preds)
    return out


def _auc(df: pd.DataFrame, label: str) -> float | None:
    d = df.dropna(subset=["pred"])
    if d[label].nunique() < 2:
        return None
    return float(roc_auc_score(d[label], d["pred"]))


def cell_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sender messages to one document per cell for the coup task."""
    return (df.groupby(["arm", "cell", "theta", "coup"])
              .agg(text=("text", lambda s: " \n ".join(s)))
              .reset_index())


def main() -> None:
    df = load()
    print(f"[clf] {len(df):,} matched messages, {df.cell.nunique()} cells")
    rng_splits = {"crisis": df.theta < 0, "calm": df.theta >= 0, "all": df.theta.notna()}
    results: dict = {}

    # Task A: cell-level coup prediction (aggregate documents)
    cells = cell_level(df)
    base_c, surv_c = cells[cells.arm == "baseline"], cells[cells.arm == "surveillance"]
    scored = crossfit_scores(base_c, {"baseline": base_c, "surveillance": surv_c}, "coup")
    results["coup_in_domain"] = {
        split: {
            arm: _auc(scored[arm][
                scored[arm].theta < 0 if split == "crisis"
                else scored[arm].theta >= 0 if split == "calm"
                else scored[arm].theta.notna()], "coup")
            for arm in ("baseline", "surveillance")
        }
        for split in ("all", "crisis", "calm")
    }
    # matched (retrained) upper bound for the surveillance arm
    scored_m = crossfit_scores(surv_c, {"surveillance": surv_c}, "coup")
    results["coup_retrained_surv"] = {
        split: _auc(scored_m["surveillance"][
            scored_m["surveillance"].theta < 0 if split == "crisis"
            else scored_m["surveillance"].theta >= 0 if split == "calm"
            else scored_m["surveillance"].theta.notna()], "coup")
        for split in ("all", "crisis", "calm")
    }

    # Task B: sender-level JOIN prediction
    base_s, surv_s = df[df.arm == "baseline"], df[df.arm == "surveillance"]
    scored_s = crossfit_scores(base_s, {"baseline": base_s, "surveillance": surv_s}, "joined")
    results["join_in_domain"] = {
        split: {
            arm: _auc(scored_s[arm][
                scored_s[arm].theta < 0 if split == "crisis"
                else scored_s[arm].theta >= 0 if split == "calm"
                else scored_s[arm].theta.notna()], "joined")
            for arm in ("baseline", "surveillance")
        }
        for split in ("all", "crisis", "calm")
    }
    scored_sm = crossfit_scores(surv_s, {"surveillance": surv_s}, "joined")
    results["join_retrained_surv"] = {
        split: _auc(scored_sm["surveillance"][
            scored_sm["surveillance"].theta < 0 if split == "crisis"
            else scored_sm["surveillance"].theta >= 0 if split == "calm"
            else scored_sm["surveillance"].theta.notna()], "joined")
        for split in ("all", "crisis", "calm")
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=1))
    for task in ("coup_in_domain", "join_in_domain"):
        for split in ("all", "crisis", "calm"):
            r = results[task][split]
            retr = results[task.replace("in_domain", "retrained_surv")][split]
            b = f"{r['baseline']:.3f}" if r["baseline"] is not None else "  na "
            s = f"{r['surveillance']:.3f}" if r["surveillance"] is not None else "  na "
            t = f"{retr:.3f}" if retr is not None else "  na "
            print(f"[clf] {task:<15} {split:<7} baseline={b} surveillance={s} retrained={t}")
    print(f"[clf] results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
