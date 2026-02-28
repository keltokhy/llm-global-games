"""
Text classifier baselines for LLM global game decisions.

Trains three classifiers on agent briefing text / slider features to predict
JOIN vs STAY, then tests cross-treatment generalization.  The key finding:
text classifiers trained on pure-game data cannot reproduce the surveillance
belief-action wedge, because the briefing text is identical across treatments
-- only the system prompt changes.

Classifiers:
  1. Bag-of-words logistic regression (TF-IDF on rendered briefing text)
  2. All-slider logistic regression (direction, clarity, coordination + interactions)
  3. Keyphrase + sentiment model (action/caution word counts, punctuation features)

Usage: uv run python analysis/classifier_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_PATH = ANALYSIS_DIR / "classifier_results.json"

MODEL_SLUG = "mistralai--mistral-small-creative"
MODEL_DIR = OUTPUT_ROOT / MODEL_SLUG
SURV_DIR = OUTPUT_ROOT / "surveillance" / MODEL_SLUG

# ── Keyphrase lists ───────────────────────────────────────────────────
ACTION_WORDS = [
    "act", "join", "rise", "fight", "resist", "unite", "together",
    "now", "ready", "opportunity", "collapse", "crumbling", "eroding",
]
CAUTION_WORDS = [
    "careful", "stable", "patience", "steady", "risk", "danger",
    "wait", "uncertain", "cautious",
]


# ── Data loading ──────────────────────────────────────────────────────

def _load_log(path: Path) -> list[dict]:
    """Load an experiment log JSON file."""
    if not path.exists():
        print(f"WARNING: {path} not found, skipping.")
        return []
    with open(path) as f:
        return json.load(f)


def _load_calibrated_params() -> dict:
    """Load calibrated briefing generator parameters."""
    p = MODEL_DIR / f"calibrated_params_{MODEL_SLUG}.json"
    if not p.exists():
        print(f"WARNING: calibrated params not found at {p}")
        return {}
    with open(p) as f:
        return json.load(f)


def _build_briefing_generator(calibrated: dict):
    """Build a BriefingGenerator from calibrated parameters."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from agent_based_simulation.briefing import BriefingGenerator

    return BriefingGenerator(
        cutoff_center=calibrated.get("cutoff_center", 0.0),
        clarity_width=calibrated.get("clarity_width", 1.0),
        direction_slope=calibrated.get("direction_slope", 0.8),
        coordination_slope=calibrated.get("coordination_slope", 0.6),
        dissent_floor=calibrated.get("dissent_floor", 0.25),
        mixed_cue_clarity=calibrated.get("mixed_cue_clarity", 0.5),
        bottomline_cuts=calibrated.get("bottomline_cuts"),
        unclear_cuts=calibrated.get("unclear_cuts"),
        coordination_cuts=calibrated.get("coordination_cuts"),
        coordination_blend_prob=calibrated.get("coordination_blend_prob", 0.6),
        language_variant=calibrated.get("language_variant", "baseline"),
        seed=5150,  # default experiment seed
    )


def _extract_agent_rows(log_data: list[dict], briefing_gen=None) -> pd.DataFrame:
    """Flatten log JSON into a per-agent DataFrame.

    Parameters
    ----------
    log_data : list[dict]
        Loaded experiment log (list of period dicts).
    briefing_gen : BriefingGenerator, optional
        If provided, regenerate briefing text from z_score + agent_id + period.

    Returns
    -------
    pd.DataFrame with columns: z_score, direction, clarity, coordination,
        decision, label, briefing_text (if briefing_gen), treatment, theta,
        period, country, agent_id.
    """
    rows = []
    for period_dict in log_data:
        period = period_dict["period"]
        country = period_dict["country"]
        theta = period_dict["theta"]
        treatment = period_dict["treatment"]
        for agent in period_dict["agents"]:
            if agent.get("api_error", False):
                continue
            decision = agent["decision"]
            if decision not in ("JOIN", "STAY"):
                continue

            row = {
                "agent_id": agent["id"],
                "period": period,
                "country": country,
                "theta": theta,
                "treatment": treatment,
                "z_score": agent["z_score"],
                "direction": agent["direction"],
                "clarity": agent["clarity"],
                "coordination": agent["coordination"],
                "decision": decision,
                "label": 1 if decision == "JOIN" else 0,
                "reasoning": agent.get("reasoning", ""),
            }

            if briefing_gen is not None:
                briefing = briefing_gen.generate(
                    agent["z_score"], agent["id"], period,
                )
                row["briefing_text"] = briefing.render()

            rows.append(row)

    return pd.DataFrame(rows)


# ── Feature extraction ────────────────────────────────────────────────

def _slider_features(df: pd.DataFrame) -> np.ndarray:
    """Direction, clarity, coordination + all pairwise interactions."""
    base = df[["direction", "clarity", "coordination"]].values
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    return poly.fit_transform(base)


def _keyphrase_features(df: pd.DataFrame) -> np.ndarray:
    """Action/caution word counts, message length, punctuation counts."""
    text_col = df["briefing_text"] if "briefing_text" in df.columns else df["reasoning"]

    features = []
    for text in text_col:
        text_lower = text.lower()
        words = text_lower.split()
        action_count = sum(1 for w in words if any(kw in w for kw in ACTION_WORDS))
        caution_count = sum(1 for w in words if any(kw in w for kw in CAUTION_WORDS))
        features.append([
            action_count,
            caution_count,
            len(text),
            text.count("!"),
            text.count("?"),
        ])
    return np.array(features, dtype=float)


# ── Evaluation ────────────────────────────────────────────────────────

def _cv_evaluate(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                 random_state: int = 42) -> dict:
    """5-fold stratified CV of LogisticRegression.  Returns accuracy and AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, aucs = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        probs = clf.predict_proba(X[test_idx])[:, 1]
        accs.append(accuracy_score(y[test_idx], preds))
        aucs.append(roc_auc_score(y[test_idx], probs))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def _train_full_and_predict(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            random_state: int = 42) -> dict:
    """Train on full training set, evaluate on held-out test set."""
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)),
        "predicted_join_rate": float(np.mean(preds)),
        "actual_join_rate": float(np.mean(y_test)),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Classifier baselines for LLM global game decisions")
    print("=" * 72)

    # Load calibrated params and build briefing generator
    calibrated = _load_calibrated_params()
    if not calibrated:
        print("ERROR: Cannot proceed without calibrated parameters.")
        sys.exit(1)
    briefing_gen = _build_briefing_generator(calibrated)

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLoading experiment logs...")

    pure_log = _load_log(MODEL_DIR / "experiment_pure_log.json")
    comm_log = _load_log(MODEL_DIR / "experiment_comm_log.json")
    surv_log = _load_log(SURV_DIR / "experiment_comm_log.json")

    if not pure_log:
        print("ERROR: No pure log data found.")
        sys.exit(1)

    print(f"  Pure log:         {len(pure_log):>5} periods")
    print(f"  Comm log:         {len(comm_log):>5} periods")
    print(f"  Surveillance log: {len(surv_log):>5} periods")

    # Extract per-agent DataFrames (with briefing text regeneration)
    print("\nExtracting agent data and regenerating briefing text...")
    df_pure = _extract_agent_rows(pure_log, briefing_gen)
    df_comm = _extract_agent_rows(comm_log, briefing_gen) if comm_log else pd.DataFrame()
    df_surv = _extract_agent_rows(surv_log, briefing_gen) if surv_log else pd.DataFrame()

    print(f"  Pure agents:         {len(df_pure):>6} (JOIN rate: {df_pure['label'].mean():.3f})")
    if len(df_comm):
        print(f"  Comm agents:         {len(df_comm):>6} (JOIN rate: {df_comm['label'].mean():.3f})")
    if len(df_surv):
        print(f"  Surveillance agents: {len(df_surv):>6} (JOIN rate: {df_surv['label'].mean():.3f})")

    results = {}
    y_pure = df_pure["label"].values

    # ==================================================================
    # Classifier 1: Bag-of-words logistic regression (TF-IDF)
    # ==================================================================
    print("\n" + "-" * 72)
    print("Classifier 1: Bag-of-words logistic regression (TF-IDF)")
    print("-" * 72)

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english",
                            ngram_range=(1, 2), min_df=5)
    X_bow_pure = tfidf.fit_transform(df_pure["briefing_text"])

    cv_bow = _cv_evaluate(X_bow_pure, y_pure)
    print(f"  5-fold CV accuracy: {cv_bow['accuracy_mean']:.3f} +/- {cv_bow['accuracy_std']:.3f}")
    print(f"  5-fold CV AUC:      {cv_bow['auc_mean']:.3f} +/- {cv_bow['auc_std']:.3f}")
    results["bow_tfidf"] = {"cv_pure": cv_bow}

    # Cross-treatment: train on pure, test on comm
    if len(df_comm):
        X_bow_comm = tfidf.transform(df_comm["briefing_text"])
        y_comm = df_comm["label"].values
        cross_comm = _train_full_and_predict(X_bow_pure, y_pure, X_bow_comm, y_comm)
        print(f"  Cross-treatment (pure -> comm):")
        print(f"    Accuracy: {cross_comm['accuracy']:.3f}   AUC: {cross_comm['auc']:.3f}")
        print(f"    Predicted JOIN rate: {cross_comm['predicted_join_rate']:.3f}   "
              f"Actual: {cross_comm['actual_join_rate']:.3f}")
        results["bow_tfidf"]["cross_pure_to_comm"] = cross_comm

    # Cross-treatment: train on pure, test on surveillance
    if len(df_surv):
        X_bow_surv = tfidf.transform(df_surv["briefing_text"])
        y_surv = df_surv["label"].values
        cross_surv = _train_full_and_predict(X_bow_pure, y_pure, X_bow_surv, y_surv)
        print(f"  Cross-treatment (pure -> surveillance):")
        print(f"    Accuracy: {cross_surv['accuracy']:.3f}   AUC: {cross_surv['auc']:.3f}")
        print(f"    Predicted JOIN rate: {cross_surv['predicted_join_rate']:.3f}   "
              f"Actual: {cross_surv['actual_join_rate']:.3f}")
        delta = cross_surv["predicted_join_rate"] - cross_surv["actual_join_rate"]
        print(f"    Wedge (predicted - actual): {delta:+.3f} "
              f"({'classifier misses chilling effect' if abs(delta) > 0.05 else 'similar'})")
        results["bow_tfidf"]["cross_pure_to_surv"] = cross_surv

    # ==================================================================
    # Classifier 2: All-slider logistic regression
    # ==================================================================
    print("\n" + "-" * 72)
    print("Classifier 2: All-slider logistic regression (direction, clarity, coordination + interactions)")
    print("-" * 72)

    X_slider_pure = _slider_features(df_pure)

    cv_slider = _cv_evaluate(X_slider_pure, y_pure)
    print(f"  5-fold CV accuracy: {cv_slider['accuracy_mean']:.3f} +/- {cv_slider['accuracy_std']:.3f}")
    print(f"  5-fold CV AUC:      {cv_slider['auc_mean']:.3f} +/- {cv_slider['auc_std']:.3f}")
    results["slider_logistic"] = {"cv_pure": cv_slider}

    if len(df_comm):
        X_slider_comm = _slider_features(df_comm)
        y_comm = df_comm["label"].values
        cross_comm = _train_full_and_predict(X_slider_pure, y_pure, X_slider_comm, y_comm)
        print(f"  Cross-treatment (pure -> comm):")
        print(f"    Accuracy: {cross_comm['accuracy']:.3f}   AUC: {cross_comm['auc']:.3f}")
        print(f"    Predicted JOIN rate: {cross_comm['predicted_join_rate']:.3f}   "
              f"Actual: {cross_comm['actual_join_rate']:.3f}")
        results["slider_logistic"]["cross_pure_to_comm"] = cross_comm

    if len(df_surv):
        X_slider_surv = _slider_features(df_surv)
        y_surv = df_surv["label"].values
        cross_surv = _train_full_and_predict(X_slider_pure, y_pure, X_slider_surv, y_surv)
        print(f"  Cross-treatment (pure -> surveillance):")
        print(f"    Accuracy: {cross_surv['accuracy']:.3f}   AUC: {cross_surv['auc']:.3f}")
        print(f"    Predicted JOIN rate: {cross_surv['predicted_join_rate']:.3f}   "
              f"Actual: {cross_surv['actual_join_rate']:.3f}")
        delta = cross_surv["predicted_join_rate"] - cross_surv["actual_join_rate"]
        print(f"    Wedge (predicted - actual): {delta:+.3f} "
              f"({'classifier misses chilling effect' if abs(delta) > 0.05 else 'similar'})")
        results["slider_logistic"]["cross_pure_to_surv"] = cross_surv

    # ==================================================================
    # Classifier 3: Keyphrase + sentiment model
    # ==================================================================
    print("\n" + "-" * 72)
    print("Classifier 3: Keyphrase + sentiment model")
    print("-" * 72)

    X_kp_pure = _keyphrase_features(df_pure)
    feature_names = ["action_count", "caution_count", "msg_length",
                     "exclamation_count", "question_count"]

    cv_kp = _cv_evaluate(X_kp_pure, y_pure)
    print(f"  5-fold CV accuracy: {cv_kp['accuracy_mean']:.3f} +/- {cv_kp['accuracy_std']:.3f}")
    print(f"  5-fold CV AUC:      {cv_kp['auc_mean']:.3f} +/- {cv_kp['auc_std']:.3f}")
    results["keyphrase_sentiment"] = {"cv_pure": cv_kp}

    # Print feature importances
    clf_kp = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf_kp.fit(X_kp_pure, y_pure)
    print("  Feature coefficients:")
    for name, coef in zip(feature_names, clf_kp.coef_[0]):
        print(f"    {name:>20s}: {coef:+.4f}")

    if len(df_comm):
        X_kp_comm = _keyphrase_features(df_comm)
        y_comm = df_comm["label"].values
        cross_comm = _train_full_and_predict(X_kp_pure, y_pure, X_kp_comm, y_comm)
        print(f"  Cross-treatment (pure -> comm):")
        print(f"    Accuracy: {cross_comm['accuracy']:.3f}   AUC: {cross_comm['auc']:.3f}")
        print(f"    Predicted JOIN rate: {cross_comm['predicted_join_rate']:.3f}   "
              f"Actual: {cross_comm['actual_join_rate']:.3f}")
        results["keyphrase_sentiment"]["cross_pure_to_comm"] = cross_comm

    if len(df_surv):
        X_kp_surv = _keyphrase_features(df_surv)
        y_surv = df_surv["label"].values
        cross_surv = _train_full_and_predict(X_kp_pure, y_pure, X_kp_surv, y_surv)
        print(f"  Cross-treatment (pure -> surveillance):")
        print(f"    Accuracy: {cross_surv['accuracy']:.3f}   AUC: {cross_surv['auc']:.3f}")
        print(f"    Predicted JOIN rate: {cross_surv['predicted_join_rate']:.3f}   "
              f"Actual: {cross_surv['actual_join_rate']:.3f}")
        delta = cross_surv["predicted_join_rate"] - cross_surv["actual_join_rate"]
        print(f"    Wedge (predicted - actual): {delta:+.3f} "
              f"({'classifier misses chilling effect' if abs(delta) > 0.05 else 'similar'})")
        results["keyphrase_sentiment"]["cross_pure_to_surv"] = cross_surv

    # ==================================================================
    # Summary table
    # ==================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    header = f"{'Classifier':<30s} {'CV Acc':>8s} {'CV AUC':>8s}"
    if len(df_surv):
        header += f" {'Surv Acc':>9s} {'Pred JOIN':>10s} {'True JOIN':>10s} {'Wedge':>8s}"
    print(header)
    print("-" * len(header))

    for name, key in [("BoW TF-IDF", "bow_tfidf"),
                      ("Slider + interactions", "slider_logistic"),
                      ("Keyphrase + sentiment", "keyphrase_sentiment")]:
        r = results[key]
        cv = r["cv_pure"]
        line = f"{name:<30s} {cv['accuracy_mean']:>7.3f}  {cv['auc_mean']:>7.3f}"
        if len(df_surv) and "cross_pure_to_surv" in r:
            s = r["cross_pure_to_surv"]
            wedge = s["predicted_join_rate"] - s["actual_join_rate"]
            line += (f"  {s['accuracy']:>8.3f}"
                     f"  {s['predicted_join_rate']:>9.3f}"
                     f"  {s['actual_join_rate']:>9.3f}"
                     f"  {wedge:>+7.3f}")
        print(line)

    if len(df_surv):
        print("\nKey finding: Text classifiers predict surveillance JOIN rates close to")
        print("pure-game rates, missing the chilling effect.  The belief-action wedge")
        print("is not attributable to briefing text -- it arises from the surveillance")
        print("framing in the system prompt.")

    # ── Save results ──────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
