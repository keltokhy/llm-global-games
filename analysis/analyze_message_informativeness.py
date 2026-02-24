"""
Analyze message informativeness across communication treatments.

Computes how informative agent messages are about the latent state θ,
comparing regular communication, surveillance, and propaganda conditions.

Key measures:
  1. R² of message text features → θ regression (how much θ leaks through messages)
  2. Correlation of action scores with θ, z-score, and decisions
  3. Independent text baseline (no internal scores) for predicting θ

Usage: uv run python analysis/analyze_message_informativeness.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "output"

# ── Treatment log file paths ─────────────────────────────────────────

TREATMENTS = {
    "comm": OUTPUT / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
    "surveillance": OUTPUT / "surveillance" / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
    "propaganda_k5": OUTPUT / "propaganda-k5" / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
}

# ── Keyword lists ────────────────────────────────────────────────────

ACTION_WORDS = {
    "act", "action", "rise", "rising", "revolt", "rebel", "join", "fight",
    "resist", "overthrow", "protest", "strike", "march", "mobilize", "move",
    "now", "cracking", "crumbling", "collapse", "collapsing", "falling",
    "weak", "weakening", "fracture", "fracturing", "breaking", "fragile",
    "opportunity", "moment", "window", "momentum", "ready", "time",
    "together", "unite", "unified", "solidarity", "everyone",
}

CAUTION_WORDS = {
    "wait", "careful", "caution", "cautious", "patience", "patient", "risk",
    "risky", "dangerous", "danger", "trap", "stable", "strong", "strength",
    "grip", "control", "powerful", "secure", "security", "surveillance",
    "monitor", "watching", "uncertain", "unclear", "premature",
    "hold", "hesitate", "steady", "firm", "loyal", "intact",
    "suppress", "crackdown", "retaliate", "punish",
}


def extract_features(message: str) -> dict:
    """Extract text features from a single message."""
    if not message:
        return None
    words = re.findall(r"[a-z]+", message.lower())
    n_words = len(words)
    if n_words == 0:
        return None
    word_set = set(words)
    n_action = sum(1 for w in words if w in ACTION_WORDS)
    n_caution = sum(1 for w in words if w in CAUTION_WORDS)
    return {
        "word_count": n_words,
        "msg_length": len(message),
        "n_action": n_action,
        "n_caution": n_caution,
        "action_rate": n_action / n_words,
        "caution_rate": n_caution / n_words,
        "action_score": (n_action - n_caution) / n_words,
        "exclamation_count": message.count("!"),
        "question_count": message.count("?"),
        "uppercase_rate": sum(1 for c in message if c.isupper()) / max(len(message), 1),
    }


def load_treatment(path: Path) -> pd.DataFrame:
    """Load a log file into a flat DataFrame of agent-period observations."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for period in data:
        theta = period["theta"]
        theta_star = period.get("theta_star", np.nan)
        country = period.get("country", 0)
        period_num = period.get("period", 0)
        join_frac = period.get("join_fraction_valid", np.nan)

        for agent in period["agents"]:
            msg = agent.get("message_sent", "")
            if not msg:
                continue
            feats = extract_features(msg)
            if feats is None:
                continue
            row = {
                "theta": theta,
                "theta_star": theta_star,
                "country": country,
                "period": period_num,
                "agent_id": agent["id"],
                "z_score": agent["z_score"],
                "signal": agent["signal"],
                "direction": agent.get("direction", np.nan),
                "clarity": agent.get("clarity", np.nan),
                "coordination": agent.get("coordination", np.nan),
                "decision": 1 if agent.get("decision", "").upper() == "JOIN" else 0,
                "api_error": agent.get("api_error", False),
                "join_fraction": join_frac,
                **feats,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def regression_r2(X: np.ndarray, y: np.ndarray) -> float:
    """OLS R² for X (n, k) predicting y (n,). Returns 0 on failure."""
    if len(y) < 10:
        return np.nan
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return np.nan
        return 1 - ss_res / ss_tot
    except Exception:
        return np.nan


def compute_informativeness(df: pd.DataFrame) -> dict:
    """Compute all informativeness metrics for one treatment."""
    results = {}
    n = len(df)
    results["n_obs"] = n

    # --- Feature → θ regressions ---
    text_features = ["word_count", "action_rate", "caution_rate",
                     "action_score", "exclamation_count", "question_count",
                     "uppercase_rate", "msg_length"]

    X_text = df[text_features].values
    y_theta = df["theta"].values

    # R² of text features → θ (independent text baseline)
    results["R2_text_to_theta"] = regression_r2(X_text, y_theta)

    # R² of action_score alone → θ
    results["R2_action_score_to_theta"] = regression_r2(
        df[["action_score"]].values, y_theta
    )

    # R² of internal direction score → θ (for comparison)
    if df["direction"].notna().sum() > 10:
        results["R2_direction_to_theta"] = regression_r2(
            df[["direction"]].values, y_theta
        )
    else:
        results["R2_direction_to_theta"] = np.nan

    # --- Correlations of action_score with key variables ---
    action_score = df["action_score"].values

    for var in ["theta", "z_score", "decision"]:
        vals = df[var].values
        mask = np.isfinite(action_score) & np.isfinite(vals)
        if mask.sum() > 10:
            r, p = stats.pearsonr(action_score[mask], vals[mask])
            results[f"corr_action_score_{var}"] = r
            results[f"pval_action_score_{var}"] = p
        else:
            results[f"corr_action_score_{var}"] = np.nan
            results[f"pval_action_score_{var}"] = np.nan

    # --- Message statistics ---
    results["mean_word_count"] = df["word_count"].mean()
    results["mean_action_rate"] = df["action_rate"].mean()
    results["mean_caution_rate"] = df["caution_rate"].mean()
    results["mean_action_score"] = df["action_score"].mean()
    results["join_rate"] = df["decision"].mean()

    return results


def stars(p: float) -> str:
    """Significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main():
    print("=" * 80)
    print("MESSAGE INFORMATIVENESS ANALYSIS")
    print("=" * 80)

    all_results = {}
    dfs = {}

    for treatment, path in TREATMENTS.items():
        if not path.exists():
            print(f"\n  WARNING: {path} not found, skipping {treatment}")
            continue
        print(f"\nLoading {treatment} from {path.relative_to(PROJECT_ROOT)}...")
        df = load_treatment(path)
        dfs[treatment] = df
        print(f"  → {len(df)} agent-period observations")
        results = compute_informativeness(df)
        all_results[treatment] = results

    # ── Summary Table 1: R² (Message → θ) ────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE 1: HOW MUCH DO MESSAGES REVEAL ABOUT θ?")
    print("  R² from regressing θ on message text features")
    print("=" * 80)
    print(f"{'Treatment':<20} {'N':>8} {'R²(text→θ)':>12} {'R²(actScore→θ)':>16} {'R²(direction→θ)':>16}")
    print("-" * 80)
    for t in ["comm", "surveillance", "propaganda_k5"]:
        if t not in all_results:
            continue
        r = all_results[t]
        print(f"{t:<20} {r['n_obs']:>8d} {r['R2_text_to_theta']:>12.4f} "
              f"{r['R2_action_score_to_theta']:>16.4f} {r['R2_direction_to_theta']:>16.4f}")

    # ── Summary Table 2: Correlations ─────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE 2: ACTION SCORE CORRELATIONS")
    print("  Pearson r of message action_score with θ, z-score, decision")
    print("=" * 80)
    print(f"{'Treatment':<20} {'r(θ)':>10} {'r(z)':>10} {'r(decision)':>12} {'Join Rate':>10}")
    print("-" * 80)
    for t in ["comm", "surveillance", "propaganda_k5"]:
        if t not in all_results:
            continue
        r = all_results[t]
        rt = f"{r['corr_action_score_theta']:.3f}{stars(r['pval_action_score_theta'])}"
        rz = f"{r['corr_action_score_z_score']:.3f}{stars(r['pval_action_score_z_score'])}"
        rd = f"{r['corr_action_score_decision']:.3f}{stars(r['pval_action_score_decision'])}"
        print(f"{t:<20} {rt:>10} {rz:>10} {rd:>12} {r['join_rate']:>10.3f}")

    # ── Summary Table 3: Message characteristics ──────────────────────
    print("\n" + "=" * 80)
    print("TABLE 3: MESSAGE CHARACTERISTICS")
    print("=" * 80)
    print(f"{'Treatment':<20} {'Words':>8} {'ActionRate':>12} {'CautionRate':>13} {'ActionScore':>13}")
    print("-" * 80)
    for t in ["comm", "surveillance", "propaganda_k5"]:
        if t not in all_results:
            continue
        r = all_results[t]
        print(f"{t:<20} {r['mean_word_count']:>8.1f} {r['mean_action_rate']:>12.4f} "
              f"{r['mean_caution_rate']:>13.4f} {r['mean_action_score']:>13.4f}")

    # ── Statistical tests: R² differences ─────────────────────────────
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)

    # Bootstrap R² differences between treatments
    if "comm" in dfs and "surveillance" in dfs:
        print("\n  Bootstrap test: R²(comm) vs R²(surveillance) for text→θ")
        _bootstrap_r2_diff(dfs["comm"], dfs["surveillance"], "comm", "surveillance")

    if "comm" in dfs and "propaganda_k5" in dfs:
        print("\n  Bootstrap test: R²(comm) vs R²(propaganda_k5) for text→θ")
        _bootstrap_r2_diff(dfs["comm"], dfs["propaganda_k5"], "comm", "propaganda_k5")

    # Fisher z-test for correlation differences
    if "comm" in all_results and "surveillance" in all_results:
        print("\n  Fisher z-test for corr(action_score, θ) difference:")
        _fisher_z_test(
            all_results["comm"]["corr_action_score_theta"],
            all_results["comm"]["n_obs"],
            all_results["surveillance"]["corr_action_score_theta"],
            all_results["surveillance"]["n_obs"],
            "comm", "surveillance",
        )

    # ── Mediation-style analysis ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("MEDIATION CHECK: Does message informativeness explain action differences?")
    print("=" * 80)

    if "comm" in dfs and "surveillance" in dfs:
        _mediation_check(dfs["comm"], dfs["surveillance"])

    # ── Save results ──────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "analysis" / "message_informativeness_results.json"
    # Convert for JSON serialization
    serializable = {}
    for t, r in all_results.items():
        serializable[t] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                           for k, v in r.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path.relative_to(PROJECT_ROOT)}")


def _bootstrap_r2_diff(df1, df2, name1, name2, n_boot=2000):
    """Bootstrap test for R² difference between two treatments."""
    text_features = ["word_count", "action_rate", "caution_rate",
                     "action_score", "exclamation_count", "question_count",
                     "uppercase_rate", "msg_length"]

    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        idx1 = rng.choice(len(df1), size=len(df1), replace=True)
        idx2 = rng.choice(len(df2), size=len(df2), replace=True)
        r2_1 = regression_r2(df1.iloc[idx1][text_features].values, df1.iloc[idx1]["theta"].values)
        r2_2 = regression_r2(df2.iloc[idx2][text_features].values, df2.iloc[idx2]["theta"].values)
        if np.isfinite(r2_1) and np.isfinite(r2_2):
            diffs.append(r2_1 - r2_2)

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_value = np.mean(diffs <= 0)  # one-sided: H0: diff <= 0
    print(f"    R²({name1}) - R²({name2}): {mean_diff:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    P(diff <= 0) = {p_value:.4f}  →  {'significant' if p_value < 0.05 else 'not significant'} at α=0.05")


def _fisher_z_test(r1, n1, r2, n2, name1, name2):
    """Fisher z-transformation test for difference in correlations."""
    z1 = np.arctanh(np.clip(r1, -0.999, 0.999))
    z2 = np.arctanh(np.clip(r2, -0.999, 0.999))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"    r({name1})={r1:.3f}, r({name2})={r2:.3f}")
    print(f"    z-stat={z_stat:.3f}, p={p_value:.4f}  →  {'significant' if p_value < 0.05 else 'not significant'} at α=0.05")


def _mediation_check(df_comm, df_surv):
    """Check whether message informativeness mediates action differences."""
    # Combine datasets with treatment indicator
    df_comm = df_comm.copy()
    df_surv = df_surv.copy()
    df_comm["treatment"] = 0  # comm = 0
    df_surv["treatment"] = 1  # surveillance = 1
    df = pd.concat([df_comm, df_surv], ignore_index=True)

    # Path a: treatment → message action_score
    mask = np.isfinite(df["action_score"].values) & np.isfinite(df["theta"].values)
    df_clean = df[mask].copy()

    # Residualize action_score on theta first (to isolate treatment effect on informativeness)
    X_theta = np.column_stack([np.ones(len(df_clean)), df_clean["theta"].values])
    beta_theta = np.linalg.lstsq(X_theta, df_clean["action_score"].values, rcond=None)[0]
    df_clean["action_score_resid"] = df_clean["action_score"].values - X_theta @ beta_theta

    # Treatment effect on residualized action score
    comm_resid = df_clean.loc[df_clean["treatment"] == 0, "action_score_resid"]
    surv_resid = df_clean.loc[df_clean["treatment"] == 1, "action_score_resid"]
    t_stat_a, p_a = stats.ttest_ind(comm_resid, surv_resid)

    print(f"\n  Path a (treatment → action_score | θ):")
    print(f"    Comm mean residual: {comm_resid.mean():.4f}, Surv mean residual: {surv_resid.mean():.4f}")
    print(f"    t={t_stat_a:.3f}, p={p_a:.4f}")

    # Path c: treatment → decision (total effect)
    comm_join = df_clean.loc[df_clean["treatment"] == 0, "decision"].mean()
    surv_join = df_clean.loc[df_clean["treatment"] == 1, "decision"].mean()
    print(f"\n  Path c (treatment → decision, total effect):")
    print(f"    Comm join rate: {comm_join:.3f}, Surv join rate: {surv_join:.3f}")
    print(f"    Difference: {comm_join - surv_join:.3f}")

    # Path b: action_score → decision (controlling for treatment and θ)
    from scipy.special import expit
    X_full = np.column_stack([
        np.ones(len(df_clean)),
        df_clean["treatment"].values,
        df_clean["theta"].values,
        df_clean["action_score"].values,
    ])
    y_dec = df_clean["decision"].values

    # Linear probability model for interpretability
    beta_full = np.linalg.lstsq(X_full, y_dec, rcond=None)[0]
    print(f"\n  Path b (action_score → decision | treatment, θ):")
    print(f"    LPM coefficients: intercept={beta_full[0]:.3f}, treatment={beta_full[1]:.3f}, "
          f"θ={beta_full[2]:.3f}, action_score={beta_full[3]:.3f}")

    # Sobel-style summary
    print(f"\n  Summary: surveillance {'reduces' if surv_resid.mean() < comm_resid.mean() else 'increases'} "
          f"message action content (conditional on θ),")
    print(f"  and action content {'predicts' if abs(beta_full[3]) > 0.1 else 'weakly predicts'} "
          f"joining decisions (coeff={beta_full[3]:.3f}).")


if __name__ == "__main__":
    main()
