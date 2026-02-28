"""
Agent-level regressions for referee response.

Implements:
  Item 5  – Agent-level logit: Pr(Join) = logit(theta + treatment + theta*treatment + model_FE)
  Item 4A – Coordination ablation: do coordination cues independently affect join?
  Item 4B – Finite-N benchmark: Binomial(25, p(theta)) vs empirical regime fall rate

Reads JSON logs from output/, produces:
  - analysis/regression_results.json   (machine-readable results)
  - paper/tables/tab_regressions.tex   (LaTeX regression table)

Usage:
    uv run python analysis/agent_regressions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

# ── Paths ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_PATH = Path(__file__).resolve().parent / "regression_results.json"
TABLES_DIR = PROJECT_ROOT / "paper" / "tables"

# ── Model roster (same order as verify_paper_stats.py) ───────────────────

PART1_MODELS = [
    "mistralai--mistral-small-creative",
    "meta-llama--llama-3.3-70b-instruct",
    "mistralai--ministral-3b-2512",
    "qwen--qwen3-30b-a3b-instruct-2507",
    "openai--gpt-oss-120b",
    "qwen--qwen3-235b-a22b-2507",
    "arcee-ai--trinity-large-preview_free",
    "minimax--minimax-m2-her",
]

SHORT_NAMES = {
    "mistralai--mistral-small-creative": "Mistral",
    "meta-llama--llama-3.3-70b-instruct": "Llama 70B",
    "mistralai--ministral-3b-2512": "Ministral 3B",
    "qwen--qwen3-30b-a3b-instruct-2507": "Qwen 30B",
    "openai--gpt-oss-120b": "GPT-OSS 120B",
    "qwen--qwen3-235b-a22b-2507": "Qwen 235B",
    "arcee-ai--trinity-large-preview_free": "Trinity",
    "minimax--minimax-m2-her": "MiniMax",
}

# ── Mapping of (subdir, model_slug, treatment_label) for all experiments ─

# Core Part I experiments: model root directory, standard treatments
CORE_TREATMENTS = ["pure", "comm", "scramble", "flip"]

# Additional experiment directories with special structure
EXTRA_EXPERIMENTS: list[tuple[str, str, str]] = [
    # (output subdir path relative to OUTPUT_DIR, model_slug, treatment_label)
    # Surveillance
    ("surveillance/mistralai--mistral-small-creative", "mistralai--mistral-small-creative", "surveillance"),
    ("surveillance/meta-llama--llama-3.3-70b-instruct", "meta-llama--llama-3.3-70b-instruct", "surveillance"),
    ("surveillance/qwen--qwen3-30b-a3b-instruct-2507", "qwen--qwen3-30b-a3b-instruct-2507", "surveillance"),
    # Propaganda k=5
    ("propaganda-k5/mistralai--mistral-small-creative", "mistralai--mistral-small-creative", "propaganda_k5"),
    ("propaganda-k5/meta-llama--llama-3.3-70b-instruct", "meta-llama--llama-3.3-70b-instruct", "propaganda_k5"),
    # Propaganda k=2, k=10
    ("propaganda-k2/mistralai--mistral-small-creative", "mistralai--mistral-small-creative", "propaganda_k2"),
    ("propaganda-k10/mistralai--mistral-small-creative", "mistralai--mistral-small-creative", "propaganda_k10"),
    # Propaganda + surveillance
    ("propaganda-surveillance/mistralai--mistral-small-creative", "mistralai--mistral-small-creative", "propaganda_surveillance"),
]

# Belief experiments (have 'belief' field in agent data)
BELIEF_EXPERIMENTS: list[tuple[str, str, str]] = [
    ("mistralai--mistral-small-creative/_beliefs_comm/mistralai--mistral-small-creative",
     "mistralai--mistral-small-creative", "beliefs_comm"),
    ("mistralai--mistral-small-creative/_overwrite_200period_backup",
     "mistralai--mistral-small-creative", "beliefs_pure"),
    ("mistralai--mistral-small-creative/_overwrite_200period_backup",
     "mistralai--mistral-small-creative", "beliefs_surveillance"),
    ("meta-llama--llama-3.3-70b-instruct/_beliefs/meta-llama--llama-3.3-70b-instruct",
     "meta-llama--llama-3.3-70b-instruct", "beliefs_pure"),
    ("meta-llama--llama-3.3-70b-instruct/_beliefs/meta-llama--llama-3.3-70b-instruct",
     "meta-llama--llama-3.3-70b-instruct", "beliefs_comm"),
    ("mistralai--mistral-small-creative/_beliefs_propaganda_k5/mistralai--mistral-small-creative",
     "mistralai--mistral-small-creative", "beliefs_propaganda_k5"),
]

# Mapping from belief experiment label to the actual log filename
BELIEF_LOG_NAMES = {
    "beliefs_comm": "experiment_comm_log.json",
    "beliefs_pure": "experiment_pure_beliefs_log.json",
    "beliefs_surveillance": "experiment_surveillance_beliefs_log.json",
    "beliefs_propaganda_k5": "experiment_comm_log.json",
}


# =====================================================================
# Section A: Data extraction
# =====================================================================

def _load_json_log(path: Path) -> list[dict]:
    """Load a JSON log file, return empty list if missing."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_agent_data_from_log(
    log_data: list[dict],
    model_slug: str,
    treatment_label: str,
) -> pd.DataFrame:
    """Flatten a JSON log's agent array into a DataFrame of agent-level rows."""
    rows: list[dict] = []
    for entry in log_data:
        theta = entry.get("theta", np.nan)
        theta_star = entry.get("theta_star", np.nan)
        country = entry.get("country", entry.get("rep", 0))
        period = entry.get("period", 0)
        coup_success = entry.get("coup_success", None)

        for ag in entry.get("agents", []):
            # Skip API errors and unparseable responses
            if ag.get("api_error", False):
                continue
            decision_raw = ag.get("decision", "")
            if not isinstance(decision_raw, str):
                continue
            decision_upper = decision_raw.strip().upper()
            if decision_upper not in ("JOIN", "STAY"):
                continue

            is_propaganda = ag.get("is_propaganda", False)

            row = {
                "model": model_slug,
                "treatment": treatment_label,
                "country": country,
                "period": period,
                "agent_id": ag.get("id", 0),
                "theta": theta,
                "theta_star": theta_star,
                "z_score": ag.get("z_score", np.nan),
                "signal": ag.get("signal", np.nan),
                "direction": ag.get("direction", np.nan),
                "clarity": ag.get("clarity", np.nan),
                "coordination": ag.get("coordination", np.nan),
                "join": 1 if decision_upper == "JOIN" else 0,
                "is_propaganda": is_propaganda,
                "coup_success": coup_success,
            }

            # Belief (only present in some experiments)
            if "belief" in ag and ag["belief"] is not None:
                row["belief"] = float(ag["belief"])
            else:
                row["belief"] = np.nan

            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _log_filename_for_treatment(treatment: str) -> str:
    """Map treatment label to expected log filename."""
    mapping = {
        "pure": "experiment_pure_log.json",
        "comm": "experiment_comm_log.json",
        "scramble": "experiment_scramble_log.json",
        "flip": "experiment_flip_log.json",
        "surveillance": "experiment_comm_log.json",  # surveillance uses comm treatment
        "propaganda_k2": "experiment_comm_log.json",
        "propaganda_k5": "experiment_comm_log.json",
        "propaganda_k10": "experiment_comm_log.json",
        "propaganda_surveillance": "experiment_comm_log.json",
    }
    return mapping.get(treatment, f"experiment_{treatment}_log.json")


def build_full_dataset() -> pd.DataFrame:
    """
    Build the complete agent-level dataset from all available JSON logs.
    Returns a DataFrame with columns:
        model, treatment, country, period, agent_id, theta, theta_star,
        z_score, signal, direction, clarity, coordination, join,
        is_propaganda, belief, coup_success
    """
    frames: list[pd.DataFrame] = []

    # ── Core Part I experiments ──
    for model_slug in PART1_MODELS:
        for treatment in CORE_TREATMENTS:
            log_file = _log_filename_for_treatment(treatment)
            path = OUTPUT_DIR / model_slug / log_file
            if not path.exists():
                continue
            log_data = _load_json_log(path)
            if not log_data:
                continue
            df = extract_agent_data_from_log(log_data, model_slug, treatment)
            if not df.empty:
                frames.append(df)
                print(f"  Loaded {len(df):>7,} agent-rows: {model_slug} / {treatment}")

    # ── Extra experiments (surveillance, propaganda, etc.) ──
    for subdir, model_slug, label in EXTRA_EXPERIMENTS:
        log_file = _log_filename_for_treatment(label)
        path = OUTPUT_DIR / subdir / log_file
        if not path.exists():
            continue
        log_data = _load_json_log(path)
        if not log_data:
            continue
        df = extract_agent_data_from_log(log_data, model_slug, label)
        if not df.empty:
            frames.append(df)
            print(f"  Loaded {len(df):>7,} agent-rows: {model_slug} / {label}")

    if not frames:
        print("ERROR: No data loaded. Check output/ directory.", file=sys.stderr)
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    print(f"\n  Total agent-level observations: {len(full):,}")
    print(f"  Models: {full['model'].nunique()}")
    print(f"  Treatments: {sorted(full['treatment'].unique())}")
    return full


def build_belief_dataset() -> pd.DataFrame:
    """Build dataset from experiments that include belief elicitation."""
    frames: list[pd.DataFrame] = []

    for subdir, model_slug, label in BELIEF_EXPERIMENTS:
        log_file = BELIEF_LOG_NAMES.get(label, f"experiment_{label}_log.json")
        path = OUTPUT_DIR / subdir / log_file
        if not path.exists():
            continue
        log_data = _load_json_log(path)
        if not log_data:
            continue
        df = extract_agent_data_from_log(log_data, model_slug, label)
        if not df.empty and df["belief"].notna().any():
            frames.append(df)
            n_beliefs = df["belief"].notna().sum()
            print(f"  Loaded {n_beliefs:>6,} belief obs: {model_slug} / {label}")

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    # Keep only rows with valid belief data
    full = full[full["belief"].notna()].copy()
    print(f"\n  Total belief observations: {len(full):,}")
    return full


# =====================================================================
# Section C: Agent-level logit (Referee Item 5)
# =====================================================================

def _cluster_id(df: pd.DataFrame) -> pd.Series:
    """Create cluster IDs at model-country-period level."""
    return (
        df["model"].astype(str) + "_"
        + df["country"].astype(str) + "_"
        + df["period"].astype(str)
    )


def _format_logit_results(result, var_names: list[str]) -> dict:
    """Extract key stats from a statsmodels Logit result."""
    out: dict[str, Any] = {
        "n_obs": int(result.nobs),
        "pseudo_r2": round(float(result.prsquared), 4),
        "log_likelihood": round(float(result.llf), 2),
        "aic": round(float(result.aic), 2),
        "bic": round(float(result.bic), 2),
        "coefficients": {},
    }
    for i, name in enumerate(var_names):
        out["coefficients"][name] = {
            "coef": round(float(result.params[i]), 4),
            "se": round(float(result.bse[i]), 4),
            "z": round(float(result.tvalues[i]), 3),
            "p": round(float(result.pvalues[i]), 6),
            "ci_lo": round(float(result.conf_int()[i, 0]), 4),
            "ci_hi": round(float(result.conf_int()[i, 1]), 4),
        }
    return out


def _format_ols_results(result, var_names: list[str]) -> dict:
    """Extract key stats from a statsmodels OLS result."""
    out: dict[str, Any] = {
        "n_obs": int(result.nobs),
        "r_squared": round(float(result.rsquared), 4),
        "adj_r_squared": round(float(result.rsquared_adj), 4),
        "f_stat": round(float(result.fvalue), 3) if not np.isnan(result.fvalue) else None,
        "f_pvalue": round(float(result.f_pvalue), 6) if not np.isnan(result.f_pvalue) else None,
        "coefficients": {},
    }
    for i, name in enumerate(var_names):
        out["coefficients"][name] = {
            "coef": round(float(result.params[i]), 4),
            "se": round(float(result.bse[i]), 4),
            "t": round(float(result.tvalues[i]), 3),
            "p": round(float(result.pvalues[i]), 6),
            "ci_lo": round(float(result.conf_int()[i, 0]), 4),
            "ci_hi": round(float(result.conf_int()[i, 1]), 4),
        }
    return out


def run_agent_logit(df: pd.DataFrame) -> dict[str, Any]:
    """
    Main agent-level logit specification:
        Pr(Join=1) = Logit(beta0 + beta1*theta + beta2*treatment_dummies
                          + beta3*theta*treatment + model_FE)
    Clustered standard errors at model-country-period level.

    Returns dict with regression results.
    """
    results: dict[str, Any] = {}

    # ── Prepare data: exclude propaganda agents ──
    reg_df = df[~df["is_propaganda"]].copy()

    # Treatment dummies (pure is base category)
    treatments = sorted([t for t in reg_df["treatment"].unique() if t != "pure"])
    for t in treatments:
        reg_df[f"treat_{t}"] = (reg_df["treatment"] == t).astype(float)

    # Theta interactions
    for t in treatments:
        reg_df[f"theta_x_{t}"] = reg_df["theta"] * reg_df[f"treat_{t}"]

    # Model fixed effects (first model is base)
    models = sorted(reg_df["model"].unique())
    base_model = models[0] if models else ""
    other_models = [m for m in models if m != base_model]
    for m in other_models:
        reg_df[f"fe_{m}"] = (reg_df["model"] == m).astype(float)

    # ── Assemble X matrix ──
    x_cols = ["theta"]
    x_cols += [f"treat_{t}" for t in treatments]
    x_cols += [f"theta_x_{t}" for t in treatments]
    x_cols += [f"fe_{m}" for m in other_models]

    y = reg_df["join"].values
    X = sm.add_constant(reg_df[x_cols].values)
    var_names = ["const"] + x_cols

    # Drop rows with NaN
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    y, X = y[mask], X[mask]
    cluster_ids = _cluster_id(reg_df)[mask].values

    print(f"\n{'='*70}")
    print(f"AGENT-LEVEL LOGIT: Pr(Join=1) ~ theta + treatment + theta*treatment + model_FE")
    print(f"{'='*70}")
    print(f"N = {len(y):,} agent-decisions (excl. propaganda, API errors)")
    print(f"Treatments: {treatments}")
    print(f"Models: {models}")
    print(f"Base model: {base_model}")
    print(f"Clusters (model-country-period): {len(np.unique(cluster_ids)):,}")

    # ── Fit logit with clustered SEs ──
    try:
        logit_model = Logit(y, X)
        logit_clustered = logit_model.fit(
            disp=0, maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": cluster_ids},
        )

        results["main_logit"] = _format_logit_results(logit_clustered, var_names)
        results["main_logit"]["base_model"] = base_model
        results["main_logit"]["base_treatment"] = "pure"
        results["main_logit"]["n_clusters"] = int(len(np.unique(cluster_ids)))

        print(f"\nPseudo R-squared: {results['main_logit']['pseudo_r2']:.4f}")
        print(f"\n{'Variable':<35} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8}")
        print("-" * 75)
        for name, vals in results["main_logit"]["coefficients"].items():
            stars = ""
            if vals["p"] < 0.01:
                stars = "***"
            elif vals["p"] < 0.05:
                stars = "**"
            elif vals["p"] < 0.10:
                stars = "*"
            # Shorten fe_ names for display
            display_name = name.replace("fe_", "FE:").replace("treat_", "").replace("theta_x_", "theta*")
            print(f"{display_name:<35} {vals['coef']:>8.4f} {vals['se']:>8.4f} {vals['z']:>8.3f} {vals['p']:>8.4f} {stars}")

    except Exception as e:
        print(f"ERROR fitting main logit: {e}")
        results["main_logit"] = {"error": str(e)}

    # ── Simpler specification: theta only (no treatment interactions) ──
    try:
        x_simple = sm.add_constant(reg_df[["theta"]].values[mask])
        simple_model = Logit(y, x_simple)
        simple_clustered = simple_model.fit(
            disp=0,
            cov_type="cluster",
            cov_kwds={"groups": cluster_ids},
        )
        results["theta_only_logit"] = _format_logit_results(
            simple_clustered, ["const", "theta"]
        )
        print(f"\nSimple theta-only logit:")
        for name, vals in results["theta_only_logit"]["coefficients"].items():
            print(f"  {name}: coef={vals['coef']:.4f}, SE={vals['se']:.4f}, p={vals['p']:.4f}")
    except Exception as e:
        print(f"ERROR fitting simple logit: {e}")
        results["theta_only_logit"] = {"error": str(e)}

    return results


def run_belief_regressions(belief_df: pd.DataFrame) -> dict[str, Any]:
    """
    Belief-related regressions:
      (1) belief ~ z_score + treatment
      (2) join ~ belief + z_score + treatment  (partial effect of belief)

    Returns dict with regression results.
    """
    results: dict[str, Any] = {}

    if belief_df.empty or belief_df["belief"].notna().sum() < 50:
        print("\nInsufficient belief data for regressions.")
        return results

    # Exclude propaganda agents
    reg_df = belief_df[~belief_df["is_propaganda"]].copy()
    reg_df = reg_df[reg_df["belief"].notna()].copy()

    if len(reg_df) < 50:
        print("\nInsufficient belief data after filtering.")
        return results

    # Treatment dummies
    treatments = sorted(reg_df["treatment"].unique())
    base_treatment = treatments[0]
    other_treatments = [t for t in treatments if t != base_treatment]
    for t in other_treatments:
        reg_df[f"treat_{t}"] = (reg_df["treatment"] == t).astype(float)

    cluster_ids = _cluster_id(reg_df).values

    # ── (1) Belief equation: belief ~ z_score + treatment ──
    print(f"\n{'='*70}")
    print(f"BELIEF EQUATION: belief ~ z_score + treatment")
    print(f"{'='*70}")
    print(f"N = {len(reg_df):,} (with belief data)")
    print(f"Base treatment: {base_treatment}")

    try:
        x_cols_1 = ["z_score"] + [f"treat_{t}" for t in other_treatments]
        X1 = sm.add_constant(reg_df[x_cols_1].values)
        y1 = reg_df["belief"].values
        var_names_1 = ["const"] + x_cols_1

        mask1 = np.isfinite(X1).all(axis=1) & np.isfinite(y1)
        X1, y1 = X1[mask1], y1[mask1]
        clu1 = cluster_ids[mask1]

        ols_model = sm.OLS(y1, X1)
        ols_clustered = ols_model.fit(
            cov_type="cluster",
            cov_kwds={"groups": clu1},
        )
        results["belief_equation"] = _format_ols_results(ols_clustered, var_names_1)
        results["belief_equation"]["base_treatment"] = base_treatment

        print(f"R-squared: {results['belief_equation']['r_squared']:.4f}")
        print(f"\n{'Variable':<35} {'Coef':>8} {'SE':>8} {'t':>8} {'p':>8}")
        print("-" * 75)
        for name, vals in results["belief_equation"]["coefficients"].items():
            stars = "***" if vals["p"] < 0.01 else "**" if vals["p"] < 0.05 else "*" if vals["p"] < 0.10 else ""
            print(f"{name:<35} {vals['coef']:>8.4f} {vals['se']:>8.4f} {vals['t']:>8.3f} {vals['p']:>8.4f} {stars}")

    except Exception as e:
        print(f"ERROR fitting belief equation: {e}")
        results["belief_equation"] = {"error": str(e)}

    # ── (2) Action equation: join ~ belief + z_score + treatment ──
    print(f"\n{'='*70}")
    print(f"ACTION EQUATION: Pr(Join=1) ~ belief + z_score + treatment")
    print(f"{'='*70}")

    try:
        x_cols_2 = ["belief", "z_score"] + [f"treat_{t}" for t in other_treatments]
        X2 = sm.add_constant(reg_df[x_cols_2].values)
        y2 = reg_df["join"].values
        var_names_2 = ["const"] + x_cols_2

        mask2 = np.isfinite(X2).all(axis=1) & np.isfinite(y2)
        X2, y2 = X2[mask2], y2[mask2]
        clu2 = cluster_ids[mask2]

        logit_model2 = Logit(y2, X2)
        logit_clustered2 = logit_model2.fit(
            disp=0, maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": clu2},
        )
        results["action_equation"] = _format_logit_results(logit_clustered2, var_names_2)
        results["action_equation"]["base_treatment"] = base_treatment

        print(f"Pseudo R-squared: {results['action_equation']['pseudo_r2']:.4f}")
        print(f"\n{'Variable':<35} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8}")
        print("-" * 75)
        for name, vals in results["action_equation"]["coefficients"].items():
            stars = "***" if vals["p"] < 0.01 else "**" if vals["p"] < 0.05 else "*" if vals["p"] < 0.10 else ""
            print(f"{name:<35} {vals['coef']:>8.4f} {vals['se']:>8.4f} {vals['z']:>8.3f} {vals['p']:>8.4f} {stars}")

    except Exception as e:
        print(f"ERROR fitting action equation: {e}")
        results["action_equation"] = {"error": str(e)}

    return results


# =====================================================================
# Section D: Coordination ablation (Referee Item 4A)
# =====================================================================

def run_coordination_ablation(df: pd.DataFrame) -> dict[str, Any]:
    """
    Test whether coordination cues independently affect join decisions:
        Pr(Join=1) = Logit(beta0 + beta1*direction + beta2*coordination
                          + beta3*direction*coordination)

    Direction captures "how much the briefing signals regime weakness"
    Coordination captures "how much the briefing signals others will act"

    If coordination has a significant partial effect (beta2 != 0),
    agents are responsive to strategic complementarities in the text.

    Returns dict with regression results.
    """
    results: dict[str, Any] = {}

    # Use only pure treatment to avoid confounding with communication
    reg_df = df[(df["treatment"] == "pure") & (~df["is_propaganda"])].copy()

    if len(reg_df) < 100:
        print("\nInsufficient pure-treatment data for coordination ablation.")
        return results

    # Drop rows with missing slider values
    reg_df = reg_df.dropna(subset=["direction", "clarity", "coordination"])

    # Create interaction
    reg_df["dir_x_coord"] = reg_df["direction"] * reg_df["coordination"]

    cluster_ids = _cluster_id(reg_df).values

    print(f"\n{'='*70}")
    print(f"COORDINATION ABLATION: Pr(Join=1) ~ direction + coordination + dir*coord")
    print(f"{'='*70}")
    print(f"N = {len(reg_df):,} (pure treatment only, excl. propaganda)")
    print(f"Direction range: [{reg_df['direction'].min():.3f}, {reg_df['direction'].max():.3f}]")
    print(f"Coordination range: [{reg_df['coordination'].min():.3f}, {reg_df['coordination'].max():.3f}]")

    # ── (1) Full model: direction + coordination + interaction ──
    try:
        x_cols = ["direction", "coordination", "dir_x_coord"]
        X = sm.add_constant(reg_df[x_cols].values)
        y = reg_df["join"].values
        var_names = ["const"] + x_cols

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]
        clu = cluster_ids[mask]

        logit_model = Logit(y, X)
        logit_clustered = logit_model.fit(
            disp=0, maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": clu},
        )
        results["full"] = _format_logit_results(logit_clustered, var_names)
        results["full"]["n_clusters"] = int(len(np.unique(clu)))

        print(f"\nFull model (direction + coordination + interaction):")
        print(f"Pseudo R-squared: {results['full']['pseudo_r2']:.4f}")
        print(f"N clusters: {results['full']['n_clusters']}")
        print(f"\n{'Variable':<35} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8}")
        print("-" * 75)
        for name, vals in results["full"]["coefficients"].items():
            stars = "***" if vals["p"] < 0.01 else "**" if vals["p"] < 0.05 else "*" if vals["p"] < 0.10 else ""
            print(f"{name:<35} {vals['coef']:>8.4f} {vals['se']:>8.4f} {vals['z']:>8.3f} {vals['p']:>8.4f} {stars}")

    except Exception as e:
        print(f"ERROR fitting full coordination ablation: {e}")
        results["full"] = {"error": str(e)}

    # ── (2) Direction-only model for comparison ──
    try:
        X_dir = sm.add_constant(reg_df[["direction"]].values)
        mask_dir = np.isfinite(X_dir).all(axis=1) & np.isfinite(reg_df["join"].values)
        X_dir, y_dir = X_dir[mask_dir], reg_df["join"].values[mask_dir]
        clu_dir = cluster_ids[mask_dir]

        logit_dir = Logit(y_dir, X_dir)
        logit_dir_clustered = logit_dir.fit(
            disp=0, maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": clu_dir},
        )
        results["direction_only"] = _format_logit_results(
            logit_dir_clustered, ["const", "direction"]
        )

        print(f"\nDirection-only model:")
        print(f"  Pseudo R2: {results['direction_only']['pseudo_r2']:.4f}")
        print(f"  direction coef: {results['direction_only']['coefficients']['direction']['coef']:.4f}")

    except Exception as e:
        results["direction_only"] = {"error": str(e)}

    # ── (3) With model fixed effects ──
    try:
        models = sorted(reg_df["model"].unique())
        if len(models) > 1:
            base_model = models[0]
            other_models = [m for m in models if m != base_model]
            for m in other_models:
                reg_df[f"fe_{m}"] = (reg_df["model"] == m).astype(float)

            x_cols_fe = ["direction", "coordination", "dir_x_coord"] + \
                        [f"fe_{m}" for m in other_models]
            X_fe = sm.add_constant(reg_df[x_cols_fe].values)
            y_fe = reg_df["join"].values
            var_names_fe = ["const"] + x_cols_fe

            mask_fe = np.isfinite(X_fe).all(axis=1) & np.isfinite(y_fe)
            X_fe, y_fe = X_fe[mask_fe], y_fe[mask_fe]
            clu_fe = cluster_ids[mask_fe]

            logit_fe = Logit(y_fe, X_fe)
            logit_fe_clustered = logit_fe.fit(
                disp=0, maxiter=200,
                cov_type="cluster",
                cov_kwds={"groups": clu_fe},
            )
            results["with_model_fe"] = _format_logit_results(logit_fe_clustered, var_names_fe)
            results["with_model_fe"]["base_model"] = base_model

            print(f"\nWith model FE:")
            print(f"  Pseudo R2: {results['with_model_fe']['pseudo_r2']:.4f}")
            coord_coef = results["with_model_fe"]["coefficients"]["coordination"]
            print(f"  coordination coef: {coord_coef['coef']:.4f} (p={coord_coef['p']:.4f})")
    except Exception as e:
        results["with_model_fe"] = {"error": str(e)}

    # ── (4) Across all treatments (pure + comm + scramble + flip) ──
    try:
        all_df = df[
            (df["treatment"].isin(CORE_TREATMENTS)) & (~df["is_propaganda"])
        ].copy()
        all_df = all_df.dropna(subset=["direction", "clarity", "coordination"])
        all_df["dir_x_coord"] = all_df["direction"] * all_df["coordination"]

        # Treatment dummies
        for t in ["comm", "scramble", "flip"]:
            all_df[f"treat_{t}"] = (all_df["treatment"] == t).astype(float)

        x_cols_all = ["direction", "coordination", "dir_x_coord",
                       "treat_comm", "treat_scramble", "treat_flip"]
        X_all = sm.add_constant(all_df[x_cols_all].values)
        y_all = all_df["join"].values
        var_names_all = ["const"] + x_cols_all
        clu_all = _cluster_id(all_df).values

        mask_all = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)
        X_all, y_all = X_all[mask_all], y_all[mask_all]
        clu_all = clu_all[mask_all]

        logit_all = Logit(y_all, X_all)
        logit_all_clustered = logit_all.fit(
            disp=0, maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": clu_all},
        )
        results["all_treatments"] = _format_logit_results(logit_all_clustered, var_names_all)

        print(f"\nAll core treatments (with treatment dummies):")
        print(f"  Pseudo R2: {results['all_treatments']['pseudo_r2']:.4f}")
        print(f"  N = {results['all_treatments']['n_obs']:,}")
        coord_coef = results["all_treatments"]["coefficients"]["coordination"]
        print(f"  coordination coef: {coord_coef['coef']:.4f} (p={coord_coef['p']:.4f})")

    except Exception as e:
        results["all_treatments"] = {"error": str(e)}

    return results


# =====================================================================
# Section E: Finite-N benchmark (Referee Item 4B)
# =====================================================================

def _logistic(x, L, k, x0):
    """Standard logistic function."""
    return L / (1.0 + np.exp(-k * (x - x0)))


def run_finite_n_benchmark(df: pd.DataFrame) -> dict[str, Any]:
    """
    With N=25 agents, compute:
      - From fitted logistic p(theta), compute Pr(Binomial(25, p(theta)) > 25*theta)
      - Compare to empirical regime fall rate at each theta
      - Report correlation between predicted and actual fall rates

    Uses pure-treatment data from each model separately and pooled.

    Returns dict with benchmark results.
    """
    results: dict[str, Any] = {}

    pure_df = df[(df["treatment"] == "pure") & (~df["is_propaganda"])].copy()

    if pure_df.empty:
        print("\nNo pure-treatment data for finite-N benchmark.")
        return results

    print(f"\n{'='*70}")
    print(f"FINITE-N BENCHMARK: Binomial(25, p(theta)) vs empirical fall rate")
    print(f"{'='*70}")

    # ── Pooled across all models ──
    print(f"\n--- Pooled across all models ---")
    pooled_result = _finite_n_for_subset(pure_df, "pooled")
    if pooled_result:
        results["pooled"] = pooled_result

    # ── Per-model ──
    results["per_model"] = {}
    for model_slug in sorted(pure_df["model"].unique()):
        model_df = pure_df[pure_df["model"] == model_slug]
        short = SHORT_NAMES.get(model_slug, model_slug)
        print(f"\n--- {short} ---")
        model_result = _finite_n_for_subset(model_df, short)
        if model_result:
            results["per_model"][model_slug] = model_result

    return results


def _finite_n_for_subset(df: pd.DataFrame, label: str) -> dict[str, Any] | None:
    """
    Compute finite-N benchmark for a subset of pure-treatment data.

    Steps:
    1. Bin theta, compute empirical p(theta) = mean join rate per bin
    2. Fit logistic to the binned data
    3. For each theta bin, compute Pr(Binom(25, p_hat) > 25*theta)
    4. Compare to empirical fraction of periods where coup succeeded
    """
    # We need period-level data: group by model+country+period to get theta and join_fraction
    # (different models share the same country/period indices)
    period_groups = df.groupby(["model", "country", "period"]).agg(
        theta=("theta", "first"),
        n_join=("join", "sum"),
        n_agents=("join", "count"),
        coup_success=("coup_success", "first"),
    ).reset_index()

    period_groups["join_rate"] = period_groups["n_join"] / period_groups["n_agents"]

    if len(period_groups) < 20:
        print(f"  {label}: too few periods ({len(period_groups)})")
        return None

    # Bin theta into ~15 bins
    theta_vals = period_groups["theta"].values
    n_bins = min(15, max(5, len(period_groups) // 10))
    bins = np.linspace(theta_vals.min(), theta_vals.max(), n_bins + 1)
    period_groups["theta_bin"] = pd.cut(period_groups["theta"], bins=bins, labels=False)
    period_groups = period_groups.dropna(subset=["theta_bin"])

    binned = period_groups.groupby("theta_bin").agg(
        theta_mid=("theta", "mean"),
        empirical_join_rate=("join_rate", "mean"),
        empirical_fall_rate=("coup_success", "mean"),
        n_periods=("theta", "count"),
    ).reset_index()

    if len(binned) < 4:
        print(f"  {label}: too few non-empty bins ({len(binned)})")
        return None

    # Fit logistic to empirical join rate
    theta_mid = binned["theta_mid"].values
    emp_join = binned["empirical_join_rate"].values

    try:
        popt, _ = curve_fit(
            _logistic, theta_mid, emp_join,
            p0=[1.0, -3.0, 0.5], bounds=([0.5, -20, -3], [1.0, -0.1, 3]),
            maxfev=5000,
        )
        L_fit, k_fit, x0_fit = popt
    except Exception as e:
        print(f"  {label}: logistic fit failed: {e}")
        return None

    print(f"  Fitted logistic: L={L_fit:.3f}, k={k_fit:.3f}, x0={x0_fit:.3f}")

    # For each theta bin, compute Pr(Binom(N, p_hat) > N*theta)
    # Regime falls when fraction joining > theta (normalized)
    N = 25
    predicted_fall_rates = []
    for _, row in binned.iterrows():
        theta_val = row["theta_mid"]
        p_hat = _logistic(theta_val, L_fit, k_fit, x0_fit)
        p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)

        # Regime falls if n_join > N * theta
        # But theta can be negative (very weak regime) or > 1 (very strong)
        threshold = N * theta_val
        if threshold < 0:
            # Threshold negative => any nonzero joiners topple
            pred_fall = 1.0 - stats.binom.pmf(0, N, p_hat)
        elif threshold >= N:
            # Threshold >= N => impossible to topple
            pred_fall = 0.0
        else:
            # Pr(X > threshold) = 1 - Pr(X <= floor(threshold))
            pred_fall = 1.0 - stats.binom.cdf(int(np.floor(threshold)), N, p_hat)

        predicted_fall_rates.append(pred_fall)

    binned["predicted_fall_rate"] = predicted_fall_rates

    # Correlation
    emp_fall = binned["empirical_fall_rate"].values
    pred_fall = np.array(predicted_fall_rates)

    r, p_val = stats.pearsonr(pred_fall, emp_fall)
    rmse = np.sqrt(np.mean((pred_fall - emp_fall) ** 2))
    mae = np.mean(np.abs(pred_fall - emp_fall))

    print(f"  Pearson r(predicted, empirical fall rate): {r:.4f} (p={p_val:.6f})")
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    result = {
        "logistic_params": {"L": round(L_fit, 4), "k": round(k_fit, 4), "x0": round(x0_fit, 4)},
        "n_bins": int(len(binned)),
        "n_periods": int(len(period_groups)),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p_val), 6),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "bins": [],
    }
    for _, row in binned.iterrows():
        result["bins"].append({
            "theta_mid": round(float(row["theta_mid"]), 4),
            "empirical_join_rate": round(float(row["empirical_join_rate"]), 4),
            "empirical_fall_rate": round(float(row["empirical_fall_rate"]), 4),
            "predicted_fall_rate": round(float(row["predicted_fall_rate"]), 4),
            "n_periods": int(row["n_periods"]),
        })

    return result


# =====================================================================
# Section G: LaTeX table generation
# =====================================================================

def _stars(p: float) -> str:
    if p < 0.01:
        return "^{***}"
    elif p < 0.05:
        return "^{**}"
    elif p < 0.10:
        return "^{*}"
    return ""


def _coef_cell(coef_dict: dict) -> str:
    """Format a coefficient cell: coef with stars on top, (SE) below."""
    c = coef_dict["coef"]
    se = coef_dict["se"]
    p = coef_dict["p"]
    return f"${c:.3f}{_stars(p)}$ & $({se:.3f})$"


def generate_regression_table(results: dict) -> str:
    """
    Generate a LaTeX regression table with three panels:
      (1) Main logit with treatment effects
      (2) Coordination ablation
      (3) Belief equation (if available)

    Uses a standard economics three-column layout.
    """
    main = results.get("agent_logit", {}).get("main_logit", {})
    coord = results.get("coordination_ablation", {}).get("full", {})
    belief_eq = results.get("belief_regressions", {}).get("belief_equation", {})
    action_eq = results.get("belief_regressions", {}).get("action_equation", {})

    # Determine which columns we can produce
    has_main = "coefficients" in main
    has_coord = "coefficients" in coord
    has_belief = "coefficients" in belief_eq
    has_action = "coefficients" in action_eq

    n_cols = sum([has_main, has_coord, has_action])
    if n_cols == 0:
        return "% No regression results available\n"

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")

    lines.append(r"\caption{Agent-Level Regressions}")
    lines.append(r"\label{tab:regressions}")

    # Build column spec
    col_spec = "l" + "cc" * n_cols
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row
    headers = []
    if has_main:
        headers.append(r"\multicolumn{2}{c}{(1) Join Decision}")
    if has_coord:
        headers.append(r"\multicolumn{2}{c}{(2) Coordination}")
    if has_action:
        headers.append(r"\multicolumn{2}{c}{(3) Belief $\to$ Action}")
    lines.append(" & " + " & ".join(headers) + r" \\")

    # Sub-header
    sub_headers = []
    if has_main:
        sub_headers.append(r"\multicolumn{2}{c}{Logit}")
    if has_coord:
        sub_headers.append(r"\multicolumn{2}{c}{Logit}")
    if has_action:
        sub_headers.append(r"\multicolumn{2}{c}{Logit}")
    lines.append(" & " + " & ".join(sub_headers) + r" \\")
    lines.append(r"\midrule")

    # Helper to add a row
    def add_row(var_label: str, main_key: str | None, coord_key: str | None, action_key: str | None):
        cells = []
        for col_data, key in [
            (main if has_main else None, main_key),
            (coord if has_coord else None, coord_key),
            (action_eq if has_action else None, action_key),
        ]:
            if col_data is None:
                continue
            if key and key in col_data.get("coefficients", {}):
                cells.append(_coef_cell(col_data["coefficients"][key]))
            else:
                cells.append(r" & ")
        lines.append(f"{var_label} & " + " & ".join(cells) + r" \\")

    # ── Rows ──
    add_row(r"$\theta$", "theta", None, None)
    add_row("Direction", None, "direction", None)
    add_row("Coordination", None, "coordination", None)
    add_row(r"Dir $\times$ Coord", None, "dir_x_coord", None)
    add_row("Belief", None, None, "belief")
    add_row(r"$z$-score", None, None, "z_score")

    # Treatment dummies from main logit
    if has_main:
        main_coefs = main.get("coefficients", {})
        treat_keys = [k for k in main_coefs if k.startswith("treat_")]
        for tk in sorted(treat_keys):
            label = tk.replace("treat_", "").replace("_", " ").title()
            add_row(label, tk, None, None)

        # Theta interactions
        interact_keys = [k for k in main_coefs if k.startswith("theta_x_")]
        for ik in sorted(interact_keys):
            label = r"$\theta \times$ " + ik.replace("theta_x_", "").replace("_", " ").title()
            add_row(label, ik, None, None)

    # Treatment dummies in action equation
    if has_action:
        act_coefs = action_eq.get("coefficients", {})
        act_treat_keys = [k for k in act_coefs if k.startswith("treat_")]
        for tk in sorted(act_treat_keys):
            label = tk.replace("treat_", "").replace("_", " ").title() + " (belief)"
            add_row(label, None, None, tk)

    add_row("Constant", "const", "const", "const")

    lines.append(r"\midrule")

    # Model FE indicator
    model_fe_row = ""
    if has_main:
        has_fe = any(k.startswith("fe_") for k in main.get("coefficients", {}))
        model_fe_row += r" & \multicolumn{2}{c}{" + ("Yes" if has_fe else "No") + "}"
    if has_coord:
        model_fe_row += r" & \multicolumn{2}{c}{No}"
    if has_action:
        model_fe_row += r" & \multicolumn{2}{c}{No}"
    lines.append(f"Model FE{model_fe_row}" + r" \\")

    # Clustered SE indicator
    cluster_row = ""
    if has_main:
        cluster_row += r" & \multicolumn{2}{c}{Yes}"
    if has_coord:
        cluster_row += r" & \multicolumn{2}{c}{Yes}"
    if has_action:
        cluster_row += r" & \multicolumn{2}{c}{Yes}"
    lines.append(f"Clustered SE{cluster_row}" + r" \\")

    # N
    n_row = ""
    if has_main:
        n_row += r" & \multicolumn{2}{c}{" + f"{main['n_obs']:,}" + "}"
    if has_coord:
        n_row += r" & \multicolumn{2}{c}{" + f"{coord['n_obs']:,}" + "}"
    if has_action:
        n_row += r" & \multicolumn{2}{c}{" + f"{action_eq['n_obs']:,}" + "}"
    lines.append(f"$N${n_row}" + r" \\")

    # Pseudo R2
    r2_row = ""
    if has_main:
        r2_row += r" & \multicolumn{2}{c}{" + f"{main['pseudo_r2']:.3f}" + "}"
    if has_coord:
        r2_row += r" & \multicolumn{2}{c}{" + f"{coord['pseudo_r2']:.3f}" + "}"
    if has_action:
        r2_row += r" & \multicolumn{2}{c}{" + f"{action_eq['pseudo_r2']:.3f}" + "}"
    lines.append(f"Pseudo $R^2${r2_row}" + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Notes
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item \textit{Notes:} Logit coefficients reported with clustered standard errors")
    lines.append(r"(model--country--period) in parentheses.")
    lines.append(r"Column (1): agent-level join decision on $\theta$, treatment dummies, and interactions,")
    lines.append(r"with model fixed effects. Base category: pure treatment.")
    lines.append(r"Column (2): coordination ablation using briefing slider values (pure treatment only).")
    lines.append(r"Column (3): partial effect of elicited belief on action, controlling for $z$-score.")
    lines.append(r"${}^{*}p<0.10$, ${}^{**}p<0.05$, ${}^{***}p<0.01$.")
    lines.append(r"\end{tablenotes}")


    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_finite_n_table(results: dict) -> str:
    """Generate a compact LaTeX table for finite-N benchmark results."""
    fn = results.get("finite_n_benchmark", {})
    if not fn:
        return "% No finite-N results available\n"

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")

    lines.append(r"\caption{Finite-$N$ Benchmark: Predicted vs.\ Empirical Regime Fall Rates}")
    lines.append(r"\label{tab:finite_n}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & $N$ periods & Logistic $x_0$ & Pearson $r$ & RMSE & MAE \\")
    lines.append(r"\midrule")

    # Per-model results
    per_model = fn.get("per_model", {})
    for model_slug in PART1_MODELS:
        if model_slug not in per_model:
            continue
        m = per_model[model_slug]
        short = SHORT_NAMES.get(model_slug, model_slug)
        r_val = m["pearson_r"]
        r_stars = _stars(m["pearson_p"])
        lines.append(
            f"  {short} & {m['n_periods']} & "
            f"${m['logistic_params']['x0']:.2f}$ & "
            f"${r_val:.3f}{r_stars}$ & "
            f"${m['rmse']:.3f}$ & "
            f"${m['mae']:.3f}$ \\\\"
        )

    lines.append(r"\midrule")

    # Pooled
    pooled = fn.get("pooled", {})
    if pooled:
        r_val = pooled["pearson_r"]
        r_stars = _stars(pooled["pearson_p"])
        lines.append(
            f"  \\textit{{Pooled}} & {pooled['n_periods']} & "
            f"${pooled['logistic_params']['x0']:.2f}$ & "
            f"${r_val:.3f}{r_stars}$ & "
            f"${pooled['rmse']:.3f}$ & "
            f"${pooled['mae']:.3f}$ \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item \textit{Notes:} For each $\theta$ bin, the predicted fall rate is")
    lines.append(r"$\Pr(\text{Binom}(25, \hat{p}(\theta)) > 25\theta)$ where $\hat{p}(\theta)$")
    lines.append(r"is the fitted logistic join probability. Pearson $r$ measures correlation")
    lines.append(r"between predicted and empirical fall rates across $\theta$ bins.")
    lines.append(r"${}^{*}p<0.10$, ${}^{**}p<0.05$, ${}^{***}p<0.01$.")
    lines.append(r"\end{tablenotes}")

    lines.append(r"\end{table}")

    return "\n".join(lines)


# =====================================================================
# Section F: Main entry point
# =====================================================================

def main():
    print("=" * 70)
    print("AGENT-LEVEL REGRESSIONS")
    print("Referee Items 5 (logit), 4A (coordination ablation), 4B (finite-N)")
    print("=" * 70)

    all_results: dict[str, Any] = {}

    # ── Step 1: Build full dataset ──
    print("\n--- Building full agent-level dataset ---")
    full_df = build_full_dataset()
    if full_df.empty:
        print("FATAL: No data loaded. Exiting.")
        sys.exit(1)

    # ── Step 2: Build belief dataset ──
    print("\n--- Building belief dataset ---")
    belief_df = build_belief_dataset()

    # ── Step 3: Agent-level logit (Item 5) ──
    all_results["agent_logit"] = run_agent_logit(full_df)

    # ── Step 4: Coordination ablation (Item 4A) ──
    all_results["coordination_ablation"] = run_coordination_ablation(full_df)

    # ── Step 5: Belief regressions ──
    if not belief_df.empty:
        all_results["belief_regressions"] = run_belief_regressions(belief_df)
    else:
        print("\nNo belief data available; skipping belief regressions.")
        all_results["belief_regressions"] = {}

    # ── Step 6: Finite-N benchmark (Item 4B) ──
    all_results["finite_n_benchmark"] = run_finite_n_benchmark(full_df)

    # ── Step 7: Save results ──
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Step 8: Generate LaTeX tables ──
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    reg_table = generate_regression_table(all_results)
    reg_path = TABLES_DIR / "tab_regressions.tex"
    reg_path.write_text(reg_table, encoding="utf-8")
    print(f"Regression table saved to {reg_path}")

    fn_table = generate_finite_n_table(all_results)
    fn_path = TABLES_DIR / "tab_finite_n.tex"
    fn_path.write_text(fn_table, encoding="utf-8")
    print(f"Finite-N table saved to {fn_path}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if "main_logit" in all_results.get("agent_logit", {}):
        ml = all_results["agent_logit"]["main_logit"]
        if "coefficients" in ml:
            theta_coef = ml["coefficients"].get("theta", {})
            print(f"  Main logit: theta coef = {theta_coef.get('coef', 'N/A')}, "
                  f"p = {theta_coef.get('p', 'N/A')}, N = {ml.get('n_obs', 'N/A'):,}")

    if "full" in all_results.get("coordination_ablation", {}):
        ca = all_results["coordination_ablation"]["full"]
        if "coefficients" in ca:
            coord = ca["coefficients"].get("coordination", {})
            print(f"  Coordination ablation: coef = {coord.get('coef', 'N/A')}, "
                  f"p = {coord.get('p', 'N/A')}, N = {ca.get('n_obs', 'N/A'):,}")

    fn = all_results.get("finite_n_benchmark", {}).get("pooled", {})
    if fn:
        print(f"  Finite-N benchmark (pooled): r = {fn.get('pearson_r', 'N/A')}, "
              f"RMSE = {fn.get('rmse', 'N/A')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
