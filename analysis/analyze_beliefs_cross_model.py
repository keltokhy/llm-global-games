"""Cross-model belief elicitation analysis.

Compares belief elicitation patterns across models (Mistral vs Llama)
and across treatments (pure, comm, surveillance, propaganda).

Reads from:
  - Mistral: output/mistralai--mistral-small-creative/_overwrite_200period_backup/
  - Mistral comm: output/mistralai--mistral-small-creative/_beliefs_comm/mistralai--mistral-small-creative/
  - Llama: output/meta-llama--llama-3.3-70b-instruct/_beliefs/
  - Mistral propaganda: output/mistralai--mistral-small-creative/_beliefs_propaganda_k5/
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGMA = 0.3


def load_agents(log_path, verbose=False):
    """Extract flat agent-level records from a log file."""
    with open(log_path) as f:
        periods = json.load(f)
    rows = []
    total = 0
    parsed = 0
    for p in periods:
        theta = p["theta"]
        theta_star = p["theta_star"]
        for a in p["agents"]:
            if a.get("api_error") or a.get("is_propaganda"):
                continue
            total += 1
            if a.get("belief") is None:
                continue
            parsed += 1
            signal = a["signal"]
            belief = a["belief"] / 100.0
            decision = 1 if a["decision"] == "JOIN" else 0
            posterior = stats.norm.cdf((theta_star - signal) / SIGMA)
            rows.append({
                "belief": belief,
                "decision": decision,
                "posterior": posterior,
                "signal": signal,
                "theta": theta,
                "theta_star": theta_star,
            })
    if verbose and total > 0:
        print(f"  Parse rate: {parsed}/{total} ({100*parsed/total:.1f}%)")
    return rows


def analyze_treatment(rows, label):
    """Compute key belief statistics for one treatment."""
    n = len(rows)
    beliefs = np.array([r["belief"] for r in rows])
    decisions = np.array([r["decision"] for r in rows])
    posteriors = np.array([r["posterior"] for r in rows])
    signals = np.array([r["signal"] for r in rows])

    r_post, p_post = stats.pearsonr(posteriors, beliefs)
    r_bd, p_bd = stats.pearsonr(beliefs, decisions)
    slope, intercept, r_val, _, _ = stats.linregress(posteriors, beliefs)

    # Partial correlation: belief → decision | signal
    slope_ds, int_ds = np.polyfit(signals, decisions, 1)
    resid_d = decisions - (int_ds + slope_ds * signals)
    slope_bs, int_bs = np.polyfit(signals, beliefs, 1)
    resid_b = beliefs - (int_bs + slope_bs * signals)
    r_partial, p_partial = stats.pearsonr(resid_b, resid_d)

    # 60-80% belief bin
    mask_6080 = (beliefs >= 0.6) & (beliefs < 0.8)
    n_6080 = mask_6080.sum()
    join_6080 = decisions[mask_6080].mean() if n_6080 > 0 else float("nan")
    se_6080 = np.sqrt(join_6080 * (1 - join_6080) / n_6080) if n_6080 > 0 else float("nan")

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  N = {n}")
    print(f"  Mean belief:  {beliefs.mean():.3f}")
    print(f"  Mean JOIN:    {decisions.mean():.3f}")
    print(f"  r(posterior, belief) = {r_post:+.3f}  (p = {p_post:.2e})")
    print(f"  OLS: belief = {intercept:.3f} + {slope:.3f} * posterior  (R² = {r_val**2:.3f})")
    print(f"  r(belief, decision) = {r_bd:+.3f}  (p = {p_bd:.2e})")
    print(f"  Partial r(belief, decision | signal) = {r_partial:+.3f}  (p = {p_partial:.2e})")
    print(f"  60-80% bin: join = {join_6080:.3f} (N={n_6080}, 95% CI [{join_6080-1.96*se_6080:.3f}, {join_6080+1.96*se_6080:.3f}])")

    return {
        "label": label, "n": n,
        "mean_belief": beliefs.mean(), "mean_join": decisions.mean(),
        "r_posterior": r_post, "ols_slope": slope, "ols_intercept": intercept,
        "r_belief_decision": r_bd, "r_partial": r_partial,
        "join_6080": join_6080, "n_6080": n_6080,
    }


def main():
    results = []

    # Mistral treatments
    mistral_backup = PROJECT_ROOT / "output" / "mistralai--mistral-small-creative" / "_overwrite_200period_backup"
    mistral_comm = PROJECT_ROOT / "output" / "mistralai--mistral-small-creative" / "_beliefs_comm" / "mistralai--mistral-small-creative"
    mistral_prop = PROJECT_ROOT / "output" / "mistralai--mistral-small-creative" / "_beliefs_propaganda_k5"

    for path, label in [
        (mistral_backup / "experiment_pure_beliefs_log.json", "Mistral — Pure"),
        (mistral_comm / "experiment_comm_log.json", "Mistral — Comm"),
        (mistral_backup / "experiment_surveillance_beliefs_log.json", "Mistral — Surveillance"),
    ]:
        if path.exists():
            rows = load_agents(path, verbose=True)
            results.append(analyze_treatment(rows, label))
        else:
            print(f"SKIP: {path} not found")

    # Mistral propaganda (may not exist yet)
    prop_paths = [
        mistral_prop / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
        mistral_prop / "experiment_comm_log.json",
    ]
    for p in prop_paths:
        if p.exists():
            rows = load_agents(p, verbose=True)
            results.append(analyze_treatment(rows, "Mistral — Propaganda k=5"))
            break
    else:
        print("\nSKIP: Mistral propaganda k=5 beliefs not found yet")

    # Llama treatments — check both direct and nested path structures
    llama_base = PROJECT_ROOT / "output" / "meta-llama--llama-3.3-70b-instruct"
    llama_search_paths = [
        (llama_base / "_beliefs" / "meta-llama--llama-3.3-70b-instruct" / "experiment_pure_log.json", "Llama 70B — Pure"),
        (llama_base / "_beliefs" / "meta-llama--llama-3.3-70b-instruct" / "experiment_comm_log.json", "Llama 70B — Comm"),
        (llama_base / "_beliefs_surveillance" / "meta-llama--llama-3.3-70b-instruct" / "experiment_comm_log.json", "Llama 70B — Surveillance"),
        # Fallback: direct paths
        (llama_base / "_beliefs" / "experiment_pure_log.json", "Llama 70B — Pure"),
        (llama_base / "_beliefs" / "experiment_comm_log.json", "Llama 70B — Comm"),
    ]
    seen_labels = set()
    for path, label in llama_search_paths:
        if label in seen_labels:
            continue
        if path.exists():
            rows = load_agents(path, verbose=True)
            if len(rows) > 0:
                results.append(analyze_treatment(rows, label))
                seen_labels.add(label)

    if not any("Llama" in r["label"] for r in results):
        print("\nSKIP: Llama belief runs not found yet — run scripts/run_beliefs.sh first")

    # Cross-model comparison table
    if len(results) > 1:
        print(f"\n\n{'=' * 80}")
        print("  CROSS-MODEL COMPARISON")
        print(f"{'=' * 80}")
        print(f"{'Treatment':<30s} {'N':>6s} {'Belief':>7s} {'JOIN':>6s} {'r(post)':>8s} {'slope':>6s} {'r(b,d)':>7s} {'partial':>8s} {'60-80%':>7s}")
        print("-" * 80)
        for r in results:
            print(f"{r['label']:<30s} {r['n']:>6d} {r['mean_belief']:>7.3f} {r['mean_join']:>6.3f} {r['r_posterior']:>+8.3f} {r['ols_slope']:>6.2f} {r['r_belief_decision']:>+7.3f} {r['r_partial']:>+8.3f} {r['join_6080']:>7.3f}")


if __name__ == "__main__":
    main()
