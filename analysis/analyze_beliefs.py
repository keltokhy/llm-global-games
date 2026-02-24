"""Analyze belief elicitation data from the 200-period backup experiments.

Reads agent-level logs with elicited beliefs (P(success) on 0-100 scale)
and tests whether beliefs track the Bayesian posterior and whether actions
are monotone in beliefs.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP = PROJECT_ROOT / "output" / "mistralai--mistral-small-creative" / "_overwrite_200period_backup"


def load_agents(log_path):
    """Extract flat agent-level records from a log file."""
    with open(log_path) as f:
        periods = json.load(f)
    rows = []
    for p in periods:
        theta = p["theta"]
        theta_star = p["theta_star"]
        sigma = 0.3  # experiment default
        # Theoretical posterior belief for agent with signal x_i:
        # P(regime falls | x_i) = P(A > theta | x_i)
        # In equilibrium: A(theta) = Phi((x* - theta)/sigma)
        # Agent threshold: x* = theta* + sigma * Phi^{-1}(theta*)
        x_star = theta_star + sigma * stats.norm.ppf(theta_star)
        for a in p["agents"]:
            if a.get("belief") is None or a.get("api_error"):
                continue
            signal = a["signal"]
            z_score = a["z_score"]
            belief = a["belief"] / 100.0  # normalize to [0, 1]
            decision = 1 if a["decision"] == "JOIN" else 0
            # Theoretical attack mass at this theta
            attack_mass = stats.norm.cdf((x_star - theta) / sigma)
            # Agent's posterior: P(theta < x*) given signal x_i
            # With diffuse prior: posterior is N(x_i, sigma^2)
            # P(regime falls) = P(A > theta) which in equilibrium = attack mass
            # But agent-level: P(success | x_i) ≈ based on their signal
            # Agent believes theta ~ N(x_i, sigma^2), regime falls if A > theta
            # i.e. if enough others join. In equilibrium, fraction joining = Phi((x* - theta)/sigma)
            # Agent's P(success) = P(Phi((x*-theta)/sigma) > theta | x_i)
            # = P(theta < theta* | x_i) = Phi((theta* - x_i)/sigma) ... wait
            # Actually: regime falls iff A > theta. In equilibrium A = Phi((x*-theta)/sigma).
            # A > theta iff theta < theta*. So P(success | x_i) = P(theta < theta* | x_i)
            # With diffuse prior, posterior is theta | x_i ~ N(x_i, sigma^2)
            # So P(success | x_i) = Phi((theta* - x_i) / sigma)
            posterior_success = stats.norm.cdf((theta_star - signal) / sigma)
            rows.append({
                "theta": theta,
                "theta_star": theta_star,
                "signal": signal,
                "z_score": z_score,
                "belief": belief,
                "decision": decision,
                "attack_mass": attack_mass,
                "posterior_success": posterior_success,
            })
    return rows


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def analyze(rows, label):
    print_section(label)
    n = len(rows)
    beliefs = np.array([r["belief"] for r in rows])
    decisions = np.array([r["decision"] for r in rows])
    signals = np.array([r["signal"] for r in rows])
    z_scores = np.array([r["z_score"] for r in rows])
    thetas = np.array([r["theta"] for r in rows])
    posteriors = np.array([r["posterior_success"] for r in rows])
    attack_mass = np.array([r["attack_mass"] for r in rows])

    print(f"\nN = {n} agent-level observations")
    print(f"Mean belief: {beliefs.mean():.3f} (SD {beliefs.std():.3f})")
    print(f"Mean decision (JOIN rate): {decisions.mean():.3f}")

    # 1. Do beliefs track the Bayesian posterior?
    print_section("1. Beliefs vs. Bayesian Posterior P(success | x_i)")
    r_post, p_post = stats.pearsonr(posteriors, beliefs)
    print(f"r(posterior, belief) = {r_post:+.3f}  (p = {p_post:.2e})")
    slope, intercept, r_val, p_val, se = stats.linregress(posteriors, beliefs)
    print(f"OLS: belief = {intercept:.3f} + {slope:.3f} * posterior  (R² = {r_val**2:.3f})")
    print(f"  Perfect calibration would be intercept=0, slope=1")

    # 2. Do beliefs track theta?
    print_section("2. Beliefs vs. Regime Strength (theta)")
    r_theta, p_theta = stats.pearsonr(thetas, beliefs)
    print(f"r(theta, belief) = {r_theta:+.3f}  (p = {p_theta:.2e})")

    # 3. Do beliefs track z-score?
    print_section("3. Beliefs vs. Agent Z-Score")
    r_z, p_z = stats.pearsonr(z_scores, beliefs)
    print(f"r(z_score, belief) = {r_z:+.3f}  (p = {p_z:.2e})")
    # z < 0 means weak regime signal → should believe success more likely
    # so we expect negative correlation (lower z → higher belief)
    print(f"  (Expect negative: lower z = weaker regime = higher P(success))")

    # 4. Are actions monotone in beliefs?
    print_section("4. Actions Monotone in Beliefs")
    r_ab, p_ab = stats.pearsonr(beliefs, decisions)
    print(f"r(belief, decision) = {r_ab:+.3f}  (p = {p_ab:.2e})")

    # Belief distribution by decision
    join_beliefs = beliefs[decisions == 1]
    stay_beliefs = beliefs[decisions == 0]
    print(f"\nBelief | JOIN: mean={join_beliefs.mean():.3f}, median={np.median(join_beliefs):.3f}, N={len(join_beliefs)}")
    print(f"Belief | STAY: mean={stay_beliefs.mean():.3f}, median={np.median(stay_beliefs):.3f}, N={len(stay_beliefs)}")
    t_stat, t_p = stats.ttest_ind(join_beliefs, stay_beliefs)
    print(f"t-test (JOIN vs STAY beliefs): t = {t_stat:.2f}, p = {t_p:.2e}")

    # Belief threshold: at what belief do agents switch?
    # Bin beliefs and compute JOIN rate per bin
    print(f"\nJoin rate by belief bin:")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in bins:
        mask = (beliefs >= lo) & (beliefs < hi)
        if mask.sum() > 0:
            print(f"  [{lo:.0%}-{hi:.0%}): JOIN rate = {decisions[mask].mean():.3f}  (N={mask.sum()})")

    # 5. Belief calibration: binned belief vs realized success rate
    print_section("5. Belief Calibration (Overconfidence / Underconfidence)")
    # Group by period to get realized outcomes
    # We need period-level success data
    theta_stars = np.array([r["theta_star"] for r in rows])
    realized_success = (thetas < theta_stars).astype(float)  # regime falls iff theta < theta*

    print(f"Realized success rate: {realized_success.mean():.3f}")
    print(f"Mean stated belief: {beliefs.mean():.3f}")
    print(f"Calibration gap: {beliefs.mean() - realized_success.mean():+.3f}")
    print(f"\nBy belief bin:")
    for lo, hi in bins:
        mask = (beliefs >= lo) & (beliefs < hi)
        if mask.sum() > 0:
            print(f"  [{lo:.0%}-{hi:.0%}): stated={beliefs[mask].mean():.3f}, realized={realized_success[mask].mean():.3f}, gap={beliefs[mask].mean()-realized_success[mask].mean():+.3f}  (N={mask.sum()})")

    # 6. Does belief add info beyond the signal?
    print_section("6. Belief vs Signal: Redundancy Check")
    r_sig_bel, _ = stats.pearsonr(signals, beliefs)
    print(f"r(signal, belief) = {r_sig_bel:+.3f}")
    # Partial: does belief predict decision controlling for signal?
    # Quick logistic-style check: residualize decision on signal, check belief
    from numpy.polynomial import polynomial as P
    # Simple: correlate decision residual with belief residual
    slope_ds, int_ds = np.polyfit(signals, decisions, 1)
    resid_d = decisions - (int_ds + slope_ds * signals)
    slope_bs, int_bs = np.polyfit(signals, beliefs, 1)
    resid_b = beliefs - (int_bs + slope_bs * signals)
    r_resid, p_resid = stats.pearsonr(resid_b, resid_d)
    print(f"r(belief residual, decision residual | signal) = {r_resid:+.3f}  (p = {p_resid:.2e})")
    print(f"  (Tests whether belief predicts action beyond what signal alone predicts)")

    return {
        "n": n,
        "r_posterior_belief": r_post,
        "r_theta_belief": r_theta,
        "r_zscore_belief": r_z,
        "r_belief_decision": r_ab,
        "mean_belief_join": float(join_beliefs.mean()),
        "mean_belief_stay": float(stay_beliefs.mean()),
        "calibration_gap": float(beliefs.mean() - realized_success.mean()),
        "r_resid_belief_decision": r_resid,
    }


if __name__ == "__main__":
    print("BELIEF ELICITATION ANALYSIS")
    print("Mistral Small Creative, 200 periods × 25 agents per treatment")
    print("Data source: _overwrite_200period_backup/")

    pure_rows = load_agents(BACKUP / "experiment_pure_beliefs_log.json")
    surv_rows = load_agents(BACKUP / "experiment_surveillance_beliefs_log.json")

    pure_stats = analyze(pure_rows, "PURE TREATMENT (no communication)")
    surv_stats = analyze(surv_rows, "SURVEILLANCE TREATMENT (communication + monitoring)")

    # Cross-treatment comparison
    print_section("CROSS-TREATMENT: Pure vs Surveillance")
    pure_beliefs = np.array([r["belief"] for r in pure_rows])
    surv_beliefs = np.array([r["belief"] for r in surv_rows])
    pure_decisions = np.array([r["decision"] for r in pure_rows])
    surv_decisions = np.array([r["decision"] for r in surv_rows])

    print(f"Mean belief  — Pure: {pure_beliefs.mean():.3f}, Surveillance: {surv_beliefs.mean():.3f}, Δ = {surv_beliefs.mean()-pure_beliefs.mean():+.3f}")
    print(f"Mean JOIN    — Pure: {pure_decisions.mean():.3f}, Surveillance: {surv_decisions.mean():.3f}, Δ = {surv_decisions.mean()-pure_decisions.mean():+.3f}")

    t_bel, p_bel = stats.ttest_ind(pure_beliefs, surv_beliefs)
    print(f"t-test beliefs: t = {t_bel:.2f}, p = {p_bel:.2e}")
    t_dec, p_dec = stats.ttest_ind(pure_decisions, surv_decisions)
    print(f"t-test decisions: t = {t_dec:.2f}, p = {p_dec:.2e}")

    # Key question: does surveillance change beliefs or just actions?
    print(f"\nDoes surveillance change beliefs, actions, or both?")
    print(f"  Belief gap: {surv_beliefs.mean()-pure_beliefs.mean():+.3f}")
    print(f"  Action gap: {surv_decisions.mean()-pure_decisions.mean():+.3f}")
    print(f"  If surveillance changes actions MORE than beliefs,")
    print(f"  agents are preference-falsifying (acting against their beliefs).")

    # Belief-action gap by treatment
    pure_join_mask = pure_decisions == 1
    surv_join_mask = surv_decisions == 1
    pure_gap = pure_decisions.mean() - pure_beliefs.mean()
    surv_gap = surv_decisions.mean() - surv_beliefs.mean()
    print(f"\n  Action - Belief gap (positive = acting more than believing):")
    print(f"    Pure:         {pure_gap:+.3f}")
    print(f"    Surveillance: {surv_gap:+.3f}")
