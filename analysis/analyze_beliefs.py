"""Analyze belief elicitation data from experiment logs.

Reads agent-level logs with elicited beliefs (P(success) on 0-100 scale)
and tests whether beliefs track the Bayesian posterior and whether actions
are monotone in beliefs.

Supports both post-decision beliefs (original) and pre-decision beliefs
(--belief-order both) for order-effect analysis.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP = PROJECT_ROOT / "output" / "mistralai--mistral-small-creative" / "_overwrite_200period_backup"


def load_agents(log_path, require_belief=True):
    """Extract flat agent-level records from a log file.

    If require_belief=True (default), skips agents without post-decision belief.
    Always includes pre-decision belief (belief_pre) if available.
    """
    with open(log_path) as f:
        periods = json.load(f)
    rows = []
    for p in periods:
        theta = p["theta"]
        theta_star = p["theta_star"]
        sigma = 0.3  # experiment default
        x_star = theta_star + sigma * stats.norm.ppf(theta_star)
        for a in p["agents"]:
            if a.get("api_error"):
                continue
            has_post = a.get("belief") is not None
            has_pre = a.get("belief_pre") is not None
            if require_belief and not has_post and not has_pre:
                continue
            signal = a["signal"]
            z_score = a["z_score"]
            decision = 1 if a["decision"] == "JOIN" else 0
            attack_mass = stats.norm.cdf((x_star - theta) / sigma)
            posterior_success = stats.norm.cdf((theta_star - signal) / sigma)
            row = {
                "theta": theta,
                "theta_star": theta_star,
                "signal": signal,
                "z_score": z_score,
                "decision": decision,
                "attack_mass": attack_mass,
                "posterior_success": posterior_success,
            }
            if has_post:
                row["belief"] = a["belief"] / 100.0
            if has_pre:
                row["belief_pre"] = a["belief_pre"] / 100.0
            rows.append(row)
    return rows


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def analyze(rows, label):
    """Analyze post-decision beliefs. Filters to rows with 'belief' key."""
    rows = [r for r in rows if "belief" in r]
    print_section(label)
    n = len(rows)
    if n == 0:
        print("  No post-decision beliefs found")
        return {}
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


def analyze_pre_beliefs(rows, label):
    """Analyze pre-decision beliefs and compare with post-decision beliefs.

    Tests whether the belief-action correlation and posterior tracking survive
    when beliefs are elicited BEFORE the decision (eliminating ex-post
    rationalization as an explanation).
    """
    # Filter to agents that have pre-decision beliefs
    pre_rows = [r for r in rows if "belief_pre" in r]
    if not pre_rows:
        print(f"\n  No pre-decision beliefs found in {label}")
        return None

    print_section(f"PRE-DECISION BELIEFS: {label}")
    n = len(pre_rows)
    pre_beliefs = np.array([r["belief_pre"] for r in pre_rows])
    decisions = np.array([r["decision"] for r in pre_rows])
    signals = np.array([r["signal"] for r in pre_rows])
    posteriors = np.array([r["posterior_success"] for r in pre_rows])

    print(f"N = {n} agents with pre-decision beliefs")
    print(f"Mean pre-belief: {pre_beliefs.mean():.3f} (SD {pre_beliefs.std():.3f})")
    print(f"Mean decision (JOIN rate): {decisions.mean():.3f}")

    # Pre-belief vs posterior
    r_pre_post, p_pre_post = stats.pearsonr(posteriors, pre_beliefs)
    print(f"\nr(posterior, pre-belief) = {r_pre_post:+.3f}  (p = {p_pre_post:.2e})")

    # Pre-belief vs action
    r_pre_act, p_pre_act = stats.pearsonr(pre_beliefs, decisions)
    print(f"r(pre-belief, decision) = {r_pre_act:+.3f}  (p = {p_pre_act:.2e})")

    # Pre-belief adds info beyond signal?
    slope_ds, int_ds = np.polyfit(signals, decisions, 1)
    resid_d = decisions - (int_ds + slope_ds * signals)
    slope_bs, int_bs = np.polyfit(signals, pre_beliefs, 1)
    resid_b = pre_beliefs - (int_bs + slope_bs * signals)
    r_resid, p_resid = stats.pearsonr(resid_b, resid_d)
    print(f"r(pre-belief resid, decision resid | signal) = {r_resid:+.3f}  (p = {p_resid:.2e})")

    result = {
        "n_pre": n,
        "r_posterior_pre_belief": round(float(r_pre_post), 4),
        "r_pre_belief_decision": round(float(r_pre_act), 4),
        "r_resid_pre_belief_decision": round(float(r_resid), 4),
        "mean_pre_belief": round(float(pre_beliefs.mean()), 4),
    }

    # Compare pre vs post if both available
    both_rows = [r for r in rows if "belief_pre" in r and "belief" in r]
    if both_rows:
        print_section(f"ORDER EFFECTS: Pre vs Post Beliefs ({label})")
        n_both = len(both_rows)
        pre = np.array([r["belief_pre"] for r in both_rows])
        post = np.array([r["belief"] for r in both_rows])
        decs = np.array([r["decision"] for r in both_rows])
        posts_theory = np.array([r["posterior_success"] for r in both_rows])

        print(f"N = {n_both} agents with both pre and post beliefs")
        print(f"Mean pre-belief:  {pre.mean():.3f}")
        print(f"Mean post-belief: {post.mean():.3f}")
        print(f"Mean difference (post - pre): {(post - pre).mean():+.3f}")

        r_pre_post_corr, _ = stats.pearsonr(pre, post)
        print(f"r(pre, post) = {r_pre_post_corr:+.3f}")

        # Key test: does pre-belief predict action as well as post-belief?
        r_pre_d, _ = stats.pearsonr(pre, decs)
        r_post_d, _ = stats.pearsonr(post, decs)
        print(f"\nr(pre-belief, decision)  = {r_pre_d:+.3f}")
        print(f"r(post-belief, decision) = {r_post_d:+.3f}")
        print(f"Difference: {r_post_d - r_pre_d:+.3f}")

        # Posterior tracking
        r_pre_theory, _ = stats.pearsonr(posts_theory, pre)
        r_post_theory, _ = stats.pearsonr(posts_theory, post)
        print(f"\nr(posterior, pre-belief)  = {r_pre_theory:+.3f}")
        print(f"r(posterior, post-belief) = {r_post_theory:+.3f}")

        # Order effect: paired t-test
        t_order, p_order = stats.ttest_rel(pre, post)
        print(f"\nPaired t-test (pre vs post): t = {t_order:.2f}, p = {p_order:.2e}")

        # Diagnosis: do post-beliefs shift toward the decision?
        joiners = decs == 1
        stayers = decs == 0
        shift_joiners = (post[joiners] - pre[joiners]).mean() if joiners.sum() > 0 else float("nan")
        shift_stayers = (post[stayers] - pre[stayers]).mean() if stayers.sum() > 0 else float("nan")
        print(f"\nPost-pre shift by decision:")
        print(f"  JOINers: {shift_joiners:+.3f}  (N={joiners.sum()})")
        print(f"  STAYers: {shift_stayers:+.3f}  (N={stayers.sum()})")
        if shift_joiners > 0 and shift_stayers < 0:
            print(f"  → Post-beliefs shift toward the decision (ex-post rationalization)")
        else:
            print(f"  → No clear ex-post rationalization pattern")

        result.update({
            "n_both": n_both,
            "r_pre_post": round(float(r_pre_post_corr), 4),
            "r_pre_decision": round(float(r_pre_d), 4),
            "r_post_decision": round(float(r_post_d), 4),
            "r_pre_posterior": round(float(r_pre_theory), 4),
            "r_post_posterior": round(float(r_post_theory), 4),
            "mean_shift_joiners": round(float(shift_joiners), 4),
            "mean_shift_stayers": round(float(shift_stayers), 4),
            "order_t": round(float(t_order), 4),
            "order_p": round(float(p_order), 6),
        })

    return result


def _find_belief_timing_logs():
    """Search for logs from --belief-order both experiments."""
    output_dir = PROJECT_ROOT / "output"
    candidates = []
    for model_dir in sorted(output_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for f in sorted(model_dir.glob("*beliefs_timing*log.json")):
            candidates.append(f)
        # Also check for tagged experiments
        for f in sorted(model_dir.glob("*belief*both*log.json")):
            candidates.append(f)
    return candidates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze belief elicitation data")
    parser.add_argument("--timing", action="store_true",
                        help="Also analyze pre-decision beliefs and order effects")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory containing experiment logs (overrides default)")
    args = parser.parse_args()

    print("BELIEF ELICITATION ANALYSIS")

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = BACKUP

    print(f"Data source: {log_dir}/")

    pure_path = log_dir / "experiment_pure_beliefs_log.json"
    surv_path = log_dir / "experiment_surveillance_beliefs_log.json"

    if not pure_path.exists():
        # Try alternate naming conventions
        for alt in ["experiment_pure_log.json", "experiment_pure_beliefs_timing_log.json"]:
            alt_path = log_dir / alt
            if alt_path.exists():
                pure_path = alt_path
                break

    if not surv_path.exists():
        for alt in ["experiment_comm_surveillance_log.json", "experiment_surveillance_log.json",
                     "experiment_comm_surveillance_beliefs_timing_log.json"]:
            alt_path = log_dir / alt
            if alt_path.exists():
                surv_path = alt_path
                break

    pure_rows = load_agents(pure_path) if pure_path.exists() else []
    surv_rows = load_agents(surv_path) if surv_path.exists() else []

    if pure_rows:
        pure_stats = analyze(pure_rows, "PURE TREATMENT (no communication)")
    if surv_rows:
        surv_stats = analyze(surv_rows, "SURVEILLANCE TREATMENT (communication + monitoring)")

    # Cross-treatment comparison
    if pure_rows and surv_rows:
        print_section("CROSS-TREATMENT: Pure vs Surveillance")
        pure_beliefs = np.array([r["belief"] for r in pure_rows if "belief" in r])
        surv_beliefs = np.array([r["belief"] for r in surv_rows if "belief" in r])
        pure_decisions = np.array([r["decision"] for r in pure_rows])
        surv_decisions = np.array([r["decision"] for r in surv_rows])

        if len(pure_beliefs) > 0 and len(surv_beliefs) > 0:
            print(f"Mean belief  — Pure: {pure_beliefs.mean():.3f}, Surveillance: {surv_beliefs.mean():.3f}, Δ = {surv_beliefs.mean()-pure_beliefs.mean():+.3f}")
            print(f"Mean JOIN    — Pure: {pure_decisions.mean():.3f}, Surveillance: {surv_decisions.mean():.3f}, Δ = {surv_decisions.mean()-pure_decisions.mean():+.3f}")

            t_bel, p_bel = stats.ttest_ind(pure_beliefs, surv_beliefs)
            print(f"t-test beliefs: t = {t_bel:.2f}, p = {p_bel:.2e}")
            t_dec, p_dec = stats.ttest_ind(pure_decisions, surv_decisions)
            print(f"t-test decisions: t = {t_dec:.2f}, p = {p_dec:.2e}")

            print(f"\nDoes surveillance change beliefs, actions, or both?")
            print(f"  Belief gap: {surv_beliefs.mean()-pure_beliefs.mean():+.3f}")
            print(f"  Action gap: {surv_decisions.mean()-pure_decisions.mean():+.3f}")
            print(f"  If surveillance changes actions MORE than beliefs,")
            print(f"  agents are self-censoring (acting against their beliefs).")

            pure_gap = pure_decisions.mean() - pure_beliefs.mean()
            surv_gap = surv_decisions.mean() - surv_beliefs.mean()
            print(f"\n  Action - Belief gap (positive = acting more than believing):")
            print(f"    Pure:         {pure_gap:+.3f}")
            print(f"    Surveillance: {surv_gap:+.3f}")

    # ── Pre-decision belief analysis (order effects) ──
    if args.timing or any("belief_pre" in r for r in pure_rows + surv_rows):
        print("\n" + "=" * 60)
        print("  PRE-DECISION BELIEF ANALYSIS (ORDER EFFECTS)")
        print("=" * 60)

        if pure_rows:
            pre_pure = analyze_pre_beliefs(pure_rows, "Pure treatment")
        if surv_rows:
            pre_surv = analyze_pre_beliefs(surv_rows, "Surveillance treatment")

        # Cross-treatment comparison using PRE-beliefs
        pre_pure_rows = [r for r in pure_rows if "belief_pre" in r]
        pre_surv_rows = [r for r in surv_rows if "belief_pre" in r]
        if pre_pure_rows and pre_surv_rows:
            print_section("SURVEILLANCE WEDGE WITH PRE-BELIEFS")
            pre_pure_b = np.array([r["belief_pre"] for r in pre_pure_rows])
            pre_surv_b = np.array([r["belief_pre"] for r in pre_surv_rows])
            pre_pure_d = np.array([r["decision"] for r in pre_pure_rows])
            pre_surv_d = np.array([r["decision"] for r in pre_surv_rows])

            print(f"Mean pre-belief — Pure: {pre_pure_b.mean():.3f}, Surv: {pre_surv_b.mean():.3f}")
            print(f"Mean JOIN       — Pure: {pre_pure_d.mean():.3f}, Surv: {pre_surv_d.mean():.3f}")

            t_pre_bel, p_pre_bel = stats.ttest_ind(pre_pure_b, pre_surv_b)
            t_pre_dec, p_pre_dec = stats.ttest_ind(pre_pure_d, pre_surv_d)
            print(f"\nt-test pre-beliefs: t = {t_pre_bel:.2f}, p = {p_pre_bel:.2e}")
            print(f"t-test decisions:   t = {t_pre_dec:.2f}, p = {p_pre_dec:.2e}")

            belief_gap = pre_surv_b.mean() - pre_pure_b.mean()
            action_gap = pre_surv_d.mean() - pre_pure_d.mean()
            print(f"\nPre-belief gap (surv - pure): {belief_gap:+.3f}")
            print(f"Action gap (surv - pure):     {action_gap:+.3f}")
            if abs(action_gap) > abs(belief_gap) * 3:
                print(f"→ Surveillance wedge SURVIVES with pre-beliefs")
                print(f"  (action shift {abs(action_gap)/max(abs(belief_gap), 0.001):.1f}× larger than belief shift)")
            else:
                print(f"→ Surveillance wedge is WEAKER with pre-beliefs")
