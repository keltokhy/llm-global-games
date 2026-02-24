"""Calibration framework: grid-sweep join rates and tune briefing params to match theory."""

import asyncio
import os
import numpy as np
import pandas as pd

from .briefing import (
    BriefingGenerator,
    DEFAULT_BOTTOMLINE_CUTS,
    DEFAULT_UNCLEAR_CUTS,
    DEFAULT_COORDINATION_CUTS,
    DEFAULT_COORDINATION_BLEND_PROB,
    DEFAULT_LANGUAGE_VARIANT,
    normalize_language_variant,
)
from .experiment import _call_llm, _parse_decision, SYSTEM_DECIDE_PURE
from .runtime import ensure_agg_backend


def build_z_grid(z_min, z_max, z_steps, mode="linear", edge_power=0.7):
    """Construct z-score grid; mode='edge' pushes density toward endpoints."""
    z_min = float(z_min)
    z_max = float(z_max)
    z_steps = int(z_steps)
    if z_steps < 3:
        raise ValueError("z_steps must be >= 3")
    if z_max <= z_min:
        raise ValueError(f"z_max must be > z_min (got {z_min}, {z_max})")

    if mode == "linear":
        return np.linspace(z_min, z_max, z_steps)
    if mode != "edge":
        raise ValueError(f"Unknown z-grid mode: {mode}")
    if not (0.0 < edge_power < 1.0):
        raise ValueError("edge_power must be in (0,1) for mode='edge'")

    # Uniform in [-1,1], then concave transform to push more points to the edges.
    u = np.linspace(-1.0, 1.0, z_steps)
    v = np.sign(u) * (np.abs(u) ** edge_power)
    mid = 0.5 * (z_max + z_min)
    half_span = 0.5 * (z_max - z_min)
    grid = mid + half_span * v
    grid[0] = z_min
    grid[-1] = z_max
    return grid


async def calibration_sweep(
    z_score_grid,
    benefit=1.0,
    n_reps=10,
    model_name="google/gemini-2.0-flash-001",
    api_base_url="https://openrouter.ai/api/v1",
    max_concurrent=10,
    briefing_kwargs=None,
):
    """Run calibration sweep: generate briefings per z-score and measure join rate.

    Returns DataFrame with z_score, rep, decision, join, direction, clarity, coordination.
    """
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    try:
        semaphore = asyncio.Semaphore(max_concurrent)

        # Allow callers to control generator seed via briefing_kwargs, while
        # still providing a stable default for reproducibility.
        gen_kwargs = dict(briefing_kwargs or {})
        gen_kwargs.setdefault("seed", 42)
        gen = BriefingGenerator(**gen_kwargs)

        rows = []
        tasks = []

        for z in z_score_grid:
            for rep in range(n_reps):
                briefing = gen.generate(z, agent_id=rep, period=rep)
                briefing_text = briefing.render()

                user_prompt = (
                    f"YOUR INTELLIGENCE BRIEFING:\n\n{briefing_text}\n\n"
                    f"What is your decision?"
                )

                tasks.append({
                    "z_score": z,
                    "rep": rep,
                    "direction": briefing.direction,
                    "clarity": briefing.clarity,
                    "coordination": briefing.coordination,
                    "coro": _call_llm(client, model_name, SYSTEM_DECIDE_PURE, user_prompt, semaphore),
                })

        # Execute all calls
        coros = [t["coro"] for t in tasks]
        responses = await asyncio.gather(*coros)

        for task_info, response in zip(tasks, responses):
            api_error = bool(response.startswith("[API Error:"))
            decision = "ERROR" if api_error else _parse_decision(response)
            rows.append({
                "z_score": task_info["z_score"],
                "rep": task_info["rep"],
                "decision": decision,
                # Treat API errors as missing so they don't bias risk-aversion comparisons.
                "join": (1 if decision == "JOIN" else 0) if not api_error else np.nan,
                "api_error": 1 if api_error else 0,
                "direction": task_info["direction"],
                "clarity": task_info["clarity"],
                "coordination": task_info["coordination"],
                "response": response,
            })

        return pd.DataFrame(rows)
    finally:
        # Avoid "Event loop is closed" errors from httpx shutdown when callers
        # use repeated asyncio.run() (as in model benchmarking).
        await client.close()


def _fit_logistic(df):
    """Fit a 2-parameter logistic P(join) = 1/(1 + exp(slope*(z - center))) to raw data.

    Returns (center, slope) via MLE.
    """
    from scipy.optimize import minimize

    z = df["z_score"].values
    y = df["join"].values
    mask = np.isfinite(z) & np.isfinite(y)
    z = z[mask]
    y = y[mask]
    if len(z) < 5 or len(np.unique(z)) < 2:
        return float("nan"), float("nan")

    def neg_ll(params):
        center, slope = params
        p = 1.0 / (1.0 + np.exp(slope * (z - center)))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize(neg_ll, x0=[0.0, 1.5], method="Nelder-Mead")
    return result.x[0], result.x[1]


def _project_sorted_cuts(cuts, min_gap=0.03, low=0.02, high=0.98):
    """Project cutpoints to a valid strictly increasing sequence in (0,1)."""
    x = np.array(cuts, dtype=float)
    if len(x) == 0:
        return tuple()

    # Ensure feasibility for the requested gap.
    span = high - low
    if len(x) > 1:
        max_gap = span / (len(x) - 1)
        min_gap = float(min(min_gap, max_gap - 1e-6))
    min_gap = max(float(min_gap), 1e-6)

    x = np.clip(x, low, high)
    x = np.sort(x)

    # Forward pass: enforce lower spacing.
    for i in range(1, len(x)):
        x[i] = max(x[i], x[i - 1] + min_gap)

    # Backward pass: enforce upper bound while preserving spacing.
    x[-1] = min(x[-1], high)
    for i in range(len(x) - 2, -1, -1):
        x[i] = min(x[i], x[i + 1] - min_gap)

    # Re-anchor first point in range, then forward pass once more.
    x[0] = max(x[0], low)
    for i in range(1, len(x)):
        x[i] = max(x[i], x[i - 1] + min_gap)
    x = np.clip(x, low, high)
    return tuple(float(v) for v in x)


def _fitted_curve_rmse(summary, fitted_center, fitted_slope):
    """RMSE of the fitted logistic against empirical join rates.

    Measures goodness-of-fit to the LLM's own sigmoid curve, not a
    predetermined target.  The sigmoid shape is emergent — calibration
    only shifts cutoff_center; it never optimizes for a particular slope.
    """
    if summary.empty:
        return float("inf")
    z = summary["z_score"].values
    y = summary["join_rate"].values
    mask = np.isfinite(z) & np.isfinite(y)
    if mask.sum() < 3:
        return float("inf")
    z = z[mask]
    y = y[mask]
    if not (np.isfinite(fitted_center) and np.isfinite(fitted_slope)):
        return float("inf")
    predicted = 1.0 / (1.0 + np.exp(fitted_slope * (z - fitted_center)))
    return float(np.sqrt(np.mean((y - predicted) ** 2)))


def _calibration_loss(summary, diagnostics):
    """Scalar objective for selecting calibrated text-threshold candidates.

    The loss penalizes fitted-center offset and API errors.  It does NOT
    include a slope penalty — the sigmoid slope is emergent from the LLM's
    behavior and should not be optimized toward a target.
    """
    fitted_center = diagnostics.get("fitted_center", float("nan"))
    fitted_slope = diagnostics.get("fitted_slope", float("nan"))
    rmse = _fitted_curve_rmse(summary, fitted_center, fitted_slope)
    api_error_rate = diagnostics.get("api_error_rate", 0.0)

    center_penalty = abs(fitted_center) if np.isfinite(fitted_center) else 2.0
    loss = rmse + 0.35 * center_penalty + 1.0 * api_error_rate
    return float(loss), float(rmse)


async def auto_calibrate(
    benefit=1.0,
    model_name="google/gemini-2.0-flash-001",
    api_base_url="https://openrouter.ai/api/v1",
    max_concurrent=10,
    n_reps=8,
    z_steps=21,
    z_min=-2.5,
    z_max=2.5,
    z_grid_mode="linear",
    z_edge_power=0.7,
    max_rounds=5,
    tolerance=0.15,
    initial_cutoff_center=0.0,
    clarity_width=1.0,
    direction_slope=0.8,
    coordination_slope=0.6,
    dissent_floor=0.25,
    mixed_cue_clarity=0.5,
    bottomline_cuts=DEFAULT_BOTTOMLINE_CUTS,
    unclear_cuts=DEFAULT_UNCLEAR_CUTS,
    coordination_cuts=DEFAULT_COORDINATION_CUTS,
    coordination_blend_prob=DEFAULT_COORDINATION_BLEND_PROB,
    language_variant=DEFAULT_LANGUAGE_VARIANT,
    holdout_fraction=0.0,
):
    """Iteratively tune briefing params: sweep, fit logistic, shift cutoff_center until converged.

    Calibration only adjusts cutoff_center — the sigmoid shape is emergent
    from the LLM's behavior.  The slope is never optimized.

    holdout_fraction: when > 0, randomly hold out that fraction of z-grid
        points each round.  Metrics are reported for both train and holdout.

    Returns (best_kwargs, history, final_df).
    """
    language_variant = normalize_language_variant(language_variant)
    cutoff_center = initial_cutoff_center
    bottomline_cuts = _project_sorted_cuts(bottomline_cuts)
    unclear_cuts = _project_sorted_cuts(unclear_cuts)
    coordination_cuts = _project_sorted_cuts(coordination_cuts)
    holdout_rng = np.random.default_rng(42)
    history = []
    final_df = None

    async def _evaluate_candidate(candidate_kwargs, *, grid):
        df = await calibration_sweep(
            z_score_grid=grid,
            benefit=benefit,
            n_reps=n_reps,
            model_name=model_name,
            api_base_url=api_base_url,
            max_concurrent=max_concurrent,
            briefing_kwargs=candidate_kwargs,
        )
        summary, diagnostics = analyze_calibration(df, theoretical_benefit=benefit)
        loss, rmse = _calibration_loss(summary, diagnostics)
        diagnostics["fitted_rmse"] = rmse
        diagnostics["calibration_loss"] = loss
        return df, summary, diagnostics

    for round_i in range(max_rounds):
        full_z_grid = build_z_grid(
            z_min=z_min,
            z_max=z_max,
            z_steps=z_steps,
            mode=z_grid_mode,
            edge_power=z_edge_power,
        )

        # Holdout split
        if holdout_fraction > 0 and len(full_z_grid) >= 5:
            n_holdout = max(1, int(len(full_z_grid) * holdout_fraction))
            indices = np.arange(len(full_z_grid))
            holdout_rng.shuffle(indices)
            holdout_idx = set(indices[:n_holdout].tolist())
            train_idx = sorted(i for i in range(len(full_z_grid)) if i not in holdout_idx)
            holdout_idx_sorted = sorted(holdout_idx)
            z_grid = full_z_grid[train_idx]
            z_grid_holdout = full_z_grid[holdout_idx_sorted]
        else:
            z_grid = full_z_grid
            z_grid_holdout = None

        print(f"\n--- Auto-calibration round {round_i + 1}/{max_rounds} ---")
        print(f"  cutoff_center = {cutoff_center:.3f}, clarity_width = {clarity_width:.3f}")
        mode_str = f"edge(power={z_edge_power:.3f})" if z_grid_mode == "edge" else "linear"
        print(f"  z-grid: {mode_str}, range=({z_min:.2f}, {z_max:.2f})")
        print(f"  {z_steps} z-score points x {n_reps} reps")

        briefing_kwargs = {
            "cutoff_center": cutoff_center,
            "clarity_width": clarity_width,
            "direction_slope": direction_slope,
            "coordination_slope": coordination_slope,
            "dissent_floor": dissent_floor,
            "mixed_cue_clarity": mixed_cue_clarity,
            "bottomline_cuts": bottomline_cuts,
            "unclear_cuts": unclear_cuts,
            "coordination_cuts": coordination_cuts,
            "coordination_blend_prob": coordination_blend_prob,
            "language_variant": language_variant,
        }

        df, summary, diagnostics = await _evaluate_candidate(briefing_kwargs, grid=z_grid)
        final_df = df
        d = diagnostics
        print(f"  loss={d['calibration_loss']:.3f} rmse={d['fitted_rmse']:.3f} "
              f"center={d.get('fitted_center', float('nan')):.3f} "
              f"slope={d.get('fitted_slope', float('nan')):.3f} "
              f"err={d.get('api_error_rate', 0.0):.3f}")

        fitted_center = diagnostics.get("fitted_center", float("nan"))
        fitted_slope = diagnostics.get("fitted_slope", float("nan"))
        mean_join = diagnostics.get("mean_join_rate", float(df["join"].mean()))

        # Holdout evaluation: run on held-out z-grid points
        holdout_metrics = {}
        if z_grid_holdout is not None and len(z_grid_holdout) > 0:
            _, holdout_summary, holdout_diag = await _evaluate_candidate(
                briefing_kwargs, grid=z_grid_holdout,
            )
            holdout_rmse = _fitted_curve_rmse(holdout_summary, fitted_center, fitted_slope)
            holdout_metrics = {
                "holdout_fitted_rmse": holdout_rmse,
                "holdout_mean_join_rate": holdout_diag.get("mean_join_rate", float("nan")),
                "holdout_n_points": len(z_grid_holdout),
            }
            print(f"  holdout: rmse={holdout_rmse:.3f}, "
                  f"n_points={len(z_grid_holdout)}, "
                  f"mean_join={holdout_metrics['holdout_mean_join_rate']:.3f}")

        # Text baseline
        tb = diagnostics.get("text_baseline_corr", float("nan"))
        if np.isfinite(tb):
            print(f"  text_baseline_corr={tb:.3f}, z_direction_corr={diagnostics.get('z_direction_corr', float('nan')):.3f}")

        history.append({
            "round": round_i + 1,
            "cutoff_center": cutoff_center,
            "clarity_width": clarity_width,
            "fitted_center": fitted_center,
            "fitted_slope": fitted_slope,
            "mean_join_rate": mean_join,
            "fitted_rmse": diagnostics.get("fitted_rmse", float("nan")),
            "calibration_loss": diagnostics.get("calibration_loss", float("nan")),
            "api_error_rate": diagnostics.get("api_error_rate", float("nan")),
            "text_baseline_corr": diagnostics.get("text_baseline_corr", float("nan")),
            **holdout_metrics,
        })

        if np.isfinite(fitted_center) and abs(fitted_center) < tolerance:
            print(f"  Converged! |fitted_center| = {abs(fitted_center):.3f} < {tolerance}")
            break

        # Damped correction: shift cutoff_center to cancel the fitted offset
        correction = -fitted_center * 0.7
        cutoff_center += correction
        print(f"  Applying correction: {correction:+.3f} -> new cutoff_center = {cutoff_center:.3f}")

    best_kwargs = {
        "cutoff_center": cutoff_center,
        "clarity_width": clarity_width,
        "direction_slope": direction_slope,
        "coordination_slope": coordination_slope,
        "dissent_floor": dissent_floor,
        "mixed_cue_clarity": mixed_cue_clarity,
        "bottomline_cuts": list(bottomline_cuts),
        "unclear_cuts": list(unclear_cuts),
        "coordination_cuts": list(coordination_cuts),
        "coordination_blend_prob": coordination_blend_prob,
        "language_variant": language_variant,
    }
    print(f"\nFinal parameters: {best_kwargs}")
    return best_kwargs, history, final_df


def text_baseline_diagnostics(df):
    """Compute text-only baseline metrics to test whether the LLM adds value
    beyond reading briefing sentiment.

    The simplest text-only predictor: baseline_join = 1 - direction
    (lower direction = regime described more negatively = more likely to join).

    Returns dict with baseline correlations.
    """
    df = df.copy()
    mask = np.isfinite(df["join"]) & np.isfinite(df["direction"])
    if mask.sum() < 5:
        return {"text_baseline_n": 0}

    sub = df.loc[mask]
    baseline_join = 1.0 - sub["direction"].values
    actual_join = sub["join"].values

    # Correlation of text-only baseline with actual decisions
    baseline_corr = float(np.corrcoef(baseline_join, actual_join)[0, 1])

    # Correlation of z_score with direction (encoder monotonicity)
    z_dir_mask = np.isfinite(sub["z_score"]) & np.isfinite(sub["direction"])
    if z_dir_mask.sum() >= 5:
        z_direction_corr = float(np.corrcoef(
            sub.loc[z_dir_mask, "z_score"].values,
            sub.loc[z_dir_mask, "direction"].values,
        )[0, 1])
    else:
        z_direction_corr = float("nan")

    # Per-z-score baseline prediction (for plotting)
    baseline_by_z = sub.groupby("z_score").agg(
        baseline_join_rate=("direction", lambda x: float(1.0 - x.mean())),
    ).reset_index()

    return {
        "text_baseline_n": int(mask.sum()),
        "text_baseline_corr": baseline_corr,
        "z_direction_corr": z_direction_corr,
        "text_baseline_by_z": baseline_by_z,
    }


def analyze_calibration(df, theoretical_benefit=1.0):
    """Analyze calibration sweep: compute join-rate summary and diagnostics."""
    df = df.copy()
    if "api_error" not in df.columns:
        df["api_error"] = 0

    summary = df.groupby("z_score").agg(
        join_rate=("join", "mean"),
        n=("join", "count"),
        join_se=("join", lambda x: x.std(ddof=1) / np.sqrt(max(x.count(), 1))),
        api_error_rate=("api_error", "mean"),
    ).reset_index()

    theoretical_threshold = 1.0 / (1.0 + theoretical_benefit)

    # Find empirical cutoff: z-score where join_rate crosses theoretical_threshold
    above = summary[summary["join_rate"] >= theoretical_threshold]
    below = summary[summary["join_rate"] < theoretical_threshold]

    if len(above) > 0 and len(below) > 0:
        empirical_cutoff = (above["z_score"].max() + below["z_score"].min()) / 2
    else:
        empirical_cutoff = float("nan")

    # Fit 2-parameter logistic
    fitted_center, fitted_slope = _fit_logistic(df)

    diagnostics = {
        "empirical_cutoff": empirical_cutoff,
        "fitted_center": fitted_center,
        "fitted_slope": fitted_slope,
        "theoretical_threshold_rate": theoretical_threshold,
        "mean_join_rate": df["join"].mean(),
        "api_error_rate": float(df["api_error"].mean()) if "api_error" in df.columns else 0.0,
        "suggestion": "",
    }

    if not np.isnan(empirical_cutoff):
        if empirical_cutoff > 0.3:
            diagnostics["suggestion"] = f"Cutoff too high ({empirical_cutoff:.2f}) — agents too willing."
        elif empirical_cutoff < -0.3:
            diagnostics["suggestion"] = f"Cutoff too low ({empirical_cutoff:.2f}) — agents too reluctant."
        else:
            diagnostics["suggestion"] = f"Cutoff near zero ({empirical_cutoff:.2f}) — looks good."

    # Text-only baseline diagnostics
    baseline = text_baseline_diagnostics(df)
    diagnostics["text_baseline_corr"] = baseline.get("text_baseline_corr", float("nan"))
    diagnostics["z_direction_corr"] = baseline.get("z_direction_corr", float("nan"))
    diagnostics["_text_baseline_by_z"] = baseline.get("text_baseline_by_z", pd.DataFrame())

    return summary, diagnostics


def plot_calibration(summary, diagnostics, theoretical_benefit=1.0, output_path=None):
    """Plot calibration curve: empirical join rate vs z-score with fitted logistic."""
    ensure_agg_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical
    ax.errorbar(summary["z_score"], summary["join_rate"],
                yerr=1.96 * summary["join_se"],
                fmt="o-", color="steelblue", capsize=3, label="Empirical join rate")

    z_fine = np.linspace(summary["z_score"].min(), summary["z_score"].max(), 200)

    # Fitted curve from diagnostics (emergent, not optimized)
    if "fitted_center" in diagnostics and "fitted_slope" in diagnostics:
        fitted_rate = 1.0 / (1.0 + np.exp(diagnostics["fitted_slope"] * (z_fine - diagnostics["fitted_center"])))
        ax.plot(z_fine, fitted_rate, "r-", alpha=0.6,
                label=f"Fitted (center={diagnostics['fitted_center']:.2f}, slope={diagnostics['fitted_slope']:.2f})")

    # Threshold line
    threshold = 1.0 / (1.0 + theoretical_benefit)
    ax.axhline(threshold, color="red", linestyle=":", alpha=0.5,
               label=f"Indifference rate = {threshold:.2f}")

    # Empirical cutoff
    if not np.isnan(diagnostics.get("empirical_cutoff", float("nan"))):
        ax.axvline(diagnostics["empirical_cutoff"], color="green", linestyle="--",
                   alpha=0.5, label=f"Empirical cutoff = {diagnostics['empirical_cutoff']:.2f}")

    # Text-only baseline prediction (1 - direction)
    baseline_by_z = diagnostics.get("_text_baseline_by_z", pd.DataFrame())
    if not baseline_by_z.empty:
        baseline_corr = diagnostics.get("text_baseline_corr", float("nan"))
        label = f"Text baseline (1-direction, r={baseline_corr:.2f})" if np.isfinite(baseline_corr) else "Text baseline (1-direction)"
        ax.plot(baseline_by_z["z_score"], baseline_by_z["baseline_join_rate"],
                "s--", color="orange", alpha=0.6, markersize=4, label=label)

    ax.set_xlabel("z-score (signal strength)", fontsize=12)
    ax.set_ylabel("Join rate", fontsize=12)
    ax.set_title("Calibration: LLM Join Rate vs Signal Strength", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    return fig, ax
