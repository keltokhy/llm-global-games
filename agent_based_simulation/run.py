"""
Entry points for the LLM global games paper.

Usage:
    # Step 0: Auto-calibrate per model (saves to calibrated_index.json)
    uv run python -m agent_based_simulation.run autocalibrate --model google/gemini-2.0-flash-001
    uv run python -m agent_based_simulation.run autocalibrate --model anthropic/claude-3.5-haiku

    # Step 1: Manual calibration sweep
    uv run python -m agent_based_simulation.run calibrate

    # Step 2-4: Run experiments with --load-calibrated to use per-model params
    uv run python -m agent_based_simulation.run pure --model google/gemini-2.0-flash-001 --load-calibrated
    uv run python -m agent_based_simulation.run comm --model google/gemini-2.0-flash-001 --load-calibrated
    uv run python -m agent_based_simulation.run both --model google/gemini-2.0-flash-001 --load-calibrated
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import model_slug, parse_float_list, resolve_model_output_dir, add_common_args, OUTPUT_DIR


def _save_calibrated_params(output_dir: Path, model_name: str, params: dict):
    """Save calibrated params for a model and update the shared index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(model_name)

    # Per-model file
    per_model_path = output_dir / f"calibrated_params_{slug}.json"
    with open(per_model_path, "w") as f:
        json.dump(params, f, indent=2)

    # Update shared index: model_name -> params
    index_path = output_dir / "calibrated_index.json"
    index = {}
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    index[model_name] = params
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return per_model_path, index_path


def _load_calibrated_params(calibration_dir: Path, model_name: str) -> dict:
    """Load calibrated params for a specific model from calibration_dir."""
    slug = model_slug(model_name)

    # Search order: explicit calibration_dir, then default model output dir
    search_dirs = [Path(calibration_dir)]
    default_model_dir = OUTPUT_DIR / slug
    if default_model_dir != search_dirs[0] and default_model_dir.exists():
        search_dirs.append(default_model_dir)

    for cdir in search_dirs:
        # Try shared index
        index_path = cdir / "calibrated_index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            if model_name in index:
                print(f"  Loaded calibrated params for {model_name} from {cdir}")
                return index[model_name]

        # Try per-model file
        per_model_path = cdir / f"calibrated_params_{slug}.json"
        if per_model_path.exists():
            with open(per_model_path) as f:
                print(f"  Loaded calibrated params for {model_name} from {per_model_path}")
                return json.load(f)

    available = []
    index_path = Path(calibration_dir) / "calibrated_index.json"
    if index_path.exists():
        with open(index_path) as f:
            available = list(json.load(f).keys())
    raise FileNotFoundError(
        f"No calibrated params for model '{model_name}' in {search_dirs}. "
        f"Available: {available}. Run autocalibrate first."
    )


def run_calibrate(args):
    """Run calibration sweep."""
    from .calibration import calibration_sweep, analyze_calibration, plot_calibration, build_z_grid

    z_mode = f"edge (power={args.z_edge_power:.3f})" if args.z_grid_mode == "edge" else "linear"
    print(f"Running calibration sweep...\n"
          f"  z-score grid: {args.z_min} to {args.z_max}, {args.z_steps} steps ({z_mode})\n"
          f"  Reps per z-score: {args.n_reps}\n"
          f"  Model: {args.model}, Benefit: {args.benefit}\n"
          f"  Language variant: {args.language_variant}\n")

    z_grid = build_z_grid(
        z_min=args.z_min,
        z_max=args.z_max,
        z_steps=args.z_steps,
        mode=args.z_grid_mode,
        edge_power=args.z_edge_power,
    )

    df = asyncio.run(calibration_sweep(
        z_score_grid=z_grid,
        benefit=args.benefit,
        n_reps=args.n_reps,
        model_name=args.model,
        api_base_url=args.api_base_url,
        max_concurrent=args.max_concurrent,
        briefing_kwargs={
            "cutoff_center": args.cutoff_center,
            "clarity_width": args.clarity_width,
            "direction_slope": args.direction_slope,
            "coordination_slope": args.coordination_slope,
            "dissent_floor": args.dissent_floor,
            "mixed_cue_clarity": args.mixed_cue_clarity,
            "bottomline_cuts": parse_float_list(args.bottomline_cuts),
            "unclear_cuts": parse_float_list(args.unclear_cuts),
            "coordination_cuts": parse_float_list(args.coordination_cuts),
            "coordination_blend_prob": args.coordination_blend_prob,
            "language_variant": args.language_variant,
        },
    ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    raw_path = output_dir / "calibration_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw results saved to: {raw_path}")

    # Analyze
    summary, diagnostics = analyze_calibration(df, theoretical_benefit=args.benefit)

    summary_path = output_dir / "calibration_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nCalibration summary:")
    print(summary.to_string(index=False))
    print(f"\nDiagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")

    # Plot
    plot_calibration(summary, diagnostics, theoretical_benefit=args.benefit,
                     output_path=str(output_dir / "calibration_curve.png"))


def run_autocalibrate(args):
    """Run automated calibration loop."""
    from .calibration import auto_calibrate, analyze_calibration, plot_calibration

    holdout = getattr(args, 'holdout_fraction', 0.0)
    print(f"Running auto-calibration...\n"
          f"  Model: {args.model}, Benefit: {args.benefit}\n"
          f"  Language variant: {args.language_variant}\n"
          f"  Max rounds: {args.max_rounds}, Tolerance: {args.tolerance}, Reps: {args.n_reps}"
          + (f"\n  Holdout fraction: {holdout:.0%}" if holdout > 0 else "")
          + "\n")

    best_kwargs, history, final_df = asyncio.run(auto_calibrate(
        benefit=args.benefit,
        model_name=args.model,
        api_base_url=args.api_base_url,
        max_concurrent=args.max_concurrent,
        n_reps=args.n_reps,
        z_steps=args.z_steps,
        z_min=args.z_min,
        z_max=args.z_max,
        z_grid_mode=args.z_grid_mode,
        z_edge_power=args.z_edge_power,
        max_rounds=args.max_rounds,
        tolerance=args.tolerance,
        initial_cutoff_center=args.cutoff_center,
        clarity_width=args.clarity_width,
        direction_slope=args.direction_slope,
        coordination_slope=args.coordination_slope,
        dissent_floor=args.dissent_floor,
        mixed_cue_clarity=args.mixed_cue_clarity,
        bottomline_cuts=parse_float_list(args.bottomline_cuts),
        unclear_cuts=parse_float_list(args.unclear_cuts),
        coordination_cuts=parse_float_list(args.coordination_cuts),
        coordination_blend_prob=args.coordination_blend_prob,
        language_variant=args.language_variant,
        holdout_fraction=holdout,
    ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save history
    history_df = pd.DataFrame(history)
    history_path = output_dir / "autocalibrate_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\nCalibration history saved to: {history_path}")
    print(history_df.to_string(index=False))

    # Save final sweep data
    final_df.to_csv(output_dir / "autocalibrate_final_raw.csv", index=False)

    # Analyze and plot the final sweep
    summary, diagnostics = analyze_calibration(final_df, theoretical_benefit=args.benefit)
    plot_calibration(summary, diagnostics, theoretical_benefit=args.benefit,
                     output_path=str(output_dir / "autocalibrate_final_curve.png"))

    # Save tuned parameters per-model and update shared index
    per_model_path, index_path = _save_calibrated_params(output_dir, args.model, best_kwargs)
    print(f"\nTuned parameters saved to: {per_model_path}")
    print(f"Calibration index updated: {index_path}")
    print(f"  Model: {args.model}")
    print(f"  Use --load-calibrated in subsequent steps to auto-load these params")
    print(f"  Or manually: --cutoff-center {best_kwargs['cutoff_center']:.3f} "
          f"--clarity-width {best_kwargs['clarity_width']:.3f} "
          f"--direction-slope {best_kwargs['direction_slope']:.3f} "
          f"--dissent-floor {best_kwargs['dissent_floor']:.3f}")


def _apply_calibrated_params(args):
    """If --load-calibrated is set, override briefing params from the calibration index."""
    if not getattr(args, "load_calibrated", False):
        return
    cal_dir = getattr(args, "calibration_dir", None) or args.output_dir
    calibrated = _load_calibrated_params(cal_dir, args.model)
    # Override briefing generator params with calibrated values
    scalar_keys = [
        "cutoff_center", "clarity_width", "direction_slope",
        "coordination_slope", "dissent_floor", "mixed_cue_clarity",
        "coordination_blend_prob", "language_variant",
    ]
    for key in scalar_keys:
        if key in calibrated:
            setattr(args, key, calibrated[key])
    # List-valued params: store as lists (BriefingGenerator / parse_float_list handle both)
    for key in ("bottomline_cuts", "unclear_cuts", "coordination_cuts"):
        if key in calibrated:
            setattr(args, key, calibrated[key])
    print(f"  Loaded calibrated params for model: {args.model}")
    print(f"    cutoff_center={args.cutoff_center:.3f}, "
          f"clarity_width={args.clarity_width:.3f}, "
          f"direction_slope={args.direction_slope:.3f}")


def _load_fixed_messages(path: str) -> dict[tuple[int, int], dict[int, str]]:
    """Load pre-recorded messages from a communication experiment log JSON.

    Returns dict mapping (country, period) -> {agent_id: message_sent}.
    """
    with open(path) as f:
        log_entries = json.load(f)

    messages_by_period = {}
    for entry in log_entries:
        key = (entry["country"], entry["period"])
        agent_msgs = {}
        for agent_data in entry.get("agents", []):
            aid = agent_data["id"]
            msg = agent_data.get("message_sent", "")
            if msg:
                agent_msgs[aid] = msg
        if agent_msgs:
            messages_by_period[key] = agent_msgs
    return messages_by_period


def run_experiment(args, treatment, signal_mode="normal"):
    """Run an experiment (pure or communication). All periods run in parallel."""
    from .experiment import (
        Agent, run_pure_global_game, run_communication_game,
        PeriodResult,
    )
    from .briefing import BriefingGenerator
    from .runtime import build_network
    from openai import AsyncOpenAI

    # Load calibrated params if requested
    _apply_calibrated_params(args)

    # Load fixed messages if provided (for fixed-message surveillance test)
    fixed_messages_map = None
    if getattr(args, 'fixed_messages', None):
        fixed_messages_map = _load_fixed_messages(args.fixed_messages)
        print(f"  Loaded fixed messages from: {args.fixed_messages}")
        print(f"  Periods with messages: {len(fixed_messages_map)}")

    async def _run():
        api_key = os.environ.get("OPENROUTER_API_KEY", "") or "not-needed"
        client = AsyncOpenAI(base_url=args.api_base_url, api_key=api_key)
        semaphore = asyncio.Semaphore(args.max_concurrent)

        briefing_gen = BriefingGenerator(
            cutoff_center=args.cutoff_center,
            clarity_width=args.clarity_width,
            direction_slope=args.direction_slope,
            coordination_slope=args.coordination_slope,
            dissent_floor=args.dissent_floor,
            mixed_cue_clarity=args.mixed_cue_clarity,
            bottomline_cuts=parse_float_list(args.bottomline_cuts),
            unclear_cuts=parse_float_list(args.unclear_cuts),
            coordination_cuts=parse_float_list(args.coordination_cuts),
            coordination_blend_prob=args.coordination_blend_prob,
            language_variant=args.language_variant,
            seed=args.seed,
        )

        # Build network (used only in communication treatment)
        adjacency, graph = build_network(
            args.n_agents, n_neighbors=getattr(args, 'n_neighbors', 4),
            rewire_prob=0.3, seed=args.seed,
        )

        # Pre-compute all random draws so parallelization doesn't change RNG sequence
        rng = np.random.default_rng(args.seed)
        tasks_spec = []
        for c in range(args.n_countries):
            b_mean = rng.normal(0.5, 0.4)
            z_base = rng.normal(0.0, 0.3)
            for t in range(args.n_periods):
                z = z_base + rng.normal(0, 0.05)
                benefit = b_mean + rng.normal(0, 0.15)
                theta = rng.normal(z, 1.0)  # tau = 1
                tasks_spec.append({"c": c, "t": t, "z": z, "benefit": benefit, "theta": theta})

        n_total = len(tasks_spec)
        completed = [0]  # mutable counter for progress

        # Cross-period scramble: pre-generate all briefings, shuffle across
        # periods within each country so the θ→briefing-distribution link breaks.
        briefing_override_map = {}  # (country, period) -> list of Briefing
        if signal_mode == "scramble":
            from collections import defaultdict
            country_briefings = defaultdict(list)  # country -> flat list of Briefing
            country_keys = defaultdict(list)       # country -> list of (c, t) keys

            for spec in tasks_spec:
                period_rng = np.random.default_rng(
                    hash((spec["c"], spec["t"], "signals")) % 2**32
                )
                for i in range(args.n_agents):
                    signal = spec["theta"] + period_rng.normal(0, args.sigma)
                    z_score = (signal - spec["z"]) / args.sigma
                    briefing = briefing_gen.generate(z_score, i, spec["t"])
                    country_briefings[spec["c"]].append(briefing)
                country_keys[spec["c"]].append((spec["c"], spec["t"]))

            # Shuffle all briefings within each country across periods
            shuffle_rng = np.random.default_rng(args.seed + 999)
            for c, briefings in country_briefings.items():
                shuffle_rng.shuffle(briefings)
                keys = country_keys[c]
                for idx, (cc, tt) in enumerate(keys):
                    start = idx * args.n_agents
                    briefing_override_map[(cc, tt)] = briefings[start:start + args.n_agents]

            print(f"  Cross-period scramble: {sum(len(v) for v in country_briefings.values())} briefings shuffled")

        # Build mixed-model assignment list (cycles through all models)
        _all_models = [args.model] + (args.mixed_models or [])
        _mixed = len(_all_models) > 1

        async def _run_one(spec, tx):
            """Run one (country, period, treatment) with fresh agents."""
            # Each task gets its own agent objects to avoid mutation conflicts
            task_agents = [Agent(agent_id=i, neighbors=adjacency[i]) for i in range(args.n_agents)]
            if _mixed:
                for agent in task_agents:
                    agent.model = _all_models[agent.agent_id % len(_all_models)]
            n_prop = getattr(args, 'n_propaganda', 0)
            if n_prop > 0 and tx == "comm":
                for agent in task_agents[:n_prop]:
                    agent.is_propaganda = True
            _personas = getattr(args, 'personas', None)
            if _personas:
                for agent in task_agents:
                    agent.persona = _personas[agent.agent_id % len(_personas)]

            overrides = briefing_override_map.get((spec["c"], spec["t"]))

            if tx == "pure":
                result = await run_pure_global_game(
                    task_agents, spec["theta"], spec["z"], args.sigma, spec["benefit"],
                    briefing_gen, client, args.model, semaphore, spec["c"], spec["t"],
                    llm_max_retries=args.llm_max_retries,
                    llm_empty_retries=args.llm_empty_retries,
                    cost=args.cost, signal_mode=signal_mode,
                    briefing_overrides=overrides,
                    group_size_info=getattr(args, 'group_size_info', False),
                    elicit_beliefs=getattr(args, 'elicit_beliefs', False),
                    elicit_second_order=getattr(args, 'elicit_second_order', False),
                )
            else:
                # Resolve fixed messages for this (country, period) if available
                period_fixed = None
                if fixed_messages_map is not None:
                    period_fixed = fixed_messages_map.get((spec["c"], spec["t"]))

                result = await run_communication_game(
                    task_agents, spec["theta"], spec["z"], args.sigma, spec["benefit"],
                    briefing_gen, client, args.model, semaphore, spec["c"], spec["t"],
                    llm_max_retries=args.llm_max_retries,
                    llm_empty_retries=args.llm_empty_retries,
                    cost=args.cost, signal_mode=signal_mode,
                    surveillance=getattr(args, 'surveillance', False),
                    group_size_info=getattr(args, 'group_size_info', False),
                    elicit_beliefs=getattr(args, 'elicit_beliefs', False),
                    elicit_second_order=getattr(args, 'elicit_second_order', False),
                    fixed_messages=period_fixed,
                )

            completed[0] += 1
            print(f"  [{tx}] C{spec['c']} P{spec['t']}: "
                  f"join={result.join_fraction:.2f} "
                  f"theta={spec['theta']:.2f} theta*={result.theta_star:.2f} "
                  f"success={result.coup_success}  "
                  f"({completed[0]}/{n_total})")
            return result

        # Launch all (country, period, treatment) combinations in parallel
        coros = []
        for spec in tasks_spec:
            if treatment in ("pure", "both"):
                coros.append(_run_one(spec, "pure"))
            if treatment in ("comm", "both"):
                coros.append(_run_one(spec, "comm"))

        try:
            results = await asyncio.gather(*coros)
            return list(results)
        finally:
            # Prevent "Event loop is closed" warnings from httpx transport cleanup.
            await client.close()

    n_tasks = args.n_countries * args.n_periods
    if treatment == "both":
        n_tasks *= 2
    n_calls = n_tasks * args.n_agents
    if treatment in ("comm", "both"):
        # communication treatment has 2 LLM rounds per agent
        comm_tasks = args.n_countries * args.n_periods
        n_calls += comm_tasks * args.n_agents

    mode_label = f" [signal_mode={signal_mode}]" if signal_mode != "normal" else ""
    mixed_label = ""
    if args.mixed_models:
        all_m = [args.model] + args.mixed_models
        mixed_label = f"\n  Mixed models: {all_m} ({args.n_agents // len(all_m)}-{args.n_agents // len(all_m) + 1} agents each)"
    print(f"Running {treatment} experiment{mode_label} (fully parallel)...\n"
          f"  Agents: {args.n_agents}, Countries: {args.n_countries}, Periods: {args.n_periods}\n"
          f"  Model: {args.model}, Sigma: {args.sigma}, Language: {args.language_variant}"
          f"{mixed_label}\n"
          f"  Tasks: {n_tasks}, ~{n_calls} LLM calls, max_concurrent: {args.max_concurrent}\n"
          f"  Retry policy: api={args.llm_max_retries}, empty={args.llm_empty_retries}\n")

    results = asyncio.run(_run())

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_rows = [
        {
            "country": r.country, "period": r.period, "treatment": r.treatment,
            "theta": r.theta, "theta_star": r.theta_star, "z": r.z,
            "benefit": r.benefit, "n_join": r.n_join,
            "join_fraction": r.join_fraction,
            "n_valid": r.n_valid, "n_api_error": r.n_api_error,
            "n_unparseable": r.n_unparseable,
            "join_fraction_valid": r.join_fraction_valid,
            "api_error_rate": r.api_error_rate,
            "unparseable_rate": r.unparseable_rate,
            "coup_success": r.coup_success,
            "theoretical_attack": r.theoretical_attack,
        }
        for r in results
    ]

    summary_df = pd.DataFrame(summary_rows)
    file_label = signal_mode if signal_mode != "normal" else treatment

    # ── Append mode: merge with existing data ───────────────────────
    logs = [{**row, "agents": r.agents} for row, r in zip(summary_rows, results)]
    if getattr(args, "append", False):
        summary_path = output_dir / f"experiment_{file_label}_summary.csv"
        if summary_path.exists():
            old_df = pd.read_csv(summary_path)
            summary_df = pd.concat([old_df, summary_df], ignore_index=True)
            print(f"\n  Appended {len(summary_rows)} new rows to {len(old_df)} existing → {len(summary_df)} total")

        log_path = output_dir / f"experiment_{file_label}_log.json"
        if log_path.exists():
            with open(log_path) as f:
                old_logs = json.load(f)
            logs = old_logs + logs

    summary_path = output_dir / f"experiment_{file_label}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Full logs
    log_path = output_dir / f"experiment_{file_label}_log.json"
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2, default=str)
    print(f"Full logs saved to: {log_path}")

    # Quick stats
    stats = (f"\nResults ({file_label}):\n"
             f"  Total periods: {len(results)}\n"
             f"  Mean join fraction: {summary_df['join_fraction'].mean():.3f}")
    for col, label in [("join_fraction_valid", "Mean join fraction (valid only)"),
                        ("api_error_rate", "Mean API error rate")]:
        if col in summary_df.columns:
            val = summary_df[col].mean(skipna=True)
            if pd.notna(val):
                stats += f"\n  {label}: {val:.3f}"
    stats += f"\n  Coup success rate: {summary_df['coup_success'].mean():.3f}"
    print(stats)


def main():
    parser = argparse.ArgumentParser(description="LLM Global Games Experiments")
    parser.add_argument("command", choices=["autocalibrate", "calibrate", "pure", "comm", "both", "scramble", "flip"],
                        help="Which step to run")
    add_common_args(parser)
    # Experiment-specific
    parser.add_argument("--n-countries", type=int, default=5)
    parser.add_argument("--n-periods", type=int, default=20)
    # Calibration-specific
    parser.add_argument("--z-min", type=float, default=-3.0)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--z-steps", type=int, default=21)
    parser.add_argument("--z-grid-mode", type=str, choices=["linear", "edge"], default="linear",
                        help="How to place z-grid points")
    parser.add_argument("--z-edge-power", type=float, default=0.7,
                        help="Edge density power for --z-grid-mode edge (0<p<1)")
    parser.add_argument("--n-reps", type=int, default=10)
    # Auto-calibration specific
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--holdout-fraction", type=float, default=0.0,
                        help="Fraction of z-grid points to hold out for validation during auto-calibration (0.0 = no holdout)")
    parser.add_argument("--append", action="store_true",
                        help="Append new results to existing CSVs/logs instead of overwriting")
    parser.add_argument("--mixed-models", nargs="+", default=None,
                        help="Additional models for mixed-model games. Agents are split "
                             "evenly: first chunk uses --model, rest use these models. "
                             "E.g. --model A --mixed-models B C gives ~1/3 A, ~1/3 B, ~1/3 C")
    parser.add_argument("--n-propaganda", type=int, default=0,
                        help="Number of propaganda (regime plant) agents in comm games. "
                             "These agents send pro-regime messages and always STAY.")
    parser.add_argument("--surveillance", action="store_true",
                        help="Tell agents their messages are monitored by regime security.")
    parser.add_argument("--n-neighbors", type=int, default=4,
                        help="Number of neighbors per agent in Watts-Strogatz network (default: 4)")
    parser.add_argument("--personas", nargs="+", default=None,
                        help="Persona roles to assign round-robin to agents. "
                             "E.g. --personas 'military officer' 'university student' 'business owner'")
    parser.add_argument("--group-size-info", action="store_true",
                        help="Tell agents how many citizens are in the group (enables strategic reasoning about coordination thresholds)")
    parser.add_argument("--elicit-beliefs", action="store_true",
                        help="After each decision, ask agents for P(uprising succeeds) on 0-100 scale")
    parser.add_argument("--elicit-second-order", action="store_true",
                        help="After each decision, ask agents what %% of citizens will JOIN (0-100 scale)")
    parser.add_argument("--fixed-messages", type=str, default=None,
                        help="Path to a communication experiment log JSON. Replaces live message "
                             "generation with pre-recorded messages (for fixed-message surveillance test)")

    args = parser.parse_args()
    from .briefing import normalize_language_variant
    args.language_variant = normalize_language_variant(args.language_variant)

    # Auto-create model-specific output subfolder
    args.output_dir = str(resolve_model_output_dir(args.output_dir, args.model))

    if args.command == "autocalibrate":
        run_autocalibrate(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    elif args.command in ("pure", "comm", "both"):
        run_experiment(args, args.command)
    elif args.command == "scramble":
        run_experiment(args, "pure", signal_mode="scramble")
    elif args.command == "flip":
        run_experiment(args, "pure", signal_mode="flip")


if __name__ == "__main__":
    main()
