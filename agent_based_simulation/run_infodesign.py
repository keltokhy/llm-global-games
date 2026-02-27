"""
Experiment runner for Paper 2: Information design in LLM global games.

Runs a controlled experimental grid:
  - 9 θ values spanning [θ* - 0.30, θ* + 0.30]
  - 6 designs: baseline, stability, instability, public_signal, scramble, flip
  - 30 repetitions per cell (25 agents each)

Usage:
    # Run all designs for a model
    uv run python -m agent_based_simulation.run_infodesign \
        --model google/gemini-2.0-flash-001 --load-calibrated

    # Run specific designs
    uv run python -m agent_based_simulation.run_infodesign \
        --model google/gemini-2.0-flash-001 --load-calibrated \
        --designs baseline stability instability

    # Custom θ grid
    uv run python -m agent_based_simulation.run_infodesign \
        --model google/gemini-2.0-flash-001 --load-calibrated \
        --theta-points 13 --theta-range 0.40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .infodesign import (
    ALL_DESIGNS,
    InfoDesignConfig,
    ThetaAdaptiveBriefingGenerator,
    PublicSignal,
    base_params_from_calibrated,
)
from .run import _load_calibrated_params
from .runtime import parse_float_list, resolve_model_output_dir, add_common_args, join_fraction_column, deterministic_hash


def run_infodesign(args):
    """Run the information design experiment grid."""
    from .experiment import Agent, run_pure_global_game, run_communication_game, PeriodResult
    from .briefing import BriefingGenerator
    from .runtime import theta_star_baseline, build_network
    from openai import AsyncOpenAI

    is_comm = getattr(args, 'treatment', 'pure') == 'comm'

    # ── Load calibrated params ───────────────────────────────────────
    calibration_dir = Path(args.calibration_dir or args.output_dir)
    if args.load_calibrated:
        calibrated = _load_calibrated_params(calibration_dir, args.model)
        base_params = base_params_from_calibrated(calibrated, seed=args.seed)
        print(f"  Loaded calibrated params for {args.model}")
    else:
        base_params = {
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
            "seed": args.seed,
        }

    # ── Compute θ* and z-center ──────────────────────────────────────
    theta_star = theta_star_baseline(max(args.benefit, 1e-6))
    z_center = args.z_center if args.z_center is not None else theta_star
    theta_grid = np.linspace(
        theta_star - args.theta_range,
        theta_star + args.theta_range,
        args.theta_points,
    )
    print(f"  θ* = {theta_star:.4f}, z_center = {z_center:.4f}")
    print(f"  θ grid: [{theta_grid[0]:.3f}, ..., {theta_grid[-1]:.3f}] "
          f"({len(theta_grid)} points)")

    # ── Resolve designs to run ───────────────────────────────────────
    # "scramble" and "flip" are falsification variants of *each* design.
    # They use signal_mode="scramble"/"flip" with the baseline config.
    design_names = args.designs
    run_specs = []  # (design_name, config_or_None, signal_mode)

    for name in design_names:
        if name == "scramble":
            run_specs.append(("scramble", None, "scramble"))
        elif name == "flip":
            run_specs.append(("flip", None, "flip"))
        elif name in ALL_DESIGNS:
            config = ALL_DESIGNS[name]
            # Override bandwidth if --bandwidth CLI was provided
            if config is not None and args.bandwidth is not None:
                from dataclasses import replace
                config = replace(config, bandwidth=args.bandwidth)
            run_specs.append((name, config, "normal"))
        else:
            raise ValueError(f"Unknown design: {name}. "
                             f"Available: {list(ALL_DESIGNS) + ['scramble', 'flip']}")

    n_cells = len(run_specs) * len(theta_grid) * args.reps
    n_calls = n_cells * args.n_agents
    print(f"\n  Designs: {[s[0] for s in run_specs]}")
    print(f"  Reps per cell: {args.reps}")
    print(f"  Total cells: {n_cells}, ~{n_calls} LLM calls")
    print()

    # ── Pre-generate cross-θ scramble overrides ─────────────────────
    # Paper 1's scramble shuffles briefings across periods (different θ).
    # In infodesign, the equivalent: pre-generate briefings for all (θ, rep)
    # cells, shuffle across θ-cells, then redistribute.
    scramble_overrides = {}  # (theta_idx, rep) -> list[Briefing]
    has_scramble = any(name == "scramble" for name, _, _ in run_specs)
    if has_scramble:
        from .briefing import BriefingGenerator as _BG
        _scramble_gen = _BG(**base_params)
        _all_briefings = []  # flat list of all briefings
        _cell_keys = []      # parallel list of (theta_idx, rep) keys

        for ti, theta in enumerate(theta_grid):
            for rep in range(args.reps):
                cell_rng = np.random.default_rng(
                    deterministic_hash((rep, 0, "signals")) % 2**32
                )
                for agent_id in range(args.n_agents):
                    signal = theta + cell_rng.normal(0, args.sigma)
                    z_score = (signal - z_center) / args.sigma
                    briefing = _scramble_gen.generate(z_score, agent_id, 0)
                    _all_briefings.append(briefing)
                _cell_keys.append((ti, rep))

        # Shuffle all briefings across all θ-cells (breaks θ→briefing link)
        shuffle_rng = np.random.default_rng(args.seed + 999)
        shuffle_rng.shuffle(_all_briefings)

        # Redistribute to cells
        for idx, key in enumerate(_cell_keys):
            start = idx * args.n_agents
            scramble_overrides[key] = _all_briefings[start:start + args.n_agents]

        print(f"  Cross-θ scramble: {len(_all_briefings)} briefings shuffled "
              f"across {len(_cell_keys)} cells")

    # ── Network (for communication treatment) ───────────────────────
    adjacency = {}
    if is_comm:
        adjacency, _ = build_network(
            args.n_agents, n_neighbors=4, rewire_prob=0.3, seed=args.seed,
        )
        print(f"  Treatment: communication (4 neighbors, Watts-Strogatz)")

    # ── Async runner ─────────────────────────────────────────────────
    async def _run():
        api_key = os.environ.get("OPENROUTER_API_KEY", "") or "not-needed"
        client = AsyncOpenAI(base_url=args.api_base_url, api_key=api_key)
        semaphore = asyncio.Semaphore(args.max_concurrent)

        all_results = []
        completed = [0]

        async def _run_one_cell(
            design_name: str,
            config: InfoDesignConfig | None,
            signal_mode: str,
            theta: float,
            theta_idx: int,
            rep: int,
        ):
            """Run one cell of the experimental grid."""
            if is_comm:
                agents = [Agent(agent_id=i, neighbors=list(adjacency[i]))
                          for i in range(args.n_agents)]
            else:
                agents = [Agent(agent_id=i, neighbors=[]) for i in range(args.n_agents)]
            n_prop = getattr(args, 'n_propaganda', 0)
            if n_prop > 0 and is_comm:
                for agent in agents[:n_prop]:
                    agent.is_propaganda = True

            # Build briefing generator for this design
            # Inject provenance/rhetoric overrides from config into params
            cell_base_params = dict(base_params)
            if config is not None:
                if config.source_header:
                    cell_base_params["source_header"] = config.source_header
                if config.rhetoric_bias != 0.0:
                    cell_base_params["rhetoric_bias"] = config.rhetoric_bias

            if config is not None and config.name != "baseline":
                gen = ThetaAdaptiveBriefingGenerator(
                    cell_base_params, config, theta_star,
                )
                gen.set_theta(theta)
            else:
                gen = BriefingGenerator(**cell_base_params)

            # Scramble: use pre-generated cross-θ overrides
            briefing_overrides = None
            if design_name == "scramble":
                briefing_overrides = scramble_overrides.get((theta_idx, rep))

            # Public signal injection: pre-generate briefings with public suffix
            elif config is not None and config.inject_public_signal:
                pub = PublicSignal(
                    base_params,
                    n_observations=config.public_signal_n_observations,
                    seed=args.seed + rep,
                )
                public_text = pub.generate(
                    theta, z=z_center, sigma=args.sigma, period=rep,
                    bulletin_seed=deterministic_hash((args.seed, rep, design_name)) % (2**31),
                )

                # Pre-generate briefings and stamp the public suffix on each
                rng = np.random.default_rng(
                    deterministic_hash((rep, 0, "signals")) % 2**32
                )
                briefing_overrides = []
                for agent in agents:
                    agent.signal = theta + rng.normal(0, args.sigma)
                    agent.z_score = (agent.signal - z_center) / args.sigma
                    briefing = gen.generate(agent.z_score, agent.agent_id, 0)
                    briefing._public_suffix = public_text
                    briefing_overrides.append(briefing)

            # Within-briefing observation shuffle
            elif config is not None and getattr(config, 'shuffle_observations', False):
                rng = np.random.default_rng(
                    deterministic_hash((rep, 0, "signals")) % 2**32
                )
                shuffle_rng = np.random.default_rng(
                    deterministic_hash((rep, 0, "shuffle_obs")) % 2**32
                )
                briefing_overrides = []
                for agent in agents:
                    agent.signal = theta + rng.normal(0, args.sigma)
                    agent.z_score = (agent.signal - z_center) / args.sigma
                    briefing = gen.generate(agent.z_score, agent.agent_id, 0)
                    obs_copy = list(briefing.observations)
                    shuffle_rng.shuffle(obs_copy)
                    briefing.observations = obs_copy
                    briefing_overrides.append(briefing)

            # Domain-group scramble: scramble specific domain observations across agents
            elif config is not None and getattr(config, 'scramble_domain_indices', None):
                rng = np.random.default_rng(
                    deterministic_hash((rep, 0, "signals")) % 2**32
                )
                scramble_rng = np.random.default_rng(
                    deterministic_hash((rep, 0, "domain_scramble")) % 2**32
                )
                # Generate all briefings normally
                temp_briefings = []
                for agent in agents:
                    agent.signal = theta + rng.normal(0, args.sigma)
                    agent.z_score = (agent.signal - z_center) / args.sigma
                    briefing = gen.generate(agent.z_score, agent.agent_id, 0)
                    temp_briefings.append(briefing)
                # Scramble specified domain observations across agents
                for idx in config.scramble_domain_indices:
                    domain_obs = [b.observations[idx] for b in temp_briefings]
                    scramble_rng.shuffle(domain_obs)
                    for b, obs in zip(temp_briefings, domain_obs):
                        b.observations[idx] = obs
                briefing_overrides = temp_briefings

            # Run the game — scramble uses "normal" mode since
            # we handle it via briefing_overrides, not experiment.py's shuffle.
            effective_signal_mode = "normal" if design_name == "scramble" else signal_mode
            game_kwargs = dict(
                agents=agents, theta=theta, z=z_center, sigma=args.sigma,
                benefit=args.benefit, briefing_gen=gen, client=client,
                model_name=args.model, semaphore=semaphore, country=rep,
                period=0, llm_max_retries=args.llm_max_retries,
                llm_empty_retries=args.llm_empty_retries,
                cost=args.cost, signal_mode=effective_signal_mode,
                briefing_overrides=briefing_overrides,
                group_size_info=getattr(args, 'group_size_info', False),
                elicit_beliefs=getattr(args, 'elicit_beliefs', False),
                temperature=getattr(args, 'temperature', 0.7),
            )
            if is_comm:
                result = await run_communication_game(
                    **game_kwargs,
                    surveillance=getattr(args, 'surveillance', False),
                    surveillance_mode=getattr(args, 'surveillance_mode', 'full'),
                )
            else:
                result = await run_pure_global_game(**game_kwargs)

            return {
                "design": design_name,
                "treatment": "comm" if is_comm else "pure",
                "signal_mode": signal_mode,
                "theta": theta,
                "theta_star": theta_star,
                "theta_relative": theta - theta_star,
                "rep": rep,
                "join_fraction": result.join_fraction,
                "join_fraction_valid": getattr(result, "join_fraction_valid", float("nan")),
                "n_join": result.n_join,
                "n_agents": result.n_agents,
                "n_valid": getattr(result, "n_valid", 0),
                "n_api_error": getattr(result, "n_api_error", 0),
                "n_unparseable": getattr(result, "n_unparseable", 0),
                "api_error_rate": getattr(result, "api_error_rate", 0.0),
                "unparseable_rate": getattr(result, "unparseable_rate", 0.0),
                "coup_success": result.coup_success,
                "theoretical_attack": result.theoretical_attack,
                "benefit": args.benefit,
                "cost": args.cost,
                "model": args.model,
                "agents": result.agents,  # per-agent raw responses
            }

        # Build all coroutines
        coros = []
        for design_name, config, signal_mode in run_specs:
            for ti, theta in enumerate(theta_grid):
                for rep in range(args.reps):
                    coros.append(_run_one_cell(
                        design_name, config, signal_mode, float(theta), ti, rep,
                    ))

        try:
            tasks = [asyncio.ensure_future(c) for c in coros]
            results = []
            pbar = tqdm(total=len(tasks), desc="Infodesign", unit="cell",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            for fut in asyncio.as_completed(tasks):
                result = await fut
                results.append(result)
                pbar.set_postfix_str(
                    f"{result['design']} θ-θ*={result['theta_relative']:+.2f} "
                    f"join={result['join_fraction']:.2f}"
                )
                pbar.update(1)
            pbar.close()
            return results
        finally:
            await client.close()

    print(f"Running information design experiment (fully parallel)...")
    print(f"  Model: {args.model}, Agents: {args.n_agents}, Sigma: {args.sigma}")
    print(f"  Benefit: {args.benefit}, Cost: {args.cost}")

    results = asyncio.run(_run())

    # ── Save results ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir = resolve_model_output_dir(output_dir, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate agents (raw responses) from summary rows before building DataFrame
    raw_logs = results  # full dicts including 'agents' key
    summary_rows = [{k: v for k, v in r.items() if k != "agents"} for r in results]
    df = pd.DataFrame(summary_rows)

    # ── Append mode: merge with existing data ───────────────────────
    if getattr(args, "append", False):
        combined_path = output_dir / "experiment_infodesign_all_summary.csv"
        if combined_path.exists():
            old_df = pd.read_csv(combined_path)
            df = pd.concat([old_df, df], ignore_index=True)
            print(f"  Appended {len(summary_rows)} new rows to {len(old_df)} existing → {len(df)} total")

        log_path = output_dir / "experiment_infodesign_all_log.json"
        if log_path.exists():
            with open(log_path) as f:
                old_logs = json.load(f)
            raw_logs = old_logs + raw_logs

    # Save per-design CSVs
    for design_name in df["design"].unique():
        design_df = df[df["design"] == design_name]
        path = output_dir / f"experiment_infodesign_{design_name}_summary.csv"
        design_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    # Save combined CSV
    combined_path = output_dir / "experiment_infodesign_all_summary.csv"
    df.to_csv(combined_path, index=False)
    print(f"  Combined: {combined_path}")

    # Save full logs with per-agent raw responses
    log_path = output_dir / "experiment_infodesign_all_log.json"
    with open(log_path, "w") as f:
        json.dump(raw_logs, f, indent=2, default=str)
    print(f"  Raw logs: {log_path}")

    # ── Quick stats ──────────────────────────────────────────────────
    print(f"\nResults summary:")
    for design_name in df["design"].unique():
        ddf = df[df["design"] == design_name]
        jcol = join_fraction_column(ddf)
        print(f"  {design_name:>12s}: mean_join={ddf[jcol].mean():.3f} "
              f"std={ddf[jcol].std():.3f} n={len(ddf)}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Paper 2: Information Design in LLM Global Games"
    )
    add_common_args(parser)
    # θ grid
    parser.add_argument("--theta-points", type=int, default=9,
                        help="Number of θ grid points")
    parser.add_argument("--theta-range", type=float, default=0.30,
                        help="Half-width of θ grid around θ*")
    # Design selection
    parser.add_argument("--designs", nargs="+",
                        default=["baseline", "stability", "instability",
                                 "public_signal",
                                 "stability_clarity", "stability_direction",
                                 "stability_dissent",
                                 "censor_upper", "censor_lower",
                                 "scramble", "flip"],
                        help="Designs to run")
    parser.add_argument("--reps", type=int, default=30,
                        help="Repetitions per (design, θ) cell")
    parser.add_argument("--bandwidth", type=float, default=None,
                        help="Override proximity bandwidth for all designs (default: use each design's own)")
    parser.add_argument("--append", action="store_true",
                        help="Append new results to existing CSVs/logs instead of overwriting")
    parser.add_argument("--treatment", choices=["pure", "comm"], default="pure",
                        help="Game treatment: pure (default) or comm (communication round)")
    parser.add_argument("--n-propaganda", type=int, default=0,
                        help="Number of propaganda (regime plant) agents in comm games")
    parser.add_argument("--surveillance", action="store_true",
                        help="Tell agents their messages are monitored by regime security")
    parser.add_argument("--surveillance-mode", choices=["full", "placebo", "anonymous"], default="full",
                        help="Type of surveillance applied (if --surveillance is active).")
    parser.add_argument("--z-center", type=float, default=None,
                        help="z-score centering point (default: θ*, use 0.0 for legacy behavior)")
    parser.add_argument("--group-size-info", action="store_true",
                        help="Tell agents how many citizens are in the group")
    parser.add_argument("--elicit-beliefs", action="store_true",
                        help="After each decision, ask agents for P(uprising succeeds) on 0-100 scale")

    args = parser.parse_args()

    # Auto-create model-specific output subfolder
    args.output_dir = str(resolve_model_output_dir(args.output_dir, args.model))

    run_infodesign(args)


if __name__ == "__main__":
    main()
