# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An agent-based simulation framework testing whether LLM agents behave like Bayesian rational players in **global games** (Morris & Shin 1998). Citizens observe noisy private signals about regime strength θ, then decide to JOIN an uprising or STAY. Theory predicts a unique threshold equilibrium — the empirical question is whether LLMs approximate it.

The paper has two main components: (1) establishing that LLM join rates follow the predicted sigmoid as a function of θ, and (2) testing whether a social planner can reshape the information structure (Bayesian persuasion) to shift equilibrium outcomes.

## Running Things

All commands use `uv run` from the `agent_based_simulation/` directory:

```bash
# Core experiments
uv run python -m agent_based_simulation.run pure --model mistralai/mistral-small-creative
uv run python -m agent_based_simulation.run comm --model mistralai/mistral-small-creative
uv run python -m agent_based_simulation.run scramble --model mistralai/mistral-small-creative
uv run python -m agent_based_simulation.run flip --model mistralai/mistral-small-creative
uv run python -m agent_based_simulation.run both --model mistralai/mistral-small-creative

# Information design experiments
uv run python -m agent_based_simulation.run_infodesign --model mistralai/mistral-small-creative --designs stability instability

# Calibration
uv run python -m agent_based_simulation.run autocalibrate --model mistralai/mistral-small-creative
uv run python -m agent_based_simulation.run calibrate --model mistralai/mistral-small-creative

# Figures (reads from output/, writes to figures/)
uv run python agent_based_simulation/make_figures.py
```

Key flags: `--n-agents 25`, `--n-countries 5`, `--n-periods 20`, `--sigma 0.3`, `--benefit 1.0`, `--load-calibrated`, `--append`, `--mixed-models`, `--n-propaganda N`, `--surveillance`, `--personas`, `--group-size-info`, `--holdout-fraction 0.3`.

API key: `OPENROUTER_API_KEY` env var. LLM cache: `GGC_LLM_CACHE_DIR` env var. Local models: `--api-base-url http://localhost:1234/v1`.

## Architecture

### The Signal → Text → Decision Pipeline

```
θ (regime strength) → x_i = θ + N(0,σ) → z-score → BriefingGenerator → natural language briefing → LLM call → JOIN/STAY
```

The core insight: `briefing.py` converts continuous z-scores into graded natural language via three latent sliders (direction, clarity, coordination), 8 evidence domains, and 4 phrase ladders. Many small word-choice variations ("dithering") recover effective signal continuity despite discrete text.

### Module Roles

- **`briefing.py`** — Signal-to-text layer. `BriefingGenerator` converts z-scores into intelligence briefings via slider functions (logistic/Gaussian) that control phrase selection across 8 domains. Most complex module. Tunable parameters: `cutoff_center`, `clarity_width`, `direction_slope`, `coordination_slope`, `dissent_floor`, `mixed_cue_clarity`, plus cutpoint arrays.

- **`experiment.py`** — Game engine. `Agent` dataclass, `PeriodResult` dataclass, async LLM calling (`_call_llm`), decision parsing (`_parse_decision`), and two game modes: `run_pure_global_game()` (simultaneous private decisions) and `run_communication_game()` (message round → decision round on a Watts-Strogatz network). Signal modes: normal, scramble (permute briefings), flip (negate z-scores).

- **`calibration.py`** — Fits LLM behavior to theory. `calibration_sweep()` measures empirical join rates across z-score grid; `auto_calibrate()` iteratively adjusts `cutoff_center` via damped gradient until the fitted logistic center ≈ 0. Loss = fitted_rmse + 0.35·|center| + api_error. Includes `text_baseline_diagnostics()` and holdout validation (`--holdout-fraction`).

- **`infodesign.py`** — Bayesian persuasion layer. `InfoDesignConfig` specifies parameter modifiers; `ThetaAdaptiveBriefingGenerator` applies them via Gaussian proximity weight centered at θ*. Pre-built designs: stability, instability, public_signal, censorship variants, single-channel decompositions.

- **`runtime.py`** — Shared utilities: `model_slug()`, `resolve_model_output_dir()`, `add_common_args()`, game-theory math (`theta_star_baseline`, `attack_mass`), `build_network()`, `FileLLMCache`.

- **`run.py`** — CLI entry point for core experiments (commands: autocalibrate, calibrate, pure, comm, both, scramble, flip).
- **`run_infodesign.py`** — CLI entry point for information design experiments (runs designs over a fixed θ-grid with reps).
- **`make_figures.py`** — Reads CSVs from `output/`, produces `figures/fig01–fig14` + appendix figures.

### Output Convention

Per-model output dirs: `output/<model-slug>/` where slug = `model.replace("/", "--")`.
Files: `experiment_{treatment}_summary.csv`, `experiment_{treatment}_log.json`, `calibrated_index.json`, `calibrated_params_<slug>.json`.

### Key Design Decisions

- **1-DOF calibration**: Only `cutoff_center` matters — sigmoid slope is emergent, never optimized or penalized.
- **`join_fraction_valid`** preferred over raw `join_fraction` (excludes parse errors).
- **Language variants**: `legacy`, `baseline_min`, `baseline`, `baseline_assess`, `baseline_full` — control briefing verbosity/structure.

## Dependencies

Python ≥3.12. Core: `openai`, `numpy`, `pandas`, `scipy`, `matplotlib`, `networkx`, `tqdm`. Install via `uv sync` from repo root.
