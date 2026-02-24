#!/usr/bin/env bash
set -e

# cd to repo root (parent of this script's directory)
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

# ═══════════ BATCH 1: Cheap, High Payoff ═══════════

# 1/8: Provenance treatment — Llama 70B
#   3 source headers × 9θ × 30 reps = 810 cells × 25 agents = ~20,250 calls
uv run python -m agent_based_simulation.run_infodesign \
  --model meta-llama/llama-3.3-70b-instruct --load-calibrated \
  --designs provenance_independent provenance_state provenance_social \
  --reps 30 $MC --append

# 2/8: Decomposition replication — Llama 70B
#   3 channels × 9θ × 30 reps = 810 cells × 25 agents = ~20,250 calls
uv run python -m agent_based_simulation.run_infodesign \
  --model meta-llama/llama-3.3-70b-instruct --load-calibrated \
  --designs stability_clarity stability_direction stability_dissent \
  --reps 30 $MC --append

# ═══════════ BATCH 2: Robustness ═══════════

# 3/8: z-centered infodesign rerun — Llama 70B
#   baseline + stability + instability × 9θ × 30 reps = ~20,250 calls
#   (--z-center defaults to theta_star, correcting the z=0.0 bias)
uv run python -m agent_based_simulation.run_infodesign \
  --model meta-llama/llama-3.3-70b-instruct --load-calibrated \
  --designs baseline stability instability \
  --reps 30 $MC \
  --output-dir output/z-centered

# 4/8: Even out Mistral stability to 30 reps
#   stability has 10 reps, need 20 more → 180 cells × 25 = ~4,500 calls
uv run python -m agent_based_simulation.run_infodesign \
  --model mistralai/mistral-small-creative --load-calibrated \
  --designs stability \
  --reps 20 $MC --append

# ═══════════ BATCH 3: Top-5 Push ═══════════

# 5/8: Hot/cold rhetoric — Llama 70B
#   2 designs × 9θ × 30 reps = 540 cells × 25 = ~13,500 calls
uv run python -m agent_based_simulation.run_infodesign \
  --model meta-llama/llama-3.3-70b-instruct --load-calibrated \
  --designs rhetoric_hot rhetoric_cold \
  --reps 30 $MC --append

# 6/8: Censorship — Qwen3 30B
uv run python -m agent_based_simulation.run_infodesign \
  --model qwen/qwen3-30b-a3b-instruct-2507 --load-calibrated \
  --designs baseline censor_upper censor_lower \
  --reps 30 $MC --append

# 7/8: Censorship — OLMo 3 7B
uv run python -m agent_based_simulation.run_infodesign \
  --model allenai/olmo-3-7b-instruct --load-calibrated \
  --designs baseline censor_upper censor_lower \
  --reps 30 $MC --append

# 8/8: Censorship — Ministral 3B
uv run python -m agent_based_simulation.run_infodesign \
  --model mistralai/ministral-3b-2512 --load-calibrated \
  --designs baseline censor_upper censor_lower \
  --reps 30 $MC --append
