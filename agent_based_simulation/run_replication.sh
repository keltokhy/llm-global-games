#!/usr/bin/env bash
set -e

# Replication runs for Paper 1 & Paper 2
# Closes the main data gaps identified in the review.
#
# Estimated: ~6 experiment runs, each 100-200 periods × 25 agents.
# Cost depends on model pricing.

MC="--max-concurrent 200"

echo "============================================"
echo "  REPLICATION RUNS"
echo "============================================"

# ─────────────────────────────────────────────────
# 1. SURVEILLANCE on Llama-3.3-70B  [Paper 2, HIGH priority]
#    Replicates the 17.5pp chilling effect on a second architecture.
# ─────────────────────────────────────────────────
echo ""
echo ">>> 1/6: Surveillance — Llama 3.3 70B"
uv run python -m agent_based_simulation.run comm \
  --model meta-llama/llama-3.3-70b-instruct --load-calibrated \
  --n-agents 25 --n-periods 200 --surveillance $MC \
  --output-dir agent_based_simulation/output/surveillance

# ─────────────────────────────────────────────────
# 2. SURVEILLANCE on Qwen3-30B  [Paper 2, HIGH priority]
#    Third architecture for surveillance.
# ─────────────────────────────────────────────────
echo ""
echo ">>> 2/6: Surveillance — Qwen3 30B"
uv run python -m agent_based_simulation.run comm \
  --model qwen/qwen3-30b-a3b-instruct-2507 --load-calibrated \
  --n-agents 25 --n-periods 200 --surveillance $MC \
  --output-dir agent_based_simulation/output/surveillance

# ─────────────────────────────────────────────────
# 3. PROPAGANDA k=5 on Llama-3.3-70B  [Paper 2, HIGH priority]
#    Replicates the mechanical-vs-behavioral decomposition.
# ─────────────────────────────────────────────────
echo ""
echo ">>> 3/6: Propaganda k=5 — Llama 3.3 70B"
uv run python -m agent_based_simulation.run comm \
  --model meta-llama/llama-3.3-70b-instruct --load-calibrated \
  --n-agents 25 --n-periods 200 --n-propaganda 5 $MC \
  --output-dir agent_based_simulation/output/propaganda-k5

# ─────────────────────────────────────────────────
# 4. SCRAMBLE + FLIP for GPT-OSS-120B  [Paper 1, MEDIUM priority]
#    Completes the falsification table for a 200-obs model.
# ─────────────────────────────────────────────────
echo ""
echo ">>> 4/6: Scramble — GPT-OSS 120B"
uv run python -m agent_based_simulation.run scramble \
  --model openai/gpt-oss-120b --load-calibrated \
  --n-agents 25 --n-periods 100 $MC

echo ""
echo ">>> 5/6: Flip — GPT-OSS 120B"
uv run python -m agent_based_simulation.run flip \
  --model openai/gpt-oss-120b --load-calibrated \
  --n-agents 25 --n-periods 100 $MC

# ─────────────────────────────────────────────────
# 6. CENSORSHIP on Llama-3.3-70B  [Paper 2, MEDIUM priority]
#    Second model with censorship designs.
# ─────────────────────────────────────────────────
echo ""
echo ">>> 6/6: Censorship infodesign — Llama 3.3 70B"
uv run python -m agent_based_simulation.run_infodesign \
  --model meta-llama/llama-3.3-70b-instruct --load-calibrated \
  --designs censor_upper censor_lower \
  --reps 30 $MC --append

echo ""
echo "============================================"
echo "  ALL REPLICATION RUNS COMPLETE"
echo "============================================"
