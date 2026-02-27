#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_referee_response_v2.sh
#
# New experiments for the second referee response:
#   Section A: B/C comparative statics (cost/benefit narrative)
#   Section B: Censorship with common knowledge
#   Section C: Temperature robustness
#
# Run from repo root:
#   bash scripts/run_referee_response_v2.sh
#
# Requires: OPENROUTER_API_KEY environment variable.
#
# ── Compute estimates ────────────────────────────────────────────────
#
# Section A: 3 designs × 9 θ-points × 30 reps × 25 agents = 20,250 calls
# Section B: 3 designs × 9 θ-points × 30 reps × 25 agents = 20,250 calls
# Section C: 3 temps × 5 countries × 20 periods × 25 agents = 7,500 calls
#
# Grand total: ~48,000 LLM calls
# Expected cost: ~$50 at ~$0.001/call (small models)
# Expected runtime: 1-2 hours with --max-concurrent 200
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="mistralai/mistral-small-creative"
MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  REFEREE RESPONSE V2 EXPERIMENTS"
echo "══════════════════════════════════════════════════════"


# ══════════════════════════════════════════════════════════════════════
# SECTION A: B/C Comparative Statics
#
# Theory predicts: higher perceived cost → higher cutoff (less joining).
# We inject cost/benefit context into the briefing header and check
# whether the estimated logistic cutoff shifts in the predicted direction.
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  SECTION A: B/C Comparative Statics"
echo "  Model: $MODEL"
echo "──────────────────────────────────────────────────────"

uv run python -m agent_based_simulation.run_infodesign \
  --model "$MODEL" --load-calibrated \
  --designs baseline bc_high_cost bc_low_cost \
  --reps 30 $MC


# ══════════════════════════════════════════════════════════════════════
# SECTION B: Censorship with Common Knowledge
#
# Tests whether making censorship common knowledge changes the pooling
# effect. Theory (Kolotilin et al. 2022) assumes receivers understand
# the censorship rule. Adding a header about censorship lets agents
# update beliefs about the information structure.
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  SECTION B: Censorship Common Knowledge"
echo "  Model: $MODEL"
echo "──────────────────────────────────────────────────────"

uv run python -m agent_based_simulation.run_infodesign \
  --model "$MODEL" --load-calibrated \
  --designs baseline censor_upper censor_upper_known \
  --reps 30 $MC


# ══════════════════════════════════════════════════════════════════════
# SECTION C: Temperature Robustness
#
# Shows that the main qualitative result (monotone signal response)
# holds across decoding temperatures. T=0.3 (more deterministic),
# T=0.7 (baseline), T=1.0 (higher entropy).
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  SECTION C: Temperature Robustness"
echo "  Model: $MODEL"
echo "──────────────────────────────────────────────────────"

for T in 0.3 0.7 1.0; do
  echo ""
  echo ">>> Temperature = $T"
  uv run python -m agent_based_simulation.run pure \
    --model "$MODEL" --load-calibrated \
    --temperature "$T" \
    --n-countries 5 --n-periods 20 \
    --output-dir "output/temperature-robustness-T${T}" \
    $MC
done


# ══════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL V2 EXPERIMENTS COMPLETE"
echo ""
echo "  Next steps:"
echo "    1. uv run python analysis/verify_paper_stats.py"
echo "    2. uv run python analysis/render_paper_tables.py"
echo "    3. Review output and integrate into paper"
echo "══════════════════════════════════════════════════════"
