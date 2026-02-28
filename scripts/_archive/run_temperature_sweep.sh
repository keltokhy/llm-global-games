#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_temperature_sweep.sh — Phase 3: Temperature Robustness
#
# Run temperature sweep for 2 additional models (Llama, Qwen3 235B).
# Mistral already has temperature data.
#
# ── Compute estimates ──────────────────────────────────────────────
#   2 models × 5 temps × 5 countries × 20 periods × 25 agents
#   = 25,000 LLM calls  (~$25)
#   Expected runtime: 20-40 min with --max-concurrent 200
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  Phase 3: Temperature Robustness"
echo "  Models: Llama 3.3 70B, Qwen3 235B"
echo "  Temperatures: 0.3, 0.5, 0.7, 1.0, 1.2"
echo "══════════════════════════════════════════════════════"

for MODEL in \
  "meta-llama/llama-3.3-70b-instruct" \
  "qwen/qwen3-235b-a22b-2507"; do

  SLUG="${MODEL//\/\/--}"

  for T in 0.3 0.5 0.7 1.0 1.2; do
    TLABEL="${T//./}"
    echo ""
    echo ">>> $MODEL  T=$T"
    uv run python -m agent_based_simulation.run pure \
      --model "$MODEL" --load-calibrated \
      --temperature "$T" \
      --n-countries 5 --n-periods 20 --n-agents 25 \
      --output-dir "output/temperature-robustness/${SLUG}_t${TLABEL}" \
      $MC
  done
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Phase 3 COMPLETE"
echo "  Output: output/temperature-robustness/<slug>_t*/"
echo "══════════════════════════════════════════════════════"
