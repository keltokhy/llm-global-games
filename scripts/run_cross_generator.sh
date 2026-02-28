#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_cross_generator.sh — Phase 1: Cross-Generator Replication
#
# Tests whether the sigmoid emerges from LLM reasoning rather than
# the specific text rendering of the briefing generator.
#
# Runs calibration sweep + pure game with 3 language variants
# (baseline, cable, journalistic) on 2 core models.
#
# ── Compute estimates ──────────────────────────────────────────────
#   Per model, per variant:
#     calibration_sweep: 200 z × 3 reps = 600 calls
#     pure game: 20 θ × 5 reps × 25 agents = 2,500 calls
#   Per model: 3 × 3,100 = 9,300 calls
#   Total: 2 × 9,300 ≈ 18,600 LLM calls
#   At ~$0.001/call: ~$19
#   Expected runtime: 15-30 min with --max-concurrent 200
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL1="mistralai/mistral-small-creative"
MODEL2="meta-llama/llama-3.3-70b-instruct"
MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  Phase 1: Cross-Generator Replication"
echo "  Models: Mistral Small Creative, Llama 3.3 70B"
echo "  Variants: baseline, cable, journalistic"
echo "══════════════════════════════════════════════════════"

for MODEL in "$MODEL1" "$MODEL2"; do
  SLUG="${MODEL//\/\/--}"
  for VARIANT in baseline cable journalistic; do
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Model: $MODEL  Variant: $VARIANT"
    echo "──────────────────────────────────────────────────────"

    OUTDIR="output/cross-generator/${SLUG}_${VARIANT}"

    # Step 1: Calibration sweep
    echo "  → Calibration sweep (600 calls)"
    uv run python -m agent_based_simulation.run calibrate \
      --model "$MODEL" --load-calibrated \
      --language-variant "$VARIANT" \
      --output-dir "$OUTDIR" \
      $MC

    # Step 2: Pure game
    echo "  → Pure game (2,500 calls)"
    uv run python -m agent_based_simulation.run pure \
      --model "$MODEL" --load-calibrated \
      --language-variant "$VARIANT" \
      --n-countries 5 --n-periods 20 --n-agents 25 \
      --output-dir "$OUTDIR" \
      $MC
  done
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Phase 1 COMPLETE"
echo "  Output: output/cross-generator/<slug>_<variant>/"
echo "══════════════════════════════════════════════════════"
