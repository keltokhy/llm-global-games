#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_placebo.sh — Phase 2: Uncalibrated + Placebo Calibration
#
# 2A: Uncalibrated runs for 4 models missing uncalibrated data.
# 2B: Placebo calibration — deliberately wrong cutoff_center (±0.3).
#
# ── Compute estimates ──────────────────────────────────────────────
#   2A: 4 models × 5 countries × 20 periods × 25 agents = 10,000 calls
#   2B: 2 models × 2 shifts × 5 × 20 × 25 = 10,000 calls
#   Total: ~20,000 LLM calls  (~$20)
#   Expected runtime: 15-30 min with --max-concurrent 200
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  Phase 2: Uncalibrated + Placebo Calibration"
echo "══════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════
# 2A: Uncalibrated pure games for 4 missing models
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  2A: Uncalibrated runs (4 models × 2,500 calls each)"
echo "──────────────────────────────────────────────────────"

for MODEL in \
  "qwen/qwen3-30b-a3b-instruct-2507" \
  "openai/gpt-oss-120b" \
  "arcee-ai/trinity-large-preview:free" \
  "minimax/minimax-m2-her"; do

  SLUG="${MODEL//\/\/--}"
  echo ""
  echo ">>> Uncalibrated: $MODEL"
  uv run python -m agent_based_simulation.run pure \
    --model "$MODEL" \
    --n-countries 5 --n-periods 20 --n-agents 25 \
    --output-dir "output/uncalibrated-robustness/${SLUG}" \
    $MC
done

# ══════════════════════════════════════════════════════════════════════
# 2B: Placebo calibration — wrong center ±0.3
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  2B: Placebo calibration (2 models × 2 shifts)"
echo "──────────────────────────────────────────────────────"

for MODEL in \
  "mistralai/mistral-small-creative" \
  "meta-llama/llama-3.3-70b-instruct"; do

  SLUG="${MODEL//\/\/--}"
  for SHIFT in 0.3 -0.3; do
    LABEL="${SHIFT//./p}"
    LABEL="${LABEL//-/neg}"
    echo ""
    echo ">>> Placebo: $MODEL  shift=$SHIFT"
    uv run python -m agent_based_simulation.run pure \
      --model "$MODEL" --load-calibrated \
      --wrong-center "$SHIFT" \
      --n-countries 5 --n-periods 20 --n-agents 25 \
      --output-dir "output/placebo-calibration/${SLUG}_shift_${LABEL}" \
      $MC
  done
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Phase 2 COMPLETE"
echo "  Output: output/uncalibrated-robustness/<slug>/"
echo "          output/placebo-calibration/<slug>_shift_*/"
echo "══════════════════════════════════════════════════════"
