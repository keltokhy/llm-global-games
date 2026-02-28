#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_remaining_reviewer.sh — Remaining reviewer response experiments
#
# Phase 1: DONE (all 6 cross-generator cells complete)
# Phase 2A uncalibrated: 3/4 done — MiniMax missing
# Phase 2B placebo: not started
# Phase 3 temperature: not started
# Phase 4 surv cross-model: not started (skip Trinity/free)
# Phase 5 punishment risk: not started
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  Remaining Reviewer Experiments — $(date)"
echo "══════════════════════════════════════════════════════"


# ── Phase 2A: MiniMax uncalibrated (only missing model) ──────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  Phase 2A: Uncalibrated — MiniMax M2-Her"
echo "──────────────────────────────────────────────────────"
uv run python -m agent_based_simulation.run pure \
  --model "minimax/minimax-m2-her" \
  --n-countries 5 --n-periods 20 --n-agents 25 \
  --output-dir "output/uncalibrated-robustness/minimax--minimax-m2-her" \
  $MC


# ── Phase 2B: Placebo calibration ────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  Phase 2B: Placebo calibration (2 models × 2 shifts)"
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


# ── Phase 3: Temperature sweep ───────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  Phase 3: Temperature robustness (2 models × 5 temps)"
echo "──────────────────────────────────────────────────────"

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


# ── Phase 4: Surveillance cross-model (skip Trinity/free) ────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  Phase 4: Surveillance cross-model (2 models)"
echo "  Skipping Trinity (free tier)"
echo "──────────────────────────────────────────────────────"

for MODEL in \
  "qwen/qwen3-235b-a22b-2507" \
  "openai/gpt-oss-120b"; do

  SLUG="${MODEL//\/\/--}"

  echo ""
  echo ">>> Surveillance comm: $MODEL"
  uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" --load-calibrated \
    --surveillance \
    --n-countries 5 --n-periods 40 --n-agents 25 \
    --output-dir "output/surveillance/${SLUG}" \
    $MC

  echo ""
  echo ">>> Censorship designs: $MODEL"
  uv run python -m agent_based_simulation.run_infodesign \
    --model "$MODEL" --load-calibrated \
    --designs censor_lower censor_upper censor_both \
    --reps 30 --append $MC
done


# ── Phase 5: Punishment risk elicitation ─────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  Phase 5: Punishment risk (2 models × 3 conditions)"
echo "──────────────────────────────────────────────────────"

for MODEL in \
  "mistralai/mistral-small-creative" \
  "meta-llama/llama-3.3-70b-instruct"; do

  SLUG="${MODEL//\/\/--}"
  OUTDIR="output/punishment-risk/${SLUG}"

  echo ""
  echo ">>> Pure + punishment risk: $MODEL"
  uv run python -m agent_based_simulation.run pure \
    --model "$MODEL" --load-calibrated \
    --elicit-punishment-risk \
    --n-countries 5 --n-periods 20 --n-agents 25 \
    --output-dir "$OUTDIR" \
    $MC

  echo ""
  echo ">>> Comm + punishment risk: $MODEL"
  uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" --load-calibrated \
    --elicit-punishment-risk \
    --n-countries 5 --n-periods 20 --n-agents 25 \
    --output-dir "$OUTDIR" \
    $MC

  echo ""
  echo ">>> Surveillance + punishment risk: $MODEL"
  uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" --load-calibrated \
    --surveillance \
    --elicit-punishment-risk \
    --n-countries 5 --n-periods 20 --n-agents 25 \
    --output-dir "${OUTDIR}_surv" \
    $MC
done


echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL REMAINING EXPERIMENTS DONE — $(date)"
echo "══════════════════════════════════════════════════════"
