#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_remaining_v2.sh — Only what's left after the censor_both crash
#
# Remaining:
#   Phase 4: GPT-OSS surveillance comm (censorship already exists)
#   Phase 5: Punishment risk (2 models × 3 conditions)
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  Remaining experiments — $(date)"
echo "══════════════════════════════════════════════════════"


# ── Phase 4: GPT-OSS surveillance comm ───────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  Phase 4: GPT-OSS surveillance comm game"
echo "──────────────────────────────────────────────────────"
uv run python -m agent_based_simulation.run comm \
  --model "openai/gpt-oss-120b" --load-calibrated \
  --surveillance \
  --n-countries 5 --n-periods 40 --n-agents 25 \
  --output-dir "output/surveillance/openai--gpt-oss-120b" \
  $MC


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
echo "  ALL DONE — $(date)"
echo "══════════════════════════════════════════════════════"
