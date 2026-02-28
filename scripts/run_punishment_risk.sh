#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_punishment_risk.sh — Phase 5: Punishment Risk Elicitation
#
# Runs pure + comm games with punishment-risk elicitation enabled.
# After each JOIN/STAY decision, agents are asked to rate expected
# punishment severity (0-10) if the uprising fails.
#
# ── Compute estimates ──────────────────────────────────────────────
#   Per model:
#     pure:  5 countries × 20 periods × 25 agents × 2 (decision + risk) = 5,000
#     comm:  5 countries × 20 periods × 25 agents × 3 (msg + decision + risk) = 7,500
#   2 models × 12,500 = 25,000 LLM calls
#   At ~$0.001/call: ~$25
#   Expected runtime: 15-30 min with --max-concurrent 200
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  Phase 5: Punishment Risk Elicitation"
echo "  Models: Mistral Small Creative, Llama 3.3 70B"
echo "══════════════════════════════════════════════════════"

for MODEL in \
  "mistralai/mistral-small-creative" \
  "meta-llama/llama-3.3-70b-instruct"; do

  SLUG="${MODEL//\/\/--}"

  echo ""
  echo "──────────────────────────────────────────────────────"
  echo "  Model: $MODEL"
  echo "──────────────────────────────────────────────────────"

  OUTDIR="output/punishment-risk/${SLUG}"

  # Pure game with punishment risk elicitation
  echo "  → Pure game + punishment risk (5,000 calls)"
  uv run python -m agent_based_simulation.run pure \
    --model "$MODEL" --load-calibrated \
    --elicit-punishment-risk \
    --n-countries 5 --n-periods 20 --n-agents 25 \
    --output-dir "$OUTDIR" \
    $MC

  # Communication game with punishment risk elicitation
  echo "  → Comm game + punishment risk (7,500 calls)"
  uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" --load-calibrated \
    --elicit-punishment-risk \
    --n-countries 5 --n-periods 20 --n-agents 25 \
    --output-dir "$OUTDIR" \
    $MC

  # Surveillance + punishment risk (to see if fear rises under surveillance)
  echo "  → Surveillance + punishment risk (7,500 calls)"
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
echo "  Phase 5 COMPLETE"
echo "  Output: output/punishment-risk/<slug>/"
echo "          output/punishment-risk/<slug>_surv/"
echo "══════════════════════════════════════════════════════"
