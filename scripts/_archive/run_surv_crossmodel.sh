#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_surv_crossmodel.sh — Phase 4: Surveillance Cross-Model
#
# Replicate surveillance + censorship experiments on 3 additional
# models (currently only Mistral + Llama have this data).
#
# ── Compute estimates ──────────────────────────────────────────────
#   Per model:
#     surveillance comm: 5 countries × 40 periods × 25 agents × 2 = 10,000
#     censorship (3 designs): 3 × 9 θ × 30 reps × 25 agents = 20,250
#   3 models: 3 × 30,250 ≈ 90,750 LLM calls
#   At ~$0.001/call: ~$91
#   Expected runtime: 60-90 min with --max-concurrent 200
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  Phase 4: Surveillance × Censorship Cross-Model"
echo "  Models: Qwen3 235B, GPT-OSS 120B, Trinity Large"
echo "══════════════════════════════════════════════════════"

for MODEL in \
  "qwen/qwen3-235b-a22b-2507" \
  "openai/gpt-oss-120b" \
  "arcee-ai/trinity-large-preview:free"; do

  SLUG="${MODEL//\/\/--}"

  echo ""
  echo "──────────────────────────────────────────────────────"
  echo "  Model: $MODEL"
  echo "──────────────────────────────────────────────────────"

  # Surveillance communication game
  echo "  → Surveillance comm game (10,000 calls)"
  uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" --load-calibrated \
    --surveillance \
    --n-countries 5 --n-periods 40 --n-agents 25 \
    --output-dir "output/surveillance/${SLUG}" \
    $MC

  # Censorship info-design experiments
  echo "  → Censorship designs (20,250 calls)"
  uv run python -m agent_based_simulation.run_infodesign \
    --model "$MODEL" --load-calibrated \
    --designs censor_lower censor_upper censor_both \
    --reps 30 --append $MC
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Phase 4 COMPLETE"
echo "  Output: output/surveillance/<slug>/"
echo "          output/<slug>/experiment_infodesign_censor_*"
echo "══════════════════════════════════════════════════════"
