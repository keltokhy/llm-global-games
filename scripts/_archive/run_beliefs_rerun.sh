#!/usr/bin/env bash
# Re-run experiments to recover belief data (decisions were fine, beliefs failed from internet dropout)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="mistralai/mistral-small-creative"

echo "=== 1/3: Belief-before-action (pure, --belief-order both) ==="
uv run python -m agent_based_simulation.run pure \
  --model "$MODEL" \
  --load-calibrated \
  --n-periods 200 --n-countries 1 \
  --max-concurrent 200 \
  --elicit-beliefs --belief-order both \
  --output-dir output/mistralai--mistral-small-creative/_beliefs_pre_v2

echo ""
echo "=== 2/3: Surveillance placebo (monitored, no consequences) ==="
uv run python -m agent_based_simulation.run comm \
  --model "$MODEL" \
  --load-calibrated \
  --n-periods 200 --n-countries 1 \
  --max-concurrent 200 \
  --elicit-beliefs --surveillance --surveillance-mode placebo \
  --output-dir output/mistralai--mistral-small-creative/_surveillance_placebo_v2

echo ""
echo "=== 3/3: Surveillance anonymous (aggregated, no identity) ==="
uv run python -m agent_based_simulation.run comm \
  --model "$MODEL" \
  --load-calibrated \
  --n-periods 200 --n-countries 1 \
  --max-concurrent 200 \
  --elicit-beliefs --surveillance --surveillance-mode anonymous \
  --output-dir output/mistralai--mistral-small-creative/_surveillance_anonymous_v2

echo ""
echo "=== All belief re-runs complete ==="
