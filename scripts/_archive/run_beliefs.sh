#!/usr/bin/env bash
# Belief elicitation experiments for cross-model replication
# Each run: ~5 min, 200 periods × 25 agents = 5,000 observations
# Requires: OPENROUTER_API_KEY
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_MISTRAL="mistralai/mistral-small-creative"
MODEL_LLAMA="meta-llama/llama-3.3-70b-instruct"

echo "=== 1/4: Llama 70B pure beliefs ==="
uv run python -m agent_based_simulation.run pure \
  --model "$MODEL_LLAMA" \
  --load-calibrated \
  --n-periods 200 --n-countries 1 \
  --max-concurrent 200 \
  --elicit-beliefs \
  --output-dir output/meta-llama--llama-3.3-70b-instruct/_beliefs

echo "=== 2/4: Llama 70B comm beliefs ==="
uv run python -m agent_based_simulation.run comm \
  --model "$MODEL_LLAMA" \
  --load-calibrated \
  --n-periods 200 --n-countries 1 \
  --max-concurrent 200 \
  --elicit-beliefs \
  --output-dir output/meta-llama--llama-3.3-70b-instruct/_beliefs

echo "=== 3/4: Llama 70B surveillance beliefs ==="
uv run python -m agent_based_simulation.run comm \
  --model "$MODEL_LLAMA" \
  --load-calibrated \
  --n-periods 200 --n-countries 1 \
  --max-concurrent 200 \
  --elicit-beliefs --surveillance \
  --output-dir output/meta-llama--llama-3.3-70b-instruct/_beliefs_surveillance

echo "=== 4/4: Mistral propaganda k=5 beliefs ==="
uv run python -m agent_based_simulation.run comm \
  --model "$MODEL_MISTRAL" \
  --load-calibrated \
  --n-periods 200 --n-countries 1 \
  --max-concurrent 200 \
  --elicit-beliefs --n-propaganda 5 \
  --output-dir output/mistralai--mistral-small-creative/_beliefs_propaganda_k5

echo "=== All belief experiments complete ==="
