#!/usr/bin/env bash
# Experiment A: replay the willingness x style 2x2 variant logs to Llama
# receivers on the nested grid, decision prompt with NO surveillance reference.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export GGC_LLM_CACHE_DIR="output/.llm_cache_expA"

M=meta-llama/llama-3.3-70b-instruct
COMMON="--model $M --load-calibrated --fixed-messages-mode exact --fixed-messages-align-metadata --decision-context none --n-countries 10 --n-periods 50 --seed 4242 --max-concurrent 160"

for cell in w1-direct w1-coded w0-direct w0-coded; do
  key="${cell//-/_}"
  echo "=== replay $cell ==="
  uv run python -m agent_based_simulation.run comm $COMMON \
    --fixed-messages output/expA-$cell/${key}_log.json \
    --output-dir output/expA-$cell-replay
done
echo "=== ALL A REPLAYS DONE ==="
