#!/usr/bin/env bash
# Experiment B: replay four message logs to Llama receivers on the nested
# 500-cell grid (10x50), decision prompt with NO surveillance reference.
# Cache enabled so reruns resume.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export GGC_LLM_CACHE_DIR="output/.llm_cache_expB"

M=meta-llama/llama-3.3-70b-instruct
COMMON="--model $M --load-calibrated --fixed-messages-mode exact --fixed-messages-align-metadata --decision-context none --n-countries 10 --n-periods 50 --seed 4242 --max-concurrent 160"

echo "=== [1/4] comm-replay (baseline verbatim) ==="
uv run python -m agent_based_simulation.run comm $COMMON \
  --fixed-messages output/revision-nested-comm/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json \
  --output-dir output/expB-comm-replay

echo "=== [2/4] surv-replay (surveilled verbatim) ==="
uv run python -m agent_based_simulation.run comm $COMMON \
  --fixed-messages output/revision-nested-surv/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json \
  --output-dir output/expB-surv-replay

echo "=== [3/4] risk-stripped replay ==="
uv run python -m agent_based_simulation.run comm $COMMON \
  --fixed-messages output/expB-risk-stripped/risk_stripped_log.json \
  --output-dir output/expB-risk-stripped-replay

echo "=== [4/4] risk-only replay ==="
uv run python -m agent_based_simulation.run comm $COMMON \
  --fixed-messages output/expB-risk-only/risk_only_log.json \
  --output-dir output/expB-risk-only-replay

echo "=== ALL B REPLAYS DONE ==="
