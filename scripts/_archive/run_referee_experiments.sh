#!/usr/bin/env bash
# Run referee-response experiments: second-order beliefs + fixed-message surveillance
# Requires: OPENROUTER_API_KEY environment variable
set -euo pipefail

MODEL="mistralai/mistral-small-creative"
N_AGENTS=25
N_COUNTRIES=10
N_PERIODS=20
# 10 countries × 20 periods = 200 country-periods per treatment

echo "=== Referee Response Experiments ==="
echo "Model: $MODEL"
echo "Design: $N_COUNTRIES countries × $N_PERIODS periods = $((N_COUNTRIES * N_PERIODS)) periods per treatment"
echo ""

COMMON_ARGS="--model $MODEL --n-agents $N_AGENTS --n-countries $N_COUNTRIES --n-periods $N_PERIODS --load-calibrated"

# 1. Pure game with first- and second-order belief elicitation
echo "--- [1/4] Pure game + belief elicitation ---"
uv run python -m agent_based_simulation.run pure \
    $COMMON_ARGS \
    --elicit-beliefs --elicit-second-order \
    --append

# 2. Communication game with belief elicitation
echo ""
echo "--- [2/4] Communication game + belief elicitation ---"
uv run python -m agent_based_simulation.run comm \
    $COMMON_ARGS \
    --elicit-beliefs --elicit-second-order \
    --append

# 3. Surveillance game with belief elicitation
echo ""
echo "--- [3/4] Surveillance game + belief elicitation ---"
uv run python -m agent_based_simulation.run comm \
    $COMMON_ARGS \
    --surveillance \
    --elicit-beliefs --elicit-second-order \
    --append

# 4. Fixed-message surveillance test
# Uses messages from step 2 (regular comm) but tells agents they're monitored.
# This isolates perceived-payoff-shift from self-censorship.
SLUG=$(echo "$MODEL" | tr '/' '--')
COMM_LOG="output/$SLUG/experiment_comm_log.json"

if [ -f "$COMM_LOG" ]; then
    echo ""
    echo "--- [4/4] Fixed-message surveillance test ---"
    echo "  Using pre-recorded messages from: $COMM_LOG"
    uv run python -m agent_based_simulation.run comm \
        $COMMON_ARGS \
        --surveillance \
        --fixed-messages "$COMM_LOG" \
        --elicit-beliefs --elicit-second-order \
        --append
else
    echo ""
    echo "--- [4/4] SKIPPED: Fixed-message surveillance test ---"
    echo "  Requires comm log from step 2: $COMM_LOG"
    echo "  Run steps 1-3 first, then re-run this script."
fi

echo ""
echo "=== All referee experiments complete ==="
