#!/usr/bin/env bash
# Within-briefing falsification experiments
# Tests whether bullet ordering or domain-specific content drives correlation.
set -euo pipefail

MODEL="mistralai/mistral-small-creative"

echo "=== Within-briefing falsification experiments ==="
echo "Model: $MODEL"
echo ""

# Run baseline + three within-briefing falsification designs
uv run python -m agent_based_simulation.run_infodesign \
    --model "$MODEL" --load-calibrated \
    --designs baseline within_scramble domain_scramble_coord domain_scramble_state \
    --reps 30 --n-agents 25 --sigma 0.3

echo ""
echo "=== Done ==="
