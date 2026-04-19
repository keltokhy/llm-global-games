#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Belief elicitation WITH messages — post-decision variant
#
# Runs three conditions (comm baseline, surveillance, degraded messages)
# with post-decision beliefs that INCLUDE peer messages in the prompt.
#
# This complements the existing revision-beliefs-* runs which used
# PRE-decision beliefs with messages.
#
# Output goes to dedicated directories — nothing is overwritten.
# Uses --calibration-dir to load calibrated params from the main output.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="mistralai/mistral-small-creative"
N_COUNTRIES=10
N_PERIODS=50  # 10 × 50 = 500 country-periods per condition
SEED=7777     # different seed from original runs (5150) to avoid cache collisions
MAX_CONC=200

# Base calibration dir (where calibrated_index.json lives)
CAL_DIR="output/mistralai--mistral-small-creative"

# ── Condition 1: Communication baseline ──────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Comm baseline — post-decision beliefs WITH messages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/revision-beliefs-post-live" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --elicit-beliefs \
    --elicit-second-order \
    --beliefs-include-messages \
    --belief-order post \
    --second-order-order post

# ── Condition 2: Surveillance ────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Surveillance — post-decision beliefs WITH messages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/revision-beliefs-post-surveillance" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --surveillance \
    --elicit-beliefs \
    --elicit-second-order \
    --beliefs-include-messages \
    --belief-order post \
    --second-order-order post

# ── Condition 3: Degraded messages ───────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Degraded messages — post-decision beliefs WITH messages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/revision-beliefs-post-degraded" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --degrade-messages \
    --elicit-beliefs \
    --elicit-second-order \
    --beliefs-include-messages \
    --belief-order post \
    --second-order-order post

# ── Condition 4: Comm baseline WITHOUT messages in beliefs ───────────
# (timing-matched control: post-decision beliefs, same seed, no --beliefs-include-messages)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Comm baseline — post-decision beliefs WITHOUT messages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/revision-beliefs-post-nomsg" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --elicit-beliefs \
    --elicit-second-order \
    --belief-order post \
    --second-order-order post

# ── Condition 5: Surveillance WITHOUT messages in beliefs ────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Surveillance — post-decision beliefs WITHOUT messages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/revision-beliefs-post-nomsg-surveillance" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --surveillance \
    --elicit-beliefs \
    --elicit-second-order \
    --belief-order post \
    --second-order-order post

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All runs complete."
echo ""
echo "Output directories:"
echo "  output/revision-beliefs-post-live/           (comm, post-decision, WITH messages)"
echo "  output/revision-beliefs-post-surveillance/   (surv, post-decision, WITH messages)"
echo "  output/revision-beliefs-post-degraded/       (degraded, post-decision, WITH messages)"
echo "  output/revision-beliefs-post-nomsg/          (comm, post-decision, WITHOUT messages)"
echo "  output/revision-beliefs-post-nomsg-surveillance/ (surv, post-decision, WITHOUT messages)"
echo ""
echo "Combined with existing data, this gives the full 2x2x3 design:"
echo "  Timing:    pre-decision (existing revision-beliefs-*) vs post-decision (new)"
echo "  Messages:  included vs excluded in belief elicitation"
echo "  Condition: comm baseline / surveillance / degraded"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
