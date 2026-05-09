#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Pre-decision belief × messages-EXCLUDED  +  Mistral cross-task placebo
#
# Fills the only missing cell in the wedge factorial:
#   pre-decision × messages-excluded × {comm, surveillance}
# and replicates the Llama cross-task discriminant placebo on Mistral
# so the informational/coordination decomposition exists in two models.
#
# All runs use the same seed (7777) as run_beliefs_with_messages.sh so
# country-period draws line up with the post-decision factorial cells.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="mistralai/mistral-small-creative"
N_COUNTRIES=10
N_PERIODS=50          # 10 × 50 = 500 country-periods per condition
SEED=7777
MAX_CONC=200
CAL_DIR="output/mistralai--mistral-small-creative"

# ── Cell A: pre-decision × messages-excluded × comm baseline ─────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/4] pre+nomsg comm baseline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/revision-beliefs-pre-nomsg" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --elicit-beliefs \
    --elicit-second-order \
    --belief-order pre \
    --second-order-order pre

# ── Cell B: pre-decision × messages-excluded × surveillance ──────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/4] pre+nomsg surveillance"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/revision-beliefs-pre-nomsg-surveillance" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --surveillance \
    --elicit-beliefs \
    --elicit-second-order \
    --belief-order pre \
    --second-order-order pre

# ── Cell C: Mistral cross-task placebo, comm baseline ────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/4] Mistral cross-task individual_bet, comm baseline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/cross-task-placebo-baseline" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --task-mode individual_bet

# ── Cell D: Mistral cross-task placebo, surveillance ─────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/4] Mistral cross-task individual_bet, surveillance"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "output/cross-task-placebo-surveillance" \
    --n-countries "$N_COUNTRIES" \
    --n-periods "$N_PERIODS" \
    --seed "$SEED" \
    --max-concurrent "$MAX_CONC" \
    --surveillance \
    --task-mode individual_bet

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All four runs complete."
echo "  output/revision-beliefs-pre-nomsg/                  (comm,  pre,  no msgs)"
echo "  output/revision-beliefs-pre-nomsg-surveillance/     (surv,  pre,  no msgs)"
echo "  output/cross-task-placebo-baseline/<mistral>/       (comm,  individual_bet)"
echo "  output/cross-task-placebo-surveillance/<mistral>/   (surv,  individual_bet)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
