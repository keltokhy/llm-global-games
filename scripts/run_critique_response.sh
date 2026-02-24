#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# run_critique_response.sh
#
# Generates all new evidence for the three critiques:
#   1. Calibration circularity  →  text baseline diagnostics (offline)
#   2. Identification problem   →  text baseline + holdout validation
#   3. Missing strategic interaction  →  --group-size-info experiments
#
# Run from the repo root:
#   bash agent_based_simulation/run_critique_response.sh
#
# Step 1 is offline (no API calls). Steps 2-4 require OPENROUTER_API_KEY.
# Existing results are NOT overwritten — new data goes to separate dirs.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="mistralai/mistral-small-creative"
SLUG="mistralai--mistral-small-creative"
OUT="output"
CAL_DIR="${OUT}/${SLUG}"

echo "========================================="
echo "  Critique Response Runs"
echo "  Model: $MODEL"
echo "========================================="

# ─── Step 1: Text baseline on existing data (NO API calls) ───────────────
echo ""
echo "── Step 1/4: Text baseline diagnostics (offline) ──"
uv run python -c "
import pandas as pd
from agent_based_simulation.calibration import (
    analyze_calibration, text_baseline_diagnostics, plot_calibration,
)

df = pd.read_csv('${CAL_DIR}/autocalibrate_final_raw.csv')
print(f'Loaded {len(df)} rows from existing calibration data')

summary, diag = analyze_calibration(df, theoretical_benefit=1.0)

print()
print('Diagnostics (loss no longer penalizes slope):')
print(f'  fitted_center:      {diag[\"fitted_center\"]:.3f}')
print(f'  fitted_slope:       {diag[\"fitted_slope\"]:.3f}')
print(f'  mean_join_rate:     {diag[\"mean_join_rate\"]:.3f}')
print(f'  text_baseline_corr: {diag[\"text_baseline_corr\"]:.3f}')
print(f'  z_direction_corr:   {diag[\"z_direction_corr\"]:.3f}')

plot_calibration(summary, diag, theoretical_benefit=1.0,
    output_path='${CAL_DIR}/calibration_with_text_baseline.png')
"
echo ""

# ─── Step 2: Holdout calibration validation (API calls) ─────────────────
echo "── Step 2/4: Holdout calibration (30% holdout) ──"
uv run python -m agent_based_simulation.run autocalibrate \
    --model "$MODEL" \
    --output-dir "${OUT}/holdout-validation" \
    --holdout-fraction 0.3 \
    --n-reps 8 \
    --z-steps 21 \
    --max-rounds 3
echo ""

# ─── Step 3: Pure game with group-size info (API calls) ─────────────────
echo "── Step 3/4: Pure game WITH group-size info ──"
uv run python -m agent_based_simulation.run pure \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "${OUT}/group-size-info" \
    --group-size-info \
    --n-countries 5 \
    --n-periods 20 \
    --n-agents 25
echo ""

# ─── Step 4: Comm game with group-size info (API calls) ─────────────────
echo "── Step 4/4: Communication game WITH group-size info ──"
uv run python -m agent_based_simulation.run comm \
    --model "$MODEL" \
    --load-calibrated \
    --calibration-dir "$CAL_DIR" \
    --output-dir "${OUT}/group-size-info" \
    --group-size-info \
    --n-countries 5 \
    --n-periods 20 \
    --n-agents 25
echo ""

# ─── Done ────────────────────────────────────────────────────────────────
echo "========================================="
echo "  All runs complete. New outputs:"
echo ""
echo "  ${CAL_DIR}/"
echo "    calibration_with_text_baseline.png"
echo ""
echo "  ${OUT}/holdout-validation/${SLUG}/"
echo "    autocalibrate_history.csv  (has holdout_fitted_rmse column)"
echo ""
echo "  ${OUT}/group-size-info/${SLUG}/"
echo "    experiment_pure_summary.csv"
echo "    experiment_comm_summary.csv"
echo ""
echo "  Original data untouched in ${CAL_DIR}/"
echo "========================================="
