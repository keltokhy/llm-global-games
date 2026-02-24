#!/usr/bin/env bash
set -e  # stop on first error

MODEL="mistralai/mistral-small-creative"
MC="--max-concurrent 200"
BASE="--load-calibrated"

echo "============================================"
echo "  REMAINING EXPERIMENTS — FULL PIPELINE"
echo "============================================"

# ─────────────────────────────────────────────────
# 1. MISSING INFODESIGN DESIGNS (instability + public_signal)
#    Appends to existing Mistral-Small infodesign data
# ─────────────────────────────────────────────────
echo ""
echo ">>> 1/10: Instability + Public Signal designs"
uv run python -m agent_based_simulation.run_infodesign \
  --model $MODEL $BASE \
  --designs instability public_signal \
  --reps 30 $MC --append

# ─────────────────────────────────────────────────
# 2. BANDWIDTH ROBUSTNESS (narrow=0.05, wide=0.30)
# ─────────────────────────────────────────────────
echo ""
echo ">>> 2/10: Bandwidth robustness (0.05)"
uv run python -m agent_based_simulation.run_infodesign \
  --model $MODEL $BASE \
  --designs baseline stability censor_upper censor_lower \
  --bandwidth 0.05 --reps 20 $MC \
  --output-dir agent_based_simulation/output/bandwidth-005

echo ""
echo ">>> 3/10: Bandwidth robustness (0.30)"
uv run python -m agent_based_simulation.run_infodesign \
  --model $MODEL $BASE \
  --designs baseline stability censor_upper censor_lower \
  --bandwidth 0.30 --reps 20 $MC \
  --output-dir agent_based_simulation/output/bandwidth-030

# ─────────────────────────────────────────────────
# 4. FULL INFODESIGN ON 2ND MODEL (GPT-OSS-120B)
# ─────────────────────────────────────────────────
echo ""
echo ">>> 4/10: Full infodesign on GPT-OSS-120B"
uv run python -m agent_based_simulation.run_infodesign \
  --model openai/gpt-oss-120b --load-calibrated \
  --designs baseline stability censor_upper censor_lower scramble flip \
  --reps 30 $MC

# ─────────────────────────────────────────────────
# 5. MORE SURVEILLANCE DATA (75 more periods → 200 total)
# ─────────────────────────────────────────────────
echo ""
echo ">>> 5/10: More surveillance periods"
uv run python -m agent_based_simulation.run comm \
  --model $MODEL $BASE \
  --n-agents 25 --n-periods 75 --surveillance $MC \
  --output-dir agent_based_simulation/output/surveillance --append

# ─────────────────────────────────────────────────
# 6. SURVEILLANCE × CENSORSHIP INTERACTION
#    Does censorship + surveillance compound or is censorship redundant?
# ─────────────────────────────────────────────────
echo ""
echo ">>> 6/10: Surveillance × Censorship"
uv run python -m agent_based_simulation.run_infodesign \
  --model $MODEL $BASE \
  --treatment comm --surveillance \
  --designs baseline censor_upper censor_lower \
  --reps 30 $MC \
  --output-dir agent_based_simulation/output/surveillance-x-censorship

# ─────────────────────────────────────────────────
# 7. PROPAGANDA × SURVEILLANCE COMBO
#    Plants + chilling effect together
# ─────────────────────────────────────────────────
echo ""
echo ">>> 7/10: Propaganda + Surveillance"
uv run python -m agent_based_simulation.run comm \
  --model $MODEL $BASE \
  --n-agents 25 --n-periods 200 \
  --n-propaganda 5 --surveillance $MC \
  --output-dir agent_based_simulation/output/propaganda-surveillance

# ─────────────────────────────────────────────────
# 8. NETWORK DENSITY (8, 16, 24 neighbors)
# ─────────────────────────────────────────────────
echo ""
echo ">>> 8a/10: Network density k=8"
uv run python -m agent_based_simulation.run comm \
  --model $MODEL $BASE \
  --n-agents 25 --n-periods 200 --n-neighbors 8 $MC \
  --output-dir agent_based_simulation/output/network-k8

echo ""
echo ">>> 8b/10: Network density k=16"
uv run python -m agent_based_simulation.run comm \
  --model $MODEL $BASE \
  --n-agents 25 --n-periods 200 --n-neighbors 16 $MC \
  --output-dir agent_based_simulation/output/network-k16

echo ""
echo ">>> 8c/10: Full broadcast (k=24)"
uv run python -m agent_based_simulation.run comm \
  --model $MODEL $BASE \
  --n-agents 25 --n-periods 200 --n-neighbors 24 $MC \
  --output-dir agent_based_simulation/output/network-k24

# ─────────────────────────────────────────────────
# 9. PERSONA TREATMENTS
#    Same model, different roles — pure game
# ─────────────────────────────────────────────────
echo ""
echo ">>> 9a/10: Persona — pure game"
uv run python -m agent_based_simulation.run pure \
  --model $MODEL $BASE \
  --n-agents 25 --n-periods 200 \
  --personas "military officer" "university student" "business owner" "factory worker" "government clerk" \
  $MC \
  --output-dir agent_based_simulation/output/persona-pure

echo ""
echo ">>> 9b/10: Persona — comm game"
uv run python -m agent_based_simulation.run comm \
  --model $MODEL $BASE \
  --n-agents 25 --n-periods 200 \
  --personas "military officer" "university student" "business owner" "factory worker" "government clerk" \
  $MC \
  --output-dir agent_based_simulation/output/persona-comm

# ─────────────────────────────────────────────────
# 10. MIXED-MODEL × INFODESIGN
#     Do some models resist censorship better?
# ─────────────────────────────────────────────────
echo ""
echo ">>> 10/10: Mixed-model infodesign (placeholder — needs mixed-model infodesign support)"
echo "  SKIPPED: run_infodesign.py doesn't support --mixed-models yet"

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
