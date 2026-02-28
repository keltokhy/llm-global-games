#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_referee_response.sh
#
# Runs ALL remaining experiments for the referee report response:
#
#   E1: Common-knowledge vs private 2×2            (Concern #1)
#   E2: Fixed-messages surveillance test            (Concern #6)
#   E4: Hard scramble (primary + cross-model)       (Concern #2)
#
# Then regenerates the full analysis pipeline:
#   - verify_paper_stats.py  → verified_stats.json
#   - render_paper_tables.py → paper/tables/*.tex
#   - make_figures.py        → paper/figures/*.pdf
#   - pdflatex × 2           → paper/paper.pdf
#
# Run from repo root:
#   bash scripts/run_referee_response.sh
#
# Requires: OPENROUTER_API_KEY environment variable.
#
# ── Compute estimates ──────────────────────────────────────────────
#
# E1 (CK 2×2):
#   4 designs × 9 θ × 30 reps × 25 agents = 27,000 LLM calls
#
# E4a (Hard scramble, primary):
#   1 design × 9 θ × 30 reps × 25 agents = 6,750 calls
#
# E4b (Hard scramble, Llama):
#   1 design × 9 θ × 30 reps × 25 agents = 6,750 calls
#
# E2 (Fixed-messages surveillance):
#   5 countries × 40 periods × 25 agents × 2 rounds = 10,000 calls
#
# Grand total: ~50,500 LLM calls
# At ~$0.001/call: ~$51
# Expected runtime: 30-60 min with --max-concurrent 200
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="mistralai/mistral-small-creative"
MODEL2="meta-llama/llama-3.3-70b-instruct"
SLUG="mistralai--mistral-small-creative"
OUT="output"
MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  REFEREE RESPONSE: Wave 2 Experiments + Wave 4"
echo "  Primary: $MODEL"
echo "  Cross:   $MODEL2"
echo "══════════════════════════════════════════════════════"


# ══════════════════════════════════════════════════════════════════════
# E1: Common-Knowledge vs Private 2×2 (Concern #1)
#
# 2×2 crossing {common-knowledge, private} × {high-coord, low-coord}.
# Tests whether CK framing amplifies the coordination slope,
# confirming higher-order-belief processing beyond sentiment.
# NOTE: baseline omitted — 270 rows already exist. Compare new
# designs against existing baseline via verify_paper_stats.py.
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  E1: Common-Knowledge vs Private 2×2 (Concern #1)"
echo "  4 designs × 9θ × 30 reps × 25 agents = 27,000 calls"
echo "──────────────────────────────────────────────────────"
uv run python -m agent_based_simulation.run_infodesign \
  --model "$MODEL" --load-calibrated \
  --designs ck_high_coord ck_low_coord priv_high_coord priv_low_coord \
  --reps 30 --append $MC


# ══════════════════════════════════════════════════════════════════════
# E4: Hard Scramble (Concern #2)
#
# All briefings generated from fixed θ=0.5, breaking any possible
# θ→briefing correlation. Stronger than cross-θ permutation.
# Run for primary model + cross-model (Llama) to address weak
# scramble collapse in some architectures.
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  E4a: Hard Scramble — primary model (Concern #2)"
echo "  1 design × 9θ × 30 reps × 25 agents = 6,750 calls"
echo "──────────────────────────────────────────────────────"
uv run python -m agent_based_simulation.run_infodesign \
  --model "$MODEL" --load-calibrated \
  --designs hard_scramble \
  --reps 30 --append $MC

echo ""
echo "──────────────────────────────────────────────────────"
echo "  E4b: Hard Scramble — Llama 70B (Concern #2)"
echo "  1 design × 9θ × 30 reps × 25 agents = 6,750 calls"
echo "──────────────────────────────────────────────────────"
uv run python -m agent_based_simulation.run_infodesign \
  --model "$MODEL2" --load-calibrated \
  --designs hard_scramble \
  --reps 30 --append $MC


# ══════════════════════════════════════════════════════════════════════
# E2: Fixed-Messages Surveillance Test (Concern #6)
#
# Replays baseline communication messages under surveillance.
# The --fixed-messages flag loads pre-recorded messages from the
# baseline comm log, so agents see identical peer messages but
# the surveillance warning is active in the messaging prompt.
#
# If join rates drop → mechanism beyond message degradation.
# If unchanged → chilling effect is entirely via message content.
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  E2: Fixed-Messages Surveillance Test (Concern #6)"
echo "  5 countries × 40 periods × 25 agents = 10,000 calls"
echo "──────────────────────────────────────────────────────"
uv run python -m agent_based_simulation.run comm \
  --model "$MODEL" \
  --load-calibrated \
  --surveillance \
  --fixed-messages "${OUT}/${SLUG}/experiment_comm_log.json" \
  --n-countries 5 --n-periods 40 --n-agents 25 \
  --output-dir "${OUT}/fixed-messages-surv" \
  $MC


# ══════════════════════════════════════════════════════════════════════
# WAVE 4: Regenerate analysis pipeline
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Wave 4: Regenerate stats → tables → figures → PDF"
echo "══════════════════════════════════════════════════════"

echo ""
echo "── verify_paper_stats.py ──"
uv run python analysis/verify_paper_stats.py

echo ""
echo "── render_paper_tables.py ──"
uv run python analysis/render_paper_tables.py

echo ""
echo "── make_figures.py ──"
uv run python analysis/make_figures.py

echo ""
echo "── Compile paper (2 passes for cross-references) ──"
cd paper
pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode paper.tex 2>&1 \
  | grep -e "Output written" -e "Error" -e "undefined" || true
cd ..


# ══════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL DONE"
echo ""
echo "  E1 (CK 2×2):"
echo "    ${OUT}/${SLUG}/experiment_infodesign_ck_*_summary.csv"
echo "    ${OUT}/${SLUG}/experiment_infodesign_priv_*_summary.csv"
echo ""
echo "  E4 (Hard Scramble):"
echo "    ${OUT}/${SLUG}/experiment_infodesign_hard_scramble_summary.csv"
echo "    ${OUT}/meta-llama--llama-3.3-70b-instruct/experiment_infodesign_hard_scramble_summary.csv"
echo ""
echo "  E2 (Fixed-Messages Surveillance):"
echo "    ${OUT}/fixed-messages-surv/${SLUG}/experiment_comm_summary.csv"
echo ""
echo "  Pipeline:"
echo "    analysis/verified_stats.json"
echo "    paper/tables/ (17 tables)"
echo "    paper/figures/ (23 figures)"
echo "    paper/paper.pdf"
echo "══════════════════════════════════════════════════════"
