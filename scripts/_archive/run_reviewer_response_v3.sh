#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_reviewer_response_v3.sh — Master script for all reviewer response experiments
#
# Runs Phases 1-5 sequentially, then regenerates the analysis pipeline.
#
# ── Total compute ─────────────────────────────────────────────────
#   Phase 1 (cross-generator):     ~18,600 calls
#   Phase 2 (uncal + placebo):     ~20,000 calls
#   Phase 3 (temperature):         ~25,000 calls
#   Phase 4 (surv cross-model):    ~90,750 calls
#   Phase 5 (punishment risk):     ~25,000 calls
#   Grand total:                   ~179,350 LLM calls (~$180)
#   Expected runtime: 2-4 hours with --max-concurrent 200
#
# Each phase is independent. If one fails, subsequent phases
# will still run (set +e between phases).
#
# Run from repo root:
#   bash scripts/run_reviewer_response_v3.sh
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."

echo "══════════════════════════════════════════════════════"
echo "  REVIEWER RESPONSE V3: All Experiments"
echo "  Start: $(date)"
echo "══════════════════════════════════════════════════════"

# Each phase runs independently; continue on failure
bash scripts/run_cross_generator.sh || echo "WARNING: Phase 1 (cross-generator) failed"
bash scripts/run_placebo.sh || echo "WARNING: Phase 2 (uncal+placebo) failed"
bash scripts/run_temperature_sweep.sh || echo "WARNING: Phase 3 (temperature) failed"
bash scripts/run_surv_crossmodel.sh || echo "WARNING: Phase 4 (surv cross-model) failed"
bash scripts/run_punishment_risk.sh || echo "WARNING: Phase 5 (punishment risk) failed"

# ══════════════════════════════════════════════════════════════════════
# Regenerate analysis pipeline
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Regenerating analysis pipeline"
echo "══════════════════════════════════════════════════════"

echo "── verify_paper_stats.py ──"
uv run python analysis/verify_paper_stats.py

echo "── render_paper_tables.py ──"
uv run python analysis/render_paper_tables.py

echo "── agent_regressions.py ──"
uv run python analysis/agent_regressions.py

echo "── make_figures.py ──"
uv run python analysis/make_figures.py

echo "── Compile paper ──"
cd paper
pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode paper.tex 2>&1 \
  | command grep -e "Output written" -e "Error" -e "undefined" || true
cd ..

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL DONE — $(date)"
echo "══════════════════════════════════════════════════════"
