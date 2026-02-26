#!/usr/bin/env bash
# Overnight run — complete referee response experiments
#
# What's already done:
#   - Llama 70B surv×censorship (Section B)
#   - Mistral surv×censorship (already existed)
#   - All 7 models have stability (but Qwen3 235B only 25 rows)
#
# What this script runs:
#   B.  Surv × censorship: Qwen3 235B + GPT-OSS 120B
#   C1. Qwen3 235B full infodesign re-run (existing data only 25 rows/design)
#   C2. GPT-OSS 120B: instability + public_signal (missing designs)
#   D.  Uncalibrated robustness: Mistral + Llama + Qwen3 235B
#   A.  Re-run belief elicitation with fixed prompts (Mistral)
#
# Estimate: ~150k LLM calls, 3-5 hours
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

echo "=== OVERNIGHT RUN START: $(date) ==="

# ── B: Surveillance × censorship (2 remaining top models) ──────

echo ""
echo ">>> B.1: Surv × censorship — Qwen3 235B"
uv run python -m agent_based_simulation.run_infodesign \
  --model qwen/qwen3-235b-a22b-2507 --load-calibrated \
  --treatment comm --surveillance \
  --designs baseline censor_upper censor_lower \
  --reps 30 $MC \
  --output-dir output/surveillance-x-censorship

echo ""
echo ">>> B.2: Surv × censorship — GPT-OSS 120B"
uv run python -m agent_based_simulation.run_infodesign \
  --model openai/gpt-oss-120b --load-calibrated \
  --treatment comm --surveillance \
  --designs baseline censor_upper censor_lower \
  --reps 30 $MC \
  --output-dir output/surveillance-x-censorship

# ── C1: Qwen3 235B full infodesign (existing data is 25 rows/design — unusable)
#    Re-run all standard designs at 30 reps to match other models.
#    Using --append so the 25-row data stays (harmless, just gets pooled).

echo ""
echo ">>> C1: Full infodesign re-run — Qwen3 235B (baseline+stability+censor+falsification)"
uv run python -m agent_based_simulation.run_infodesign \
  --model qwen/qwen3-235b-a22b-2507 --load-calibrated \
  --designs baseline stability censor_upper censor_lower scramble flip \
  --reps 30 $MC --append

echo ""
echo ">>> C1b: Instability + public_signal — Qwen3 235B"
uv run python -m agent_based_simulation.run_infodesign \
  --model qwen/qwen3-235b-a22b-2507 --load-calibrated \
  --designs instability public_signal \
  --reps 30 $MC --append

# ── C2: GPT-OSS 120B missing designs ───────────────────────────

echo ""
echo ">>> C2a: Instability — GPT-OSS 120B"
uv run python -m agent_based_simulation.run_infodesign \
  --model openai/gpt-oss-120b --load-calibrated \
  --designs instability \
  --reps 30 $MC --append

echo ""
echo ">>> C2b: Public signal — GPT-OSS 120B"
uv run python -m agent_based_simulation.run_infodesign \
  --model openai/gpt-oss-120b --load-calibrated \
  --designs public_signal \
  --reps 30 $MC --append

# ── D: Uncalibrated robustness ─────────────────────────────────

echo ""
echo ">>> D.1: Uncalibrated pure — Mistral"
uv run python -m agent_based_simulation.run pure \
  --model mistralai/mistral-small-creative \
  --n-agents 25 --n-countries 5 --n-periods 20 \
  --output-dir output/uncalibrated-robustness $MC

echo ""
echo ">>> D.2: Uncalibrated pure — Llama 70B"
uv run python -m agent_based_simulation.run pure \
  --model meta-llama/llama-3.3-70b-instruct \
  --n-agents 25 --n-countries 5 --n-periods 20 \
  --output-dir output/uncalibrated-robustness $MC

echo ""
echo ">>> D.3: Uncalibrated pure — Qwen3 235B"
uv run python -m agent_based_simulation.run pure \
  --model qwen/qwen3-235b-a22b-2507 \
  --n-agents 25 --n-countries 5 --n-periods 20 \
  --output-dir output/uncalibrated-robustness $MC

# ── A: Re-run belief elicitation with fixed prompts ────────────

echo ""
echo ">>> A.1: Pure + beliefs (fixed prompts)"
uv run python -m agent_based_simulation.run pure \
  --model mistralai/mistral-small-creative --load-calibrated \
  --n-agents 25 --n-countries 10 --n-periods 20 \
  --elicit-beliefs --elicit-second-order \
  --append $MC

echo ""
echo ">>> A.2: Comm + beliefs (fixed prompts)"
uv run python -m agent_based_simulation.run comm \
  --model mistralai/mistral-small-creative --load-calibrated \
  --n-agents 25 --n-countries 10 --n-periods 20 \
  --elicit-beliefs --elicit-second-order \
  --append $MC

echo ""
echo ">>> A.3: Surveillance + beliefs (fixed prompts)"
uv run python -m agent_based_simulation.run comm \
  --model mistralai/mistral-small-creative --load-calibrated \
  --n-agents 25 --n-countries 10 --n-periods 20 \
  --surveillance --elicit-beliefs --elicit-second-order \
  --append $MC

echo ""
echo "=== OVERNIGHT RUN COMPLETE: $(date) ==="
echo ""
echo "New/updated outputs:"
echo "  B (surv×cens):   output/surveillance-x-censorship/{qwen,openai}*"
echo "  C1 (Qwen full):  output/qwen--qwen3-235b-a22b-2507/"
echo "  C2 (GPT-OSS):    output/openai--gpt-oss-120b/"
echo "  D (uncalib):     output/uncalibrated-robustness/"
echo "  A (beliefs):     output/mistralai--mistral-small-creative/"
