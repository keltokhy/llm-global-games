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
# Estimate: ~160k LLM calls, 3-5 hours
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

STEPS=(
  "B.1|Surv × censorship — Qwen3 235B|uv run python -m agent_based_simulation.run_infodesign --model qwen/qwen3-235b-a22b-2507 --load-calibrated --treatment comm --surveillance --designs baseline censor_upper censor_lower --reps 30 $MC --output-dir output/surveillance-x-censorship"
  "B.2|Surv × censorship — GPT-OSS 120B|uv run python -m agent_based_simulation.run_infodesign --model openai/gpt-oss-120b --load-calibrated --treatment comm --surveillance --designs baseline censor_upper censor_lower --reps 30 $MC --output-dir output/surveillance-x-censorship"
  "C1a|Full infodesign re-run — Qwen3 235B|uv run python -m agent_based_simulation.run_infodesign --model qwen/qwen3-235b-a22b-2507 --load-calibrated --designs baseline stability censor_upper censor_lower scramble flip --reps 30 $MC --append"
  "C1b|Instability + public_signal — Qwen3 235B|uv run python -m agent_based_simulation.run_infodesign --model qwen/qwen3-235b-a22b-2507 --load-calibrated --designs instability public_signal --reps 30 $MC --append"
  "C2a|Instability — GPT-OSS 120B|uv run python -m agent_based_simulation.run_infodesign --model openai/gpt-oss-120b --load-calibrated --designs instability --reps 30 $MC --append"
  "C2b|Public signal — GPT-OSS 120B|uv run python -m agent_based_simulation.run_infodesign --model openai/gpt-oss-120b --load-calibrated --designs public_signal --reps 30 $MC --append"
  "D.1|Uncalibrated pure — Mistral|uv run python -m agent_based_simulation.run pure --model mistralai/mistral-small-creative --n-agents 25 --n-countries 5 --n-periods 20 --output-dir output/uncalibrated-robustness $MC"
  "D.2|Uncalibrated pure — Llama 70B|uv run python -m agent_based_simulation.run pure --model meta-llama/llama-3.3-70b-instruct --n-agents 25 --n-countries 5 --n-periods 20 --output-dir output/uncalibrated-robustness $MC"
  "D.3|Uncalibrated pure — Qwen3 235B|uv run python -m agent_based_simulation.run pure --model qwen/qwen3-235b-a22b-2507 --n-agents 25 --n-countries 5 --n-periods 20 --output-dir output/uncalibrated-robustness $MC"
  "A.1|Pure + beliefs (fixed prompts)|uv run python -m agent_based_simulation.run pure --model mistralai/mistral-small-creative --load-calibrated --n-agents 25 --n-countries 10 --n-periods 20 --elicit-beliefs --elicit-second-order --append $MC"
  "A.2|Comm + beliefs (fixed prompts)|uv run python -m agent_based_simulation.run comm --model mistralai/mistral-small-creative --load-calibrated --n-agents 25 --n-countries 10 --n-periods 20 --elicit-beliefs --elicit-second-order --append $MC"
  "A.3|Surveillance + beliefs (fixed prompts)|uv run python -m agent_based_simulation.run comm --model mistralai/mistral-small-creative --load-calibrated --n-agents 25 --n-countries 10 --n-periods 20 --surveillance --elicit-beliefs --elicit-second-order --append $MC"
)

TOTAL=${#STEPS[@]}

echo ""
echo "══════════════════════════════════════════════════════"
echo "  OVERNIGHT RUN: $TOTAL steps"
echo "  Started: $(date)"
echo "══════════════════════════════════════════════════════"
echo ""

for i in "${!STEPS[@]}"; do
  IFS='|' read -r label desc cmd <<< "${STEPS[$i]}"
  step=$((i + 1))
  echo "[$step/$TOTAL] $label: $desc"
  start_time=$SECONDS
  eval "$cmd"
  elapsed=$(( SECONDS - start_time ))
  mins=$(( elapsed / 60 ))
  secs=$(( elapsed % 60 ))
  echo "[$step/$TOTAL] ✓ done in ${mins}m${secs}s"
  echo ""
done

echo "══════════════════════════════════════════════════════"
echo "  ALL $TOTAL STEPS COMPLETE: $(date)"
echo ""
echo "  New/updated outputs:"
echo "    B (surv×cens):   output/surveillance-x-censorship/{qwen,openai}*"
echo "    C1 (Qwen full):  output/qwen--qwen3-235b-a22b-2507/"
echo "    C2 (GPT-OSS):    output/openai--gpt-oss-120b/"
echo "    D (uncalib):     output/uncalibrated-robustness/"
echo "    A (beliefs):     output/mistralai--mistral-small-creative/"
echo "══════════════════════════════════════════════════════"
