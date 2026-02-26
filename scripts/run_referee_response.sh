#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_referee_response.sh
#
# All new experiments for the referee response, organized by item:
#   Section A: Second-order belief elicitation           (Item 9)
#   Section B: Cross-model surveillance x censorship     (Item 7)
#   Section C: Cross-model stability/instability         (Item 7)
#   Section D: Uncalibrated robustness check             (Item 6)
#
# Run from repo root:
#   bash scripts/run_referee_response.sh
#
# Requires: OPENROUTER_API_KEY environment variable.
#
# ── Compute estimates ────────────────────────────────────────────────
#
# Section A: Second-order beliefs (Mistral Small Creative)
#   3 treatments × 200 country-periods × 25 agents = 15,000 decision calls
#   + 15,000 belief-elicitation follow-up calls = 30,000 total
#   comm/surveillance also have message rounds: +10,000 calls
#   Total Section A: ~40,000 LLM calls
#
# Section B: Cross-model surveillance × censorship (Llama 70B + Qwen 30B)
#   Per model: 3 designs × 9 θ-points × 30 reps × 25 agents = 20,250 calls
#     (comm treatment doubles for message round: 40,500)
#   2 models × 40,500 = 81,000 calls
#   Total Section B: ~81,000 LLM calls
#
# Section C: Cross-model stability/instability replication
#   GPT-OSS-120B instability: 1 design × 9 θ × 30 reps × 25 agents = 6,750
#   Qwen3-235B instability:   1 design × 9 θ × 30 reps × 25 agents = 6,750
#   Total Section C: ~13,500 LLM calls
#
# Section D: Uncalibrated robustness
#   3 models × 100 country-periods × 25 agents = 7,500 calls
#   Total Section D: ~7,500 LLM calls
#
# ── Grand total ──────────────────────────────────────────────────────
#   ~142,000 LLM calls
#   At ~$0.001/call (small models): ~$142
#   At ~$0.003/call (70B models):   ~$280 for Sections B-C
#   Expected runtime: 2-4 hours with --max-concurrent 200
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

MC="--max-concurrent 200"

echo "══════════════════════════════════════════════════════"
echo "  REFEREE RESPONSE EXPERIMENTS"
echo "══════════════════════════════════════════════════════"


# ══════════════════════════════════════════════════════════════════════
# SECTION A: Second-order belief elicitation (Referee Item 9)
#
# This elicits "what percentage of citizens will choose to JOIN?" after
# each decision, providing direct measurement of second-order beliefs
# (uncertainty about others' actions). Second-order beliefs are the
# core strategic variable in global games: players act on what they
# think others will do. Combined with first-order beliefs (P(success)),
# this lets us test whether LLM agents' beliefs are internally
# consistent and whether surveillance shifts beliefs vs. behavior.
#
# Design: 10 countries × 20 periods = 200 country-periods per
# treatment, N=25 agents, with --elicit-beliefs and
# --elicit-second-order flags.
# ══════════════════════════════════════════════════════════════════════

MODEL_A="mistralai/mistral-small-creative"
N_AGENTS=25
N_COUNTRIES=10
N_PERIODS=20

echo ""
echo "──────────────────────────────────────────────────────"
echo "  SECTION A: Second-order belief elicitation (Item 9)"
echo "  Model: $MODEL_A"
echo "  Design: $N_COUNTRIES countries × $N_PERIODS periods = $((N_COUNTRIES * N_PERIODS)) country-periods/treatment"
echo "──────────────────────────────────────────────────────"

# A.1: Pure game with first- and second-order beliefs
echo ""
echo ">>> A.1: Pure game + belief elicitation"
uv run python -m agent_based_simulation.run pure \
  --model "$MODEL_A" --load-calibrated \
  --n-agents $N_AGENTS --n-countries $N_COUNTRIES --n-periods $N_PERIODS \
  --elicit-beliefs --elicit-second-order \
  --append $MC

# A.2: Communication game with first- and second-order beliefs
echo ""
echo ">>> A.2: Communication game + belief elicitation"
uv run python -m agent_based_simulation.run comm \
  --model "$MODEL_A" --load-calibrated \
  --n-agents $N_AGENTS --n-countries $N_COUNTRIES --n-periods $N_PERIODS \
  --elicit-beliefs --elicit-second-order \
  --append $MC

# A.3: Surveillance game with first- and second-order beliefs
# Surveillance should shift second-order beliefs downward (agents expect
# others to be more cautious under monitoring), providing a mechanism
# test for the chilling effect.
echo ""
echo ">>> A.3: Surveillance game + belief elicitation"
uv run python -m agent_based_simulation.run comm \
  --model "$MODEL_A" --load-calibrated \
  --n-agents $N_AGENTS --n-countries $N_COUNTRIES --n-periods $N_PERIODS \
  --surveillance --elicit-beliefs --elicit-second-order \
  --append $MC


# ══════════════════════════════════════════════════════════════════════
# SECTION B: Cross-model surveillance × censorship (Referee Item 7)
#
# Tests whether the surveillance × information-design interaction
# replicates across model architectures. The key question: does
# censorship compound with surveillance, or is surveillance alone
# sufficient to suppress collective action?
#
# For each model, we run the infodesign grid with baseline +
# censor_upper + censor_lower under the communication treatment with
# surveillance enabled. This produces a 3-design × 9-θ × 30-rep grid.
#
# Existing data: Mistral Small Creative already has this
# (output/surveillance-x-censorship/). We replicate on Llama 70B
# and Qwen 30B.
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  SECTION B: Cross-model surveillance × censorship (Item 7)"
echo "──────────────────────────────────────────────────────"

# B.1: Llama 3.3 70B — surveillance × censorship
echo ""
echo ">>> B.1: Surveillance × censorship — Llama 3.3 70B"
uv run python -m agent_based_simulation.run_infodesign \
  --model meta-llama/llama-3.3-70b-instruct --load-calibrated \
  --treatment comm --surveillance \
  --designs baseline censor_upper censor_lower \
  --reps 30 $MC \
  --output-dir output/surveillance-x-censorship

# B.2: Qwen3 30B — surveillance × censorship
echo ""
echo ">>> B.2: Surveillance × censorship — Qwen3 30B"
uv run python -m agent_based_simulation.run_infodesign \
  --model qwen/qwen3-30b-a3b-instruct-2507 --load-calibrated \
  --treatment comm --surveillance \
  --designs baseline censor_upper censor_lower \
  --reps 30 $MC \
  --output-dir output/surveillance-x-censorship


# ══════════════════════════════════════════════════════════════════════
# SECTION C: Cross-model stability/instability replication (Referee Item 7)
#
# Fills gaps in the information design replication table. Most models
# already have stability + instability data. Missing:
#   - openai/gpt-oss-120b:              instability only
#   - qwen/qwen3-235b-a22b-2507:        instability only
#
# Already complete (no action needed):
#   - mistralai/mistral-small-creative:  stability + instability
#   - meta-llama/llama-3.3-70b-instruct: stability + instability
#   - qwen/qwen3-30b-a3b-instruct-2507: stability + instability
#   - mistralai/ministral-3b-2512:       stability + instability
#   - allenai/olmo-3-7b-instruct:        stability + instability
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  SECTION C: Cross-model stability/instability (Item 7)"
echo "──────────────────────────────────────────────────────"

# C.1: GPT-OSS 120B — instability design (stability already exists)
echo ""
echo ">>> C.1: Instability design — GPT-OSS 120B"
uv run python -m agent_based_simulation.run_infodesign \
  --model openai/gpt-oss-120b --load-calibrated \
  --designs instability \
  --reps 30 $MC --append

# C.2: Qwen3 235B — instability design (stability already exists)
echo ""
echo ">>> C.2: Instability design — Qwen3 235B"
uv run python -m agent_based_simulation.run_infodesign \
  --model qwen/qwen3-235b-a22b-2507 --load-calibrated \
  --designs instability \
  --reps 30 $MC --append


# ══════════════════════════════════════════════════════════════════════
# SECTION D: Uncalibrated robustness check (supports Referee Item 6)
#
# Runs the pure game WITHOUT --load-calibrated (using default
# cutoff_center=0) to test whether the headline sigmoid result
# survives without per-model calibration. If the S-curve persists
# with default parameters, calibration is a refinement rather than
# a precondition for the main finding.
#
# Design: 5 countries × 20 periods = 100 country-periods per model,
# N=25 agents. Output goes to a separate directory so it cannot
# contaminate calibrated results.
# ══════════════════════════════════════════════════════════════════════

echo ""
echo "──────────────────────────────────────────────────────"
echo "  SECTION D: Uncalibrated robustness check (Item 6)"
echo "──────────────────────────────────────────────────────"

# D.1: Mistral Small Creative — uncalibrated pure game
echo ""
echo ">>> D.1: Uncalibrated pure — Mistral Small Creative"
uv run python -m agent_based_simulation.run pure \
  --model mistralai/mistral-small-creative \
  --n-agents 25 --n-countries 5 --n-periods 20 \
  --output-dir output/uncalibrated-robustness \
  $MC

# D.2: Llama 3.3 70B — uncalibrated pure game
echo ""
echo ">>> D.2: Uncalibrated pure — Llama 3.3 70B"
uv run python -m agent_based_simulation.run pure \
  --model meta-llama/llama-3.3-70b-instruct \
  --n-agents 25 --n-countries 5 --n-periods 20 \
  --output-dir output/uncalibrated-robustness \
  $MC

# D.3: Qwen3 30B — uncalibrated pure game
echo ""
echo ">>> D.3: Uncalibrated pure — Qwen3 30B"
uv run python -m agent_based_simulation.run pure \
  --model qwen/qwen3-30b-a3b-instruct-2507 \
  --n-agents 25 --n-countries 5 --n-periods 20 \
  --output-dir output/uncalibrated-robustness \
  $MC


# ══════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL REFEREE RESPONSE EXPERIMENTS COMPLETE"
echo ""
echo "  New outputs:"
echo "    Section A (beliefs):   output/mistralai--mistral-small-creative/"
echo "    Section B (surv×cens): output/surveillance-x-censorship/"
echo "    Section C (instab):    output/openai--gpt-oss-120b/"
echo "                           output/qwen--qwen3-235b-a22b-2507/"
echo "    Section D (uncalib):   output/uncalibrated-robustness/"
echo "══════════════════════════════════════════════════════"
