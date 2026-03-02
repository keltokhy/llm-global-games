#!/bin/bash
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"
MODEL="meta-llama/llama-3.1-8b" # Very fast, available base model to check logic

# pure baseline
uv run python -m agent_based_simulation.run comm --model $MODEL --n-periods 10
# surveillance 
uv run python -m agent_based_simulation.run comm --model $MODEL --surveillance --n-periods 10
