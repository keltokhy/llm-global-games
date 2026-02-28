"""
Canonical model roster — single source of truth.

Every analysis script imports from here instead of maintaining its own copy.
"""

# Slugs in paper order (Part I models with full experiment suites)
PART1_SLUGS = [
    "mistralai--mistral-small-creative",
    "meta-llama--llama-3.3-70b-instruct",
    "qwen--qwen3-30b-a3b-instruct-2507",
    "openai--gpt-oss-120b",
    "qwen--qwen3-235b-a22b-2507",
    "arcee-ai--trinity-large-preview_free",
    "minimax--minimax-m2-her",
]

# Display names used in verified_stats.json keys and table rows
DISPLAY_NAMES = {
    "mistralai--mistral-small-creative": "Mistral Small Creative",
    "meta-llama--llama-3.3-70b-instruct": "Llama 3.3 70B",
    "qwen--qwen3-30b-a3b-instruct-2507": "Qwen3 30B",
    "openai--gpt-oss-120b": "GPT-OSS 120B",
    "qwen--qwen3-235b-a22b-2507": "Qwen3 235B",
    "arcee-ai--trinity-large-preview_free": "Trinity Large",
    "minimax--minimax-m2-her": "MiniMax M2-Her",
}

# Short names for figure legends
SHORT_NAMES = {
    "mistralai--mistral-small-creative": "Mistral-Small",
    "meta-llama--llama-3.3-70b-instruct": "Llama-3.3-70B",
    "qwen--qwen3-30b-a3b-instruct-2507": "Qwen3-30B",
    "openai--gpt-oss-120b": "GPT-OSS-120B",
    "qwen--qwen3-235b-a22b-2507": "Qwen3-235B",
    "arcee-ai--trinity-large-preview_free": "Trinity",
    "minimax--minimax-m2-her": "MiniMax-M2",
    "mistralai--ministral-3b-2512": "Ministral-3B",
}

# Per-model colors for figures
MODEL_COLORS = {
    "mistralai--mistral-small-creative": "#2c7bb6",
    "meta-llama--llama-3.3-70b-instruct": "#abdda4",
    "qwen--qwen3-30b-a3b-instruct-2507": "#f46d43",
    "openai--gpt-oss-120b": "#1a9641",
    "qwen--qwen3-235b-a22b-2507": "#5e4fa2",
    "arcee-ai--trinity-large-preview_free": "#fdae61",
    "minimax--minimax-m2-her": "#7b3294",
    "mistralai--ministral-3b-2512": "#e66101",
}

# Compact names for regression tables
REGRESSION_NAMES = {
    "mistralai--mistral-small-creative": "Mistral",
    "meta-llama--llama-3.3-70b-instruct": "Llama 70B",
    "qwen--qwen3-30b-a3b-instruct-2507": "Qwen 30B",
    "openai--gpt-oss-120b": "GPT-OSS 120B",
    "qwen--qwen3-235b-a22b-2507": "Qwen 235B",
    "arcee-ai--trinity-large-preview_free": "Trinity",
    "minimax--minimax-m2-her": "MiniMax",
}

# Models excluded from auto-discovery in figures
EXCLUDE_MODELS = {"allenai--olmo-3-7b-instruct"}

# Display names in paper order (for table renderers)
DISPLAY_ORDER = [DISPLAY_NAMES[s] for s in PART1_SLUGS]
