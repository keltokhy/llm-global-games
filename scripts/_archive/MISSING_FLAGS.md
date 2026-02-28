# Missing CLI Flags in Simulation Code

This document records CLI flags that are missing from `run_infodesign.py`
but may be needed for future referee-response experiments.

## `--elicit-second-order` in `run_infodesign.py`

**Status:** Missing from `run_infodesign.py` (exists in `run.py`)

**What it does:** After each agent's JOIN/STAY decision, asks a follow-up
question: "What percentage of citizens do you think will choose to JOIN?"
This elicits second-order beliefs (uncertainty about others' actions).

**Where it exists:**
- `run.py` (line 569): `parser.add_argument("--elicit-second-order", ...)`
- `experiment.py`: Both `run_pure_global_game()` and
  `run_communication_game()` accept `elicit_second_order=` parameter.

**What's missing in `run_infodesign.py`:**
1. No `--elicit-second-order` CLI argument (line ~406, near `--elicit-beliefs`)
2. The `game_kwargs` dict (line ~248) does not pass `elicit_second_order`
   to the game functions.

**Fix required:** Add to `run_infodesign.py`:
```python
# In main() argument parser (after --elicit-beliefs):
parser.add_argument("--elicit-second-order", action="store_true",
    help="After each decision, ask agents what %% of citizens will JOIN (0-100 scale)")

# In run_infodesign(), inside game_kwargs dict:
elicit_second_order=getattr(args, 'elicit_second_order', False),
```

**Impact on current scripts:** The `run_referee_response.sh` script does
NOT use `--elicit-second-order` with `run_infodesign`. Section A uses
`run.py` directly (which already supports the flag), so this gap does not
block any planned experiments. However, if future work wants to measure
second-order beliefs within information-design treatments (e.g., "do
agents under censorship have different beliefs about others?"), this flag
would need to be added.
