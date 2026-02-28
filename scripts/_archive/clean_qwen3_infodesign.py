"""Remove 25 stale rows (treatment=NaN) from Qwen3 235B infodesign CSVs.

These rows come from an older run with a coarser theta grid.
Each file should go from 295 rows to 270 rows after cleanup.
"""

import pandas as pd
from pathlib import Path

BASE = Path("output/qwen--qwen3-235b-a22b-2507")

FILES = [
    "experiment_infodesign_baseline_summary.csv",
    "experiment_infodesign_censor_upper_summary.csv",
    "experiment_infodesign_scramble_summary.csv",
    "experiment_infodesign_stability_summary.csv",
]

for fname in FILES:
    path = BASE / fname
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["treatment"])
    after = len(df)
    assert after == 270, f"{fname}: expected 270 rows after cleanup, got {after}"
    df.to_csv(path, index=False)
    print(f"{fname}: {before} -> {after} rows (dropped {before - after})")

print("\nDone. All 4 files cleaned.")
