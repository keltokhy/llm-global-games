"""Power calculations for the matched cross-task decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import CLEAN_RUN_ROOT, require_valid_manifest


REQUIRED_ARMS = {
    "coord_baseline": "coord_with_baseline_messages",
    "coord_surveillance": "coord_with_surveillance_messages",
    "bet_baseline": "bet_with_baseline_messages",
    "bet_surveillance": "bet_with_surveillance_messages",
}


def load_periods(output_root: str | Path) -> pd.DataFrame:
    paths = sorted(Path(output_root).glob("**/periods.parquet"))
    if not paths:
        raise RuntimeError(f"Dependency needed: no periods.parquet files under {output_root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def estimate_cross_task_power(
    periods: pd.DataFrame,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> dict[str, float | int | str]:
    observed = set(periods["arm_id"].unique())
    matched = {
        key: sorted(arm for arm in observed if arm.startswith(prefix))
        for key, prefix in REQUIRED_ARMS.items()
    }
    missing = [key for key, arms in matched.items() if not arms]
    if missing:
        raise RuntimeError(f"Dependency needed: missing cross-task pilot arms: {', '.join(missing)}")

    arm_id = {key: arms[0] for key, arms in matched.items()}
    key_cols = ["model", "country", "period", "theta", "z_public", "benefit", "cost", "theta_star"]
    wide = None
    for label, arm in arm_id.items():
        df = periods[periods["arm_id"].eq(arm)][key_cols + ["join_fraction_valid"]].rename(
            columns={"join_fraction_valid": label}
        )
        wide = df if wide is None else wide.merge(df, on=key_cols, how="inner")
    if wide is None or wide.empty:
        raise RuntimeError("Dependency needed: cross-task pilot arms have no exact matched cells")

    interaction = (wide["coord_surveillance"] - wide["coord_baseline"]) - (
        wide["bet_surveillance"] - wide["bet_baseline"]
    )
    effect = float(interaction.mean())
    sd = float(interaction.std(ddof=1))
    n = int(len(interaction))
    if n < 2 or sd == 0:
        raise RuntimeError("Dependency needed: cross-task pilot variance is not estimable")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(target_power)
    required_n = int(np.ceil(((z_alpha + z_power) * sd / abs(effect)) ** 2)) if effect != 0 else None
    return {
        "pilot_matched_cells": n,
        "pilot_interaction_mean": effect,
        "pilot_interaction_sd": sd,
        "alpha": alpha,
        "target_power": target_power,
        "required_matched_cells": required_n,
        "status": "powered" if required_n is not None and required_n <= n else "needs_main_rows",
    }


def build_power_analysis_status(manifest_path: str | Path) -> dict[str, Any]:
    """Write the pre-main power-analysis status without inventing pilot estimates."""

    manifest = require_valid_manifest(manifest_path)
    arms = manifest.arms
    cross_task = [arm for arm in arms if arm["claim"] == "cross_task_decomposition"]
    direct_coded = [arm for arm in arms if arm["claim"] == "direct_coded_mechanism"]
    sender_surveillance = [arm for arm in arms if arm["claim"] == "sender_side_surveillance"]

    return {
        "status": "dependency_needed",
        "dependency_needed": [
            "CLEAN_RUN/output/pilot/**/periods.parquet for cross-task pilot arms",
            "pilot clustered variance and within-cell replay correlation",
            "QC-passing CLEAN_RUN/message_banks/direct_coded_pairs.parquet for mechanism pilots",
        ],
        "manifest": str(Path(manifest_path).resolve()),
        "declared_arm_counts": {
            "cross_task_decomposition": len(cross_task),
            "direct_coded_mechanism": len(direct_coded),
            "sender_side_surveillance": len(sender_surveillance),
        },
        "declared_expected_period_rows": {
            "cross_task_decomposition": int(sum(arm["expected_rows"] for arm in cross_task)),
            "direct_coded_mechanism": int(sum(arm["expected_rows"] for arm in direct_coded)),
            "sender_side_surveillance": int(sum(arm["expected_rows"] for arm in sender_surveillance)),
        },
        "power_rules": {
            "target_power": 0.80,
            "alpha": 0.05,
            "cross_task_estimand": "coordination_specific = coord_delta - bet_delta",
            "primary_cross_task_regression": "join ~ surveillance_message * coordination_task + task-cell controls",
            "main_run_rule": "main rows must be computed from pilot interaction variance; do not assume 1000 rows is enough",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Power the cross-task interaction from pilot outputs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status")
    p_status.add_argument("--manifest", default=str(CLEAN_RUN_ROOT / "plans" / "main.yaml"))
    p_status.add_argument("--output", default=str(CLEAN_RUN_ROOT / "artifacts" / "power_analysis.json"))

    p_cross = sub.add_parser("cross-task")
    p_cross.add_argument("--output-root", default=str(CLEAN_RUN_ROOT / "output" / "pilot"))
    p_cross.add_argument("--output", default=str(CLEAN_RUN_ROOT / "artifacts" / "power_cross_task_interaction.json"))
    args = parser.parse_args()

    if args.command == "status":
        result = build_power_analysis_status(args.manifest)
    else:
        result = estimate_cross_task_power(load_periods(args.output_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
