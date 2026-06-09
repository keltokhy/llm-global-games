"""Analysis entrypoint for CLEAN_RUN outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CLEAN_RUN_ROOT, require_valid_manifest
from .message_bank import load_bank, validate_bank
from .qc import exact_match_rate


MIN_EXACT_MATCH_RATE = 0.95


class MatchedComparisonError(RuntimeError):
    """Raised when a preregistered matched comparison lacks exact overlap."""


def _read_outputs(manifest) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_frames = []
    agent_frames = []
    for arm in manifest.arms:
        output = Path(arm["output_dir"])
        periods_path = output / "periods.parquet"
        agents_path = output / "agents.parquet"
        if periods_path.exists():
            period_frames.append(pd.read_parquet(periods_path))
        if agents_path.exists():
            agent_frames.append(pd.read_parquet(agents_path))
    periods = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    agents = pd.concat(agent_frames, ignore_index=True) if agent_frames else pd.DataFrame()
    return periods, agents


def _arm_summary(periods: pd.DataFrame) -> list[dict[str, Any]]:
    if periods.empty:
        return []
    grouped = periods.groupby(["arm_id", "model"], dropna=False)
    rows = []
    for (arm_id, model), df in grouped:
        rows.append(
            {
                "arm_id": arm_id,
                "model": model,
                "period_rows": int(len(df)),
                "mean_join_fraction_valid": _safe_mean(df["join_fraction_valid"]),
                "api_error_rate": _safe_mean(df["api_error_rate"]),
                "unparseable_rate": _safe_mean(df["unparseable_rate"]),
            }
        )
    return rows


def _safe_mean(series: pd.Series) -> float | None:
    value = series.mean(skipna=True)
    return None if pd.isna(value) else float(value)


def _correlation(series_a: pd.Series, series_b: pd.Series) -> float | None:
    clean = pd.concat([series_a, series_b], axis=1).dropna()
    if len(clean) < 3:
        return None
    return float(clean.iloc[:, 0].corr(clean.iloc[:, 1]))


def _paired_effect(periods: pd.DataFrame, treatment: str, control: str) -> dict[str, Any]:
    left = periods[periods["arm_id"].eq(treatment)]
    right = periods[periods["arm_id"].eq(control)]
    if left.empty or right.empty:
        return {"treatment": treatment, "control": control, "available": False}
    rate = exact_match_rate(left, right)
    if rate < MIN_EXACT_MATCH_RATE:
        raise MatchedComparisonError(
            f"Exact matched overlap for {treatment} vs {control} is {rate:.3f}, "
            f"below required {MIN_EXACT_MATCH_RATE:.2f}. Do not switch to unpaired inference."
        )
    merged = left.merge(
        right,
        on=["model", "country", "period", "theta", "z_public", "benefit", "cost", "theta_star"],
        suffixes=("_treat", "_control"),
    )
    delta = merged["join_fraction_valid_treat"] - merged["join_fraction_valid_control"]
    return {
        "treatment": treatment,
        "control": control,
        "available": True,
        "exact_match_rate": rate,
        "match_gate": "pass",
        "matched_rows": int(len(merged)),
        "mean_delta": _safe_mean(delta),
    }


def build_verified_stats(manifest_path: str | Path) -> dict[str, Any]:
    manifest = require_valid_manifest(manifest_path)
    periods, agents = _read_outputs(manifest)

    stats: dict[str, Any] = {
        "manifest": str(manifest.path),
        "period_rows": int(len(periods)),
        "agent_rows": int(len(agents)),
        "arms_declared": len(manifest.arms),
        "arms_observed": sorted(periods["arm_id"].unique().tolist()) if not periods.empty else [],
        "arm_summary": _arm_summary(periods),
        "signal_benchmark": [],
        "paired_effects": [],
        "message_banks": [],
    }

    if not periods.empty:
        for arm_id, df in periods.groupby("arm_id"):
            stats["signal_benchmark"].append(
                {
                    "arm_id": arm_id,
                    "r_join_theta": _correlation(df["join_fraction_valid"], df["theta"]),
                    "r_join_theoretical_attack": _correlation(
                        df["join_fraction_valid"], df["theoretical_attack"]
                    ),
                }
            )
        comparisons = _comparisons_from_manifest(manifest.arms)
        stats["paired_effects"] = [
            _paired_effect(periods, treatment, control)
            for treatment, control in comparisons
        ]

    bank_paths = sorted((CLEAN_RUN_ROOT / "message_banks").glob("*.parquet"))
    for path in bank_paths:
        stats["message_banks"].append(
            {"path": str(path), **validate_bank(load_bank(path))}
        )

    return stats


def _comparisons_from_manifest(arms: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    known_ids = {arm["arm_id"] for arm in arms}
    for arm in arms:
        for item in arm.get("preregistered_comparisons", []):
            if isinstance(item, dict):
                treatment = item.get("treatment")
                control = item.get("control")
            elif isinstance(item, str) and "-" in item:
                treatment, control = [part.strip() for part in item.split("-", 1)]
            else:
                continue
            if treatment in known_ids and control in known_ids:
                pairs.add((treatment, control))
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce CLEAN_RUN/artifacts/verified_stats.json")
    parser.add_argument("--manifest", default=str(CLEAN_RUN_ROOT / "plans" / "main.yaml"))
    parser.add_argument("--output", default=str(CLEAN_RUN_ROOT / "artifacts" / "verified_stats.json"))
    args = parser.parse_args()

    stats = build_verified_stats(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
