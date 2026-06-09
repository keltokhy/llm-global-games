"""Quality gates for clean-run manifests and outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_manifest, validate_manifest
from .schema import AGENT_COLUMNS, PERIOD_COLUMNS, validate_columns


MATCH_KEY = [
    "model",
    "country",
    "period",
    "theta",
    "z_public",
    "benefit",
    "cost",
    "theta_star",
]


def qc_output_dir(output_dir: str | Path, arm: dict[str, Any] | None = None) -> dict[str, Any]:
    output = Path(output_dir)
    result: dict[str, Any] = {
        "output_dir": str(output),
        "periods_exists": (output / "periods.parquet").exists(),
        "agents_exists": (output / "agents.parquet").exists(),
        "period_rows": 0,
        "agent_rows": 0,
        "missing_period_columns": [],
        "missing_agent_columns": [],
        "parse_error_rate": None,
        "api_error_rate": None,
        "expected_rows_match": None,
        "surveillance_leak_pass": None,
        "pass": False,
    }
    if not result["periods_exists"] or not result["agents_exists"]:
        return result

    periods = pd.read_parquet(output / "periods.parquet")
    agents = pd.read_parquet(output / "agents.parquet")
    result["period_rows"] = int(len(periods))
    result["agent_rows"] = int(len(agents))
    result["missing_period_columns"] = validate_columns(periods, PERIOD_COLUMNS)
    result["missing_agent_columns"] = validate_columns(agents, AGENT_COLUMNS)
    if "parse_error" in agents:
        result["parse_error_rate"] = float(agents["parse_error"].fillna(False).mean())
    if "api_error" in agents:
        result["api_error_rate"] = float(agents["api_error"].fillna(False).mean())
    if arm is not None:
        result["expected_rows_match"] = int(arm["expected_rows"]) == int(len(periods))
        result["surveillance_leak_pass"] = _surveillance_leak_pass(periods)

    result["pass"] = bool(
        result["periods_exists"]
        and result["agents_exists"]
        and not result["missing_period_columns"]
        and not result["missing_agent_columns"]
        and (result["expected_rows_match"] in (True, None))
        and (result["surveillance_leak_pass"] in (True, None))
        and (result["parse_error_rate"] is None or result["parse_error_rate"] < 0.02)
        and (result["api_error_rate"] is None or result["api_error_rate"] < 0.02)
    )
    return result


def _surveillance_leak_pass(periods: pd.DataFrame) -> bool:
    if periods.empty:
        return False
    sender_only = periods["message_stage_context"].eq("surveillance_full")
    if not sender_only.any():
        return True
    return bool(periods.loc[sender_only, "decision_context"].eq("none").all())


def exact_match_rate(left: pd.DataFrame, right: pd.DataFrame, keys: list[str] | None = None) -> float:
    keys = keys or MATCH_KEY
    missing = [key for key in keys if key not in left.columns or key not in right.columns]
    if missing:
        raise ValueError(f"Cannot match, missing key columns: {missing}")
    merged = left[keys].drop_duplicates().merge(
        right[keys].drop_duplicates(),
        on=keys,
        how="inner",
    )
    denom = max(len(left[keys].drop_duplicates()), len(right[keys].drop_duplicates()), 1)
    return float(len(merged) / denom)


def write_qc(output_dir: str | Path, arm: dict[str, Any]) -> dict[str, Any]:
    result = qc_output_dir(output_dir, arm)
    path = Path(output_dir) / "qc.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-run QC gates")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("validate-manifest")
    p_manifest.add_argument("--manifest", required=True)

    p_output = sub.add_parser("validate-output")
    p_output.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "validate-manifest":
        manifest = load_manifest(args.manifest)
        errors = validate_manifest(manifest)
        if errors:
            for error in errors:
                print(error)
            raise SystemExit(1)
        print(f"valid manifest: {manifest.path} ({len(manifest.arms)} arms)")
    elif args.command == "validate-output":
        result = qc_output_dir(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
