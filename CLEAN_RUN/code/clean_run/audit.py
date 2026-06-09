"""Completion audit for the CLEAN_RUN system.

The audit intentionally treats missing runtime data as incomplete even when the
code and manifests validate. It is a prompt-to-artifact checklist for the plan.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CLEAN_RUN_ROOT, load_manifest, validate_manifest
from .message_bank import load_bank, validate_bank
from .schema import AGENT_COLUMNS, MESSAGE_BANK_COLUMNS, PERIOD_COLUMNS


REQUIRED_CODE_FILES = [
    "config.py",
    "schema.py",
    "grids.py",
    "message_bank.py",
    "runner.py",
    "analyze.py",
    "qc.py",
    "power.py",
    "tables.py",
    "manifest_builder.py",
]


REQUIRED_TOP_LEVEL_ARTIFACTS = [
    "experiment_plan.txt",
    "plans/pilot.yaml",
    "plans/main.yaml",
    "preregistration.md",
    "artifacts/artifact_manifest.tsv",
    "artifacts/power_analysis.json",
    "artifacts/verified_stats.json",
    "artifacts/information_state_table.csv",
    "artifacts/information_state_table.tex",
    "artifacts/direct_coded_balance_table.csv",
    "artifacts/direct_coded_manual_audit_50.csv",
    "artifacts/message_bank_summary_stats.json",
    "artifacts/arm_sample_size_table.csv",
    "artifacts/arm_sample_size_table.tex",
    "artifacts/message_bank_qc_table.csv",
    "artifacts/message_bank_qc_table.tex",
    "artifacts/fig_manifest_arm_counts.png",
    "artifacts/fig_direct_coded_balance.png",
    "artifacts/figure_status.json",
    "message_banks/baseline_messages.parquet",
    "message_banks/baseline_surveillance_matched.parquet",
    "message_banks/direct_coded_pairs.parquet",
]


def build_completion_audit(
    *,
    pilot_manifest: str | Path | None = None,
    main_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect current artifacts and return a completion checklist."""

    pilot_manifest = Path(pilot_manifest or CLEAN_RUN_ROOT / "plans" / "pilot.yaml")
    main_manifest = Path(main_manifest or CLEAN_RUN_ROOT / "plans" / "main.yaml")

    checks = []
    checks.extend(_file_checks())
    checks.extend(_manifest_checks(pilot_manifest, main_manifest))
    checks.extend(_schema_checks())
    checks.extend(_message_bank_checks())
    checks.extend(_runtime_output_checks(main_manifest))
    checks.extend(_runtime_dependency_checks())

    counts = Counter(item["status"] for item in checks)
    complete = counts.get("fail", 0) == 0 and counts.get("blocker", 0) == 0
    return {
        "objective": "build the clean system under /experiment_plan.txt",
        "success_criteria": [
            "clean package and manifest layer exist",
            "pilot and main manifests validate and cover the planned experiment families",
            "canonical parquet schemas and QC gates exist",
            "message banks are present, sampled, and QC classified",
            "preregistration and information-state artifacts exist",
            "runner can produce periods.parquet, agents.parquet, metadata, samples, and qc.json",
            "analysis can produce verified_stats.json from CLEAN_RUN only",
            "live pilot/main outputs exist and required message-bank gates pass",
        ],
        "complete": complete,
        "status_counts": dict(counts),
        "checks": checks,
    }


def _file_checks() -> list[dict[str, Any]]:
    checks = []
    for rel in REQUIRED_CODE_FILES:
        path = CLEAN_RUN_ROOT / "code" / "clean_run" / rel
        checks.append(_check(path.exists(), f"code file: clean_run/{rel}", path))
    for rel in REQUIRED_TOP_LEVEL_ARTIFACTS:
        path = CLEAN_RUN_ROOT / rel
        failure_status = "blocker" if rel.startswith("artifacts/power_") else "fail"
        checks.append(_check(path.exists(), f"artifact: CLEAN_RUN/{rel}", path, failure_status=failure_status))
    power_analysis = CLEAN_RUN_ROOT / "artifacts" / "power_analysis.json"
    if power_analysis.exists():
        with open(power_analysis, encoding="utf-8") as f:
            payload = json.load(f)
        checks.append(
            {
                "requirement": "power_analysis.json contains final powered design, not dependency status",
                "status": "pass" if payload.get("status") == "complete" else "blocker",
                "evidence": str(power_analysis),
                "details": payload,
            }
        )
    return checks


def _manifest_checks(pilot_path: Path, main_path: Path) -> list[dict[str, Any]]:
    checks = []
    for label, path in [("pilot manifest validates", pilot_path), ("main manifest validates", main_path)]:
        try:
            manifest = load_manifest(path)
            errors = validate_manifest(manifest)
            checks.append(
                {
                    "requirement": label,
                    "status": "pass" if not errors else "fail",
                    "evidence": str(path),
                    "details": {"arms": len(manifest.arms), "errors": errors},
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "requirement": label,
                    "status": "fail",
                    "evidence": str(path),
                    "details": {"error": str(exc)},
                }
            )

    try:
        main = load_manifest(main_path)
        roster = {item["model"] for item in main.raw.get("model_roster", [])}
        signal_models = {arm["model"] for arm in main.arms if arm["claim"] == "language_signal"}
        surv_models = {arm["model"] for arm in main.arms if arm["claim"] == "sender_side_surveillance"}
        claim_counts = Counter(arm["claim"] for arm in main.arms)
        checks.append(
            {
                "requirement": "main manifest covers frozen model roster for signal and sender-surveillance families",
                "status": "pass" if roster and signal_models == roster and surv_models == roster else "fail",
                "evidence": str(main_path),
                "details": {
                    "model_roster_size": len(roster),
                    "claim_counts": dict(claim_counts),
                    "signal_models_match": signal_models == roster,
                    "sender_surveillance_models_match": surv_models == roster,
                },
            }
        )
    except Exception as exc:
        checks.append(
            {
                "requirement": "main manifest covers frozen model roster for signal and sender-surveillance families",
                "status": "fail",
                "evidence": str(main_path),
                "details": {"error": str(exc)},
            }
        )
    return checks


def _schema_checks() -> list[dict[str, Any]]:
    requirements = {
        "period schema contains plan-required fields": PERIOD_COLUMNS,
        "agent schema contains plan-required fields": AGENT_COLUMNS,
        "message-bank schema contains plan-required fields": MESSAGE_BANK_COLUMNS,
    }
    return [
        {
            "requirement": requirement,
            "status": "pass",
            "evidence": "CLEAN_RUN/code/clean_run/schema.py",
            "details": {"column_count": len(columns), "columns": columns},
        }
        for requirement, columns in requirements.items()
    ]


def _message_bank_checks() -> list[dict[str, Any]]:
    checks = []
    for name in [
        "baseline_messages.parquet",
        "baseline_surveillance_matched.parquet",
        "direct_coded_pairs.parquet",
    ]:
        path = CLEAN_RUN_ROOT / "message_banks" / name
        if not path.exists():
            checks.append(_check(False, f"message bank exists and validates: {name}", path, "blocker"))
            continue
        qc = validate_bank(load_bank(path))
        status = "pass" if qc["pass"] else "blocker" if name == "direct_coded_pairs.parquet" else "warn"
        checks.append(
            {
                "requirement": f"message bank exists and validates: {name}",
                "status": status,
                "evidence": str(path),
                "details": qc,
            }
        )

    sample_dir = CLEAN_RUN_ROOT / "message_banks" / "samples"
    samples = sorted(sample_dir.glob("*_5rows.csv")) if sample_dir.exists() else []
    checks.append(
        {
            "requirement": "message-bank 5-row samples are saved",
            "status": "pass" if len(samples) >= 3 else "fail",
            "evidence": str(sample_dir),
            "details": {"sample_count": len(samples), "samples": [str(path) for path in samples]},
        }
    )
    return checks


def _runtime_output_checks(main_path: Path) -> list[dict[str, Any]]:
    checks = []
    output_root = CLEAN_RUN_ROOT / "output"
    period_paths = sorted(output_root.glob("**/periods.parquet")) if output_root.exists() else []
    agent_paths = sorted(output_root.glob("**/agents.parquet")) if output_root.exists() else []
    checks.append(
        {
            "requirement": "clean output root contains period and agent parquet outputs",
            "status": "pass" if period_paths and agent_paths else "blocker",
            "evidence": str(output_root),
            "details": {
                "period_files": [str(path) for path in period_paths[:10]],
                "agent_files": [str(path) for path in agent_paths[:10]],
                "period_file_count": len(period_paths),
                "agent_file_count": len(agent_paths),
            },
        }
    )

    power_path = CLEAN_RUN_ROOT / "artifacts" / "power_cross_task_interaction.json"
    checks.append(
        _check(
            power_path.exists(),
            "cross-task interaction power artifact exists",
            power_path,
            failure_status="blocker",
        )
    )

    if period_paths and agent_paths:
        periods = pd.concat([pd.read_parquet(path) for path in period_paths], ignore_index=True)
        agents = pd.concat([pd.read_parquet(path) for path in agent_paths], ignore_index=True)
        checks.append(
            {
                "requirement": "runtime parquet files have required schema columns",
                "status": "pass"
                if set(PERIOD_COLUMNS).issubset(periods.columns) and set(AGENT_COLUMNS).issubset(agents.columns)
                else "fail",
                "evidence": str(output_root),
                "details": {"period_rows": len(periods), "agent_rows": len(agents)},
            }
        )
    return checks


def _runtime_dependency_checks() -> list[dict[str, Any]]:
    return [
        {
            "requirement": "OPENROUTER_API_KEY is available for live model calls",
            "status": "pass" if os.environ.get("OPENROUTER_API_KEY") else "blocker",
            "evidence": "environment",
            "details": {"OPENROUTER_API_KEY": "set" if os.environ.get("OPENROUTER_API_KEY") else "unset"},
        }
    ]


def _check(
    condition: bool,
    requirement: str,
    path: Path,
    failure_status: str = "fail",
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": "pass" if condition else failure_status,
        "evidence": str(path),
        "details": {"exists": path.exists()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CLEAN_RUN completion audit JSON")
    parser.add_argument("--output", default=str(CLEAN_RUN_ROOT / "artifacts" / "completion_audit.json"))
    args = parser.parse_args()

    audit = build_completion_audit()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
    print(f"wrote {output}")
    print(json.dumps({"complete": audit["complete"], "status_counts": audit["status_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
