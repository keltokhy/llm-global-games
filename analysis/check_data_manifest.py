#!/usr/bin/env python3
"""Check that replication-data manifests point at real local artifacts."""

from __future__ import annotations

import csv
import glob
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
ASSET_MANIFEST = ROOT / "paper" / "asset_manifest.tsv"

REQUIRED_OUTPUT_PATTERNS = [
    "*/calibrated_params_*.json",
    "*/calibrated_index.json",
    "prompt-isolation-surveillance/",
    "prompt-isolation-surveillance-placebo/",
    "prompt-isolation-surveillance-anonymous/",
    "surveillance/",
    "fixed-messages-surv/",
    "no-messages/",
    "no-messages-llama/",
    "no-messages-qwen30b/",
    "cross-task-placebo-baseline/",
    "cross-task-placebo-surveillance/",
    "xmodel-source-llama-baseline/",
    "xmodel-source-llama-surveillance/",
    "xmodel-source-qwen-baseline/",
    "xmodel-source-qwen-surveillance/",
    "xmodel-matched-llama-writes-qwen-reads-baseline/",
    "xmodel-matched-llama-writes-qwen-reads-surveillance/",
    "xmodel-matched-qwen-writes-llama-reads-baseline/",
    "xmodel-matched-qwen-writes-llama-reads-surveillance/",
    "revision-beliefs-*/",
    "punishment-risk*/",
    "temperature-robustness*/",
    "cross-generator*/",
    "mistralai--mistral-small-creative-n*/",
    "network-k8/",
    "mixed-5model-pure/",
    "mixed-5model-comm/",
    "mixed-mistral-gptoss-comm/",
    "holdout-validation/",
    "group-size-info/",
    "bandwidth-005/",
    "bandwidth-030/",
    "z-centered/",
]

REQUIRED_ARCHIVED_INFODESIGN_PATTERNS = [
    "mistralai--mistral-small-creative-infodesign-comm/",
    "surveillance-x-censorship/",
]

CORE_SUMMARY_PATTERNS = [
    "output/*/experiment_pure_summary.csv",
    "output/*/experiment_comm_summary.csv",
    "output/*/experiment_scramble_summary.csv",
    "output/*/experiment_flip_summary.csv",
]


def _expand_braces(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]

    expanded: list[str] = []
    for option in match.group(1).split(","):
        expanded.extend(_expand_braces(pattern[: match.start()] + option + pattern[match.end() :]))
    return expanded


def _matches(pattern: str) -> list[Path]:
    paths: list[Path] = []
    for expanded in _expand_braces(pattern):
        candidate = ROOT / expanded
        if any(char in expanded for char in "*?["):
            paths.extend(Path(match) for match in glob.glob(str(candidate), recursive=True))
        elif candidate.exists():
            paths.append(candidate)
    return paths


def _check_asset_manifest() -> list[str]:
    problems: list[str] = []
    if not ASSET_MANIFEST.exists():
        return [f"{ASSET_MANIFEST.relative_to(ROOT)} is missing"]

    with ASSET_MANIFEST.open(newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    if not rows:
        return ["paper/asset_manifest.tsv has no rows"]

    for row in rows:
        asset = ROOT / "paper" / row["asset"]
        generator = ROOT / row["generator"]
        if not asset.exists():
            problems.append(f"listed asset missing: {row['asset']}")
        if not generator.exists():
            problems.append(f"listed generator missing: {row['generator']}")

        for raw_pattern in [item.strip() for item in row["primary inputs"].split(";") if item.strip()]:
            if not _matches(raw_pattern):
                problems.append(f"{row['asset']} input pattern has no matches: {raw_pattern}")

    return problems


def _check_output_patterns() -> list[str]:
    problems: list[str] = []
    if not OUTPUT.exists():
        return ["output/ is missing"]

    for pattern in REQUIRED_OUTPUT_PATTERNS:
        if not _matches(f"output/{pattern}"):
            problems.append(f"DATA_MANIFEST output pattern has no matches: output/{pattern}")

    for pattern in CORE_SUMMARY_PATTERNS:
        if len(_matches(pattern)) < 1:
            problems.append(f"core summary pattern has no matches: {pattern}")

    for pattern in REQUIRED_ARCHIVED_INFODESIGN_PATTERNS:
        archived_pattern = f"_archive/infodesign/output/{pattern}"
        if not _matches(archived_pattern):
            problems.append(f"DATA_MANIFEST archived output pattern has no matches: {archived_pattern}")

    return problems


def main() -> int:
    problems = _check_asset_manifest() + _check_output_patterns()
    if problems:
        print("Data manifest check failed:", file=sys.stderr)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    print("Data manifest checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
