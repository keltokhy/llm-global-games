"""Render available CLEAN_RUN figures from local artifacts only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import CLEAN_RUN_ROOT, require_valid_manifest


def render_available_figures(manifest_path: str | Path, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered = [
        _render_manifest_arm_counts(manifest_path, output),
    ]
    balance_path = CLEAN_RUN_ROOT / "artifacts" / "direct_coded_balance_table.csv"
    if balance_path.exists():
        rendered.append(_render_balance_figure(balance_path, output))
    _write_live_figure_status(output)
    return rendered


def _render_manifest_arm_counts(manifest_path: str | Path, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    manifest = require_valid_manifest(manifest_path)
    counts = Counter(arm["claim"] for arm in manifest.arms)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.barh(labels, values, color="#386cb0")
    ax.set_xlabel("Declared arms")
    ax.set_ylabel("Claim family")
    ax.set_title("CLEAN_RUN Manifest Coverage")
    ax.invert_yaxis()
    for i, value in enumerate(values):
        ax.text(value + 0.2, i, str(value), va="center", fontsize=8)
    fig.tight_layout()
    path = output / "fig_manifest_arm_counts.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _render_balance_figure(balance_path: Path, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    df = pd.read_csv(balance_path)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#b2182b" if not ok else "#4daf4a" for ok in df["passes_0_10sd_rule"]]
    ax.barh(df["dimension"], df["standardized_difference"], color=colors)
    ax.axvline(-0.10, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0.10, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized coded-minus-direct difference")
    ax.set_ylabel("Message feature")
    ax.set_title("Direct/Coded Source-Bank Balance")
    ax.invert_yaxis()
    fig.tight_layout()
    path = output / "fig_direct_coded_balance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _write_live_figure_status(output: Path) -> Path:
    status = {
        "status": "dependency_needed",
        "dependency_needed": [
            "CLEAN_RUN/output/**/periods.parquet",
            "CLEAN_RUN/output/**/agents.parquet",
            "QC-passing CLEAN_RUN/message_banks/direct_coded_pairs.parquet",
        ],
        "pending_figures": [
            "signal benchmark pure/scramble/flip",
            "sender-side surveillance join curves",
            "direct/coded intervention effects",
            "pre-decision beliefs and actions",
            "cross-task decomposition",
        ],
    }
    path = output / "figure_status.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render available CLEAN_RUN figures")
    parser.add_argument("--manifest", default=str(CLEAN_RUN_ROOT / "plans" / "main.yaml"))
    parser.add_argument("--output-dir", default=str(CLEAN_RUN_ROOT / "artifacts"))
    args = parser.parse_args()

    for path in render_available_figures(args.manifest, args.output_dir):
        print(f"wrote {path}")
    print(f"wrote {Path(args.output_dir) / 'figure_status.json'}")


if __name__ == "__main__":
    main()
