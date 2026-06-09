"""Render clean-run tables that are derived directly from the manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CLEAN_RUN_ROOT, require_valid_manifest
from .message_bank import load_bank, validate_bank


INFO_STATE_COLUMNS = [
    "arm",
    "sender_sees_private_briefing",
    "sender_sees_monitoring_warning",
    "receiver_sees_private_briefing",
    "receiver_sees_peer_messages",
    "receiver_sees_monitoring_warning",
    "belief_prompt_includes_messages",
    "belief_timing",
    "decision_task",
]


def information_state_table(manifest_path: str | Path) -> pd.DataFrame:
    manifest = require_valid_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for arm in manifest.arms:
        rows.append(
            {
                "arm": arm["arm_id"],
                "sender_sees_private_briefing": _yes_no(arm["message_source"] == "live"),
                "sender_sees_monitoring_warning": _yes_no(arm["message_stage_context"] != "none"),
                "receiver_sees_private_briefing": "yes",
                "receiver_sees_peer_messages": _yes_no(
                    arm["message_source"] in {"live", "message_bank"}
                    and arm["message_transform"] != "no_peer"
                ),
                "receiver_sees_monitoring_warning": _yes_no(arm["decision_context"] != "none"),
                "belief_prompt_includes_messages": _yes_no(
                    arm["belief_information"] == "messages_included"
                ),
                "belief_timing": arm["belief_timing"],
                "decision_task": arm["decision_task"],
            }
        )
    return pd.DataFrame(rows, columns=INFO_STATE_COLUMNS)


def render_information_state(manifest_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = information_state_table(manifest_path)
    csv_path = output / "information_state_table.csv"
    tex_path = output / "information_state_table.tex"
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(_dataframe_to_latex(df))
    return csv_path, tex_path


def arm_sample_size_table(manifest_path: str | Path) -> pd.DataFrame:
    manifest = require_valid_manifest(manifest_path)
    rows = []
    for arm in manifest.arms:
        rows.append(
            {
                "arm_id": arm["arm_id"],
                "claim": arm["claim"],
                "model": arm["model"],
                "role": arm["role"],
                "n_agents": arm["n_agents"],
                "n_countries": arm["n_countries"],
                "n_periods": arm["n_periods"],
                "expected_rows": arm["expected_rows"],
                "message_source": arm["message_source"],
                "message_transform": arm["message_transform"],
                "decision_task": arm["decision_task"],
                "belief_timing": arm["belief_timing"],
                "belief_information": arm["belief_information"],
            }
        )
    return pd.DataFrame(rows)


def message_bank_qc_table(bank_dir: str | Path | None = None) -> pd.DataFrame:
    bank_root = Path(bank_dir) if bank_dir else CLEAN_RUN_ROOT / "message_banks"
    rows = []
    for path in sorted(bank_root.glob("*.parquet")):
        qc = validate_bank(load_bank(path))
        rows.append({"message_bank": path.name, **qc})
    return pd.DataFrame(rows)


def render_all_tables(manifest_path: str | Path, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []

    table_specs = [
        ("information_state_table", information_state_table(manifest_path)),
        ("arm_sample_size_table", arm_sample_size_table(manifest_path)),
        ("message_bank_qc_table", message_bank_qc_table()),
    ]
    balance_path = output / "direct_coded_balance_table.csv"
    if balance_path.exists():
        table_specs.append(("direct_coded_balance_table", pd.read_csv(balance_path)))

    for stem, df in table_specs:
        csv_path = output / f"{stem}.csv"
        tex_path = output / f"{stem}.tex"
        df.to_csv(csv_path, index=False)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(_dataframe_to_latex(df))
        rendered.extend([csv_path, tex_path])
    return rendered


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _dataframe_to_latex(df: pd.DataFrame) -> str:
    alignment = "l" * len(df.columns)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_latex_escape(col.replace("_", " ")) for col in df.columns) + r" \\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CLEAN_RUN manifest-derived tables")
    parser.add_argument("--manifest", default=str(CLEAN_RUN_ROOT / "plans" / "main.yaml"))
    parser.add_argument("--output-dir", default=str(CLEAN_RUN_ROOT / "artifacts"))
    parser.add_argument("--all", action="store_true", help="Render all currently available CLEAN_RUN tables")
    args = parser.parse_args()

    if args.all:
        for path in render_all_tables(args.manifest, args.output_dir):
            print(f"wrote {path}")
    else:
        csv_path, tex_path = render_information_state(args.manifest, args.output_dir)
        print(f"wrote {csv_path}")
        print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
