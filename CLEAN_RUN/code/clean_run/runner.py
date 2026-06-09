"""Manifest-driven runner for CLEAN_RUN experiment arms."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from agent_based_simulation.briefing import BriefingGenerator
from agent_based_simulation.experiment import Agent, run_communication_game, run_pure_global_game
from agent_based_simulation.runtime import build_network, parse_float_list

from .config import CLEAN_RUN_ROOT, PROJECT_ROOT, require_valid_manifest
from .grids import task_cells
from .message_bank import load_bank, messages_for_period, validate_bank
from .qc import write_qc
from .schema import agent_rows, period_rows


class RunLog:
    def __init__(self) -> None:
        self.stdout: list[str] = []
        self.stderr: list[str] = []

    def info(self, message: str) -> None:
        print(message, flush=True)
        self.stdout.append(message)

    def error(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)
        self.stderr.append(message)


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()
    except Exception as exc:  # pragma: no cover - defensive metadata path
        return f"[unavailable: {exc}]"


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "pyarrow", "openai", "networkx", "scipy", "pyyaml"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _provider(arm: dict[str, Any]) -> dict[str, Any] | None:
    provider = arm.get("provider")
    if not provider:
        return None
    return dict(provider)


def _surveillance_mode(context: str) -> tuple[bool, str]:
    if context == "none":
        return False, "full"
    if context == "surveillance_full":
        return True, "full"
    if context == "surveillance_placebo":
        return True, "placebo"
    if context == "surveillance_anonymous":
        return True, "anonymous"
    raise ValueError(f"Unknown message_stage_context={context!r}")


def _decision_context(context: str) -> str:
    return {
        "none": "none",
        "surveillance_full": "full",
        "surveillance_placebo": "placebo",
        "surveillance_anonymous": "anonymous",
    }[context]


def _briefing_kwargs(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "cutoff_center": float(arm.get("cutoff_center", 0.0)),
        "clarity_width": float(arm.get("clarity_width", 1.0)),
        "direction_slope": float(arm.get("direction_slope", 0.8)),
        "coordination_slope": float(arm.get("coordination_slope", 0.6)),
        "dissent_floor": float(arm.get("dissent_floor", 0.25)),
        "mixed_cue_clarity": float(arm.get("mixed_cue_clarity", 0.5)),
        "bottomline_cuts": parse_float_list(
            arm.get("bottomline_cuts", "0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85")
        ),
        "unclear_cuts": parse_float_list(arm.get("unclear_cuts", "0.18,0.33,0.48,0.62,0.77")),
        "coordination_cuts": parse_float_list(arm.get("coordination_cuts", "0.12,0.25,0.42,0.58,0.75")),
        "coordination_blend_prob": float(arm.get("coordination_blend_prob", 0.6)),
        "language_variant": arm.get("language_variant", "baseline"),
        "seed": int(arm["seed"]),
        "direction_transform": arm.get("direction_transform", "logistic"),
        "disabled_domains": arm.get("disabled_domains", []),
    }


def _network_for_arm(arm: dict[str, Any]) -> dict[int, list[int]]:
    n_agents = int(arm["n_agents"])
    n_neighbors = int(arm.get("n_neighbors", 4))
    if n_neighbors >= n_agents:
        n_neighbors = max(2, n_agents - 1)
    if n_neighbors % 2:
        n_neighbors -= 1
    if n_neighbors < 2:
        raise ValueError("n_agents must be at least 3 for communication arms")
    adjacency, _graph = build_network(n_agents, n_neighbors=n_neighbors, rewire_prob=0.3, seed=int(arm["seed"]))
    return adjacency


async def _run_arm_async(arm: dict[str, Any], api_base_url: str, log: RunLog) -> list[Any]:
    from openai import AsyncOpenAI

    bank = None
    if arm.get("message_bank_path"):
        bank_path = Path(arm["message_bank_path"])
        if not bank_path.exists():
            raise RuntimeError(f"Dependency needed: {bank_path}")
        bank = load_bank(bank_path)
        if arm["claim"] == "direct_coded_mechanism":
            bank_qc = validate_bank(bank)
            if not bank_qc["pass"]:
                raise RuntimeError(
                    "Dependency needed: direct/coded message bank does not pass QC. "
                    f"Path: {bank_path}. QC: {bank_qc}. "
                    "Rebuild with a factual-equivalence rewrite/QC workflow before running this arm."
                )
        required_cells = {(int(cell["country"]), int(cell["period"])) for cell in task_cells(arm)}
        bank_cells = {
            (int(row.country), int(row.period))
            for row in bank[["country", "period"]].drop_duplicates().itertuples(index=False)
        }
        missing_cells = sorted(required_cells - bank_cells)
        if missing_cells:
            preview = ", ".join(f"{country}:{period}" for country, period in missing_cells[:10])
            raise RuntimeError(
                "Dependency needed: message bank does not cover required task cells. "
                f"Path: {bank_path}. Missing {len(missing_cells)} cells; first missing: {preview}"
            )

    if "openrouter.ai" in api_base_url and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Dependency needed: OPENROUTER_API_KEY")

    api_key = os.environ.get("OPENROUTER_API_KEY", "") or "not-needed"
    client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(int(arm.get("max_concurrent", 50)))
    briefing_gen = BriefingGenerator(**_briefing_kwargs(arm))
    adjacency = _network_for_arm(arm)
    cells = task_cells(arm)
    provider = _provider(arm)

    async def run_cell(cell: dict[str, Any]) -> Any:
        agents = [Agent(agent_id=i, neighbors=adjacency[i]) for i in range(int(arm["n_agents"]))]
        signal_mode = arm.get("signal_mode", "normal")
        belief_timing = arm["belief_timing"]
        include_beliefs = belief_timing != "none"
        beliefs_include_messages = arm["belief_information"] == "messages_included"
        call_kwargs = {
            "llm_max_retries": int(arm.get("llm_max_retries", 5)),
            "llm_empty_retries": int(arm.get("llm_empty_retries", 12)),
            "cost": float(arm.get("cost", 1.0)),
            "group_size_info": bool(arm.get("group_size_info", False)),
            "elicit_beliefs": include_beliefs,
            "elicit_second_order": include_beliefs,
            "elicit_shared_understanding": include_beliefs,
            "elicit_others_expect_join": include_beliefs,
            "belief_order": belief_timing if belief_timing != "none" else "post",
            "second_order_order": belief_timing if belief_timing != "none" else "post",
            "shared_understanding_order": belief_timing if belief_timing != "none" else "post",
            "others_expect_join_order": belief_timing if belief_timing != "none" else "post",
            "beliefs_include_messages": beliefs_include_messages,
            "temperature": float(arm.get("temperature", 0.7)),
            "provider": provider,
            "extra_body": arm.get("extra_body"),
        }

        if _is_pure_arm(arm):
            return await run_pure_global_game(
                agents,
                float(cell["theta"]),
                float(cell["z_public"]),
                float(arm.get("sigma", 0.3)),
                float(cell["benefit"]),
                briefing_gen,
                client,
                arm["model"],
                semaphore,
                int(cell["country"]),
                int(cell["period"]),
                signal_mode=signal_mode,
                **call_kwargs,
            )

        fixed_messages = None
        source_key = None
        if bank is not None:
            fixed_messages, source_key = messages_for_period(
                bank,
                country=int(cell["country"]),
                period=int(cell["period"]),
                transform=arm["message_transform"],
            )
            if not fixed_messages:
                raise RuntimeError(
                    "Dependency needed: message bank lacks rows for "
                    f"{arm['arm_id']} country={cell['country']} period={cell['period']} "
                    f"at {arm['message_bank_path']}"
                )

        surveillance, surveillance_mode = _surveillance_mode(arm["message_stage_context"])
        return await run_communication_game(
            agents,
            float(cell["theta"]),
            float(cell["z_public"]),
            float(arm.get("sigma", 0.3)),
            float(cell["benefit"]),
            briefing_gen,
            client,
            arm["model"],
            semaphore,
            int(cell["country"]),
            int(cell["period"]),
            signal_mode=signal_mode,
            surveillance=surveillance,
            surveillance_mode=surveillance_mode,
            decision_context=_decision_context(arm["decision_context"]),
            fixed_messages=fixed_messages,
            degrade_messages=arm["message_transform"] == "degraded",
            no_peer_messages=arm["message_transform"] == "no_peer",
            message_bundle_mode=_message_bundle_mode(arm, fixed_messages),
            message_source_key=source_key,
            message_model_name=arm.get("writer_model"),
            decision_model_name=arm.get("reader_model") or arm["model"],
            task_mode=arm["decision_task"],
            **call_kwargs,
        )

    try:
        tasks = [run_cell(cell) for cell in cells]
        results = await asyncio.gather(*tasks)
        log.info(f"{arm['arm_id']}: completed {len(results)} period rows")
        return list(results)
    finally:
        await client.close()


def _is_pure_arm(arm: dict[str, Any]) -> bool:
    return arm["message_source"] == "none" and arm["message_transform"] == "none" and arm["role"] == "reader_only"


def _message_bundle_mode(arm: dict[str, Any], fixed_messages: dict[int, str] | None) -> str:
    if arm["message_transform"] == "no_peer":
        return "none"
    if arm["message_transform"] == "degraded":
        return "degraded"
    if fixed_messages is not None:
        return arm["message_transform"]
    return "live"


def run_arm(arm: dict[str, Any], *, api_base_url: str, command: str) -> dict[str, Any]:
    output_dir = Path(arm["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)

    log = RunLog()
    start = datetime.now(timezone.utc)
    run_id = f"{arm['arm_id']}-{start.strftime('%Y%m%dT%H%M%SZ')}"
    metadata = _metadata(arm, command, start)
    results: list[Any] = []

    try:
        log.info(f"running {arm['arm_id']} -> {output_dir}")
        results = asyncio.run(_run_arm_async(arm, api_base_url, log))
        periods = period_rows(results, arm, run_id)
        agents = agent_rows(results, arm, run_id)
        periods.to_parquet(output_dir / "periods.parquet", index=False)
        agents.to_parquet(output_dir / "agents.parquet", index=False)
        periods.head(5).to_csv(output_dir / "samples" / "periods_5rows.csv", index=False)
        agents.head(5).to_csv(output_dir / "samples" / "agents_5rows.csv", index=False)
        qc = write_qc(output_dir, arm)
        metadata["qc_pass"] = qc["pass"]
        return {"arm_id": arm["arm_id"], "output_dir": str(output_dir), "rows": len(periods), "qc": qc}
    except Exception as exc:
        log.error(str(exc))
        metadata["error"] = str(exc)
        raise
    finally:
        end = datetime.now(timezone.utc)
        metadata["end_time"] = end.isoformat()
        metadata["elapsed_seconds"] = (end - start).total_seconds()
        with open(output_dir / "stdout.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log.stdout) + ("\n" if log.stdout else ""))
        with open(output_dir / "stderr.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log.stderr) + ("\n" if log.stderr else ""))
        with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        with open(output_dir / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"arm": arm}, f, sort_keys=False)


def _metadata(arm: dict[str, Any], command: str, start: datetime) -> dict[str, Any]:
    return {
        "git_commit_hash": _git_output(["git", "rev-parse", "HEAD"]),
        "dirty_git_status": _git_output(["git", "status", "--short"]),
        "command": command,
        "start_time": start.isoformat(),
        "model": arm["model"],
        "api_base_url": arm.get("api_base_url", "https://openrouter.ai/api/v1"),
        "seed": arm["seed"],
        "package_versions": _package_versions(),
        "environment": {
            "OPENROUTER_API_KEY": "set" if os.environ.get("OPENROUTER_API_KEY") else "unset",
            "GGC_LLM_CACHE_DIR": os.environ.get("GGC_LLM_CACHE_DIR", ""),
        },
        "provider": arm.get("provider"),
    }


def run_manifest(manifest_path: str | Path, arm_ids: list[str] | None = None) -> list[dict[str, Any]]:
    manifest = require_valid_manifest(manifest_path)
    selected = [
        arm for arm in manifest.arms
        if not arm_ids or arm["arm_id"] in set(arm_ids)
    ]
    if arm_ids and len(selected) != len(set(arm_ids)):
        found = {arm["arm_id"] for arm in selected}
        missing = sorted(set(arm_ids) - found)
        raise ValueError(f"Unknown arm_id(s): {missing}")

    command = " ".join(sys.argv)
    summaries = []
    for arm in selected:
        arm = {**arm, "api_base_url": manifest.api_base_url}
        summaries.append(run_arm(arm, api_base_url=manifest.api_base_url, command=command))
    return summaries


def write_preregistration(manifest_path: str | Path, output: str | Path | None = None) -> Path:
    manifest = require_valid_manifest(manifest_path)
    output_path = Path(output) if output else CLEAN_RUN_ROOT / "preregistration.md"
    lines = [
        "# CLEAN_RUN Pre-registration",
        "",
        f"Manifest: `{manifest.path}`",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Git commit: `{_git_output(['git', 'rev-parse', 'HEAD'])}`",
        "",
        "## Arms",
        "",
        "| arm_id | claim | model | expected_rows | primary_outcomes |",
        "|---|---|---|---:|---|",
    ]
    for arm in manifest.arms:
        outcomes = ", ".join(arm["primary_outcomes"])
        lines.append(
            f"| `{arm['arm_id']}` | {arm['claim']} | `{arm['model']}` | "
            f"{arm['expected_rows']} | {outcomes} |"
        )
    lines.extend(
        [
            "",
            "## Exclusion Rules",
            "",
        ]
    )
    for arm in manifest.arms:
        rules = "; ".join(str(rule) for rule in arm["exclusion_rules"])
        lines.append(f"- `{arm['arm_id']}`: {rules}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLEAN_RUN manifest arms")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--manifest", default=str(CLEAN_RUN_ROOT / "plans" / "pilot.yaml"))

    p_run = sub.add_parser("run")
    p_run.add_argument("--manifest", default=str(CLEAN_RUN_ROOT / "plans" / "pilot.yaml"))
    p_run.add_argument("--arm", action="append", dest="arms")

    p_prereg = sub.add_parser("preregister")
    p_prereg.add_argument("--manifest", default=str(CLEAN_RUN_ROOT / "plans" / "main.yaml"))
    p_prereg.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "validate":
        manifest = require_valid_manifest(args.manifest)
        print(f"valid manifest: {manifest.path} ({len(manifest.arms)} arms)")
    elif args.command == "run":
        started = time.time()
        summaries = run_manifest(args.manifest, args.arms)
        print(json.dumps(summaries, indent=2, sort_keys=True))
        print(f"elapsed_seconds={time.time() - started:.3f}")
    elif args.command == "preregister":
        path = write_preregistration(args.manifest, args.output)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
