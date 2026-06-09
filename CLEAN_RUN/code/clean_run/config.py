"""Manifest loading and validation for the clean-run pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CLEAN_RUN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CLEAN_RUN_ROOT.parent


REQUIRED_ARM_FIELDS = (
    "arm_id",
    "claim",
    "model",
    "role",
    "n_agents",
    "n_countries",
    "n_periods",
    "seed",
    "theta_grid_mode",
    "message_source",
    "message_transform",
    "decision_task",
    "message_stage_context",
    "decision_context",
    "belief_timing",
    "belief_information",
    "output_dir",
    "expected_rows",
    "primary_outcomes",
    "preregistered_comparisons",
    "power_target",
    "exclusion_rules",
)


ALLOWED_MESSAGE_STAGE_CONTEXTS = {
    "none",
    "surveillance_full",
    "surveillance_placebo",
    "surveillance_anonymous",
}
ALLOWED_DECISION_CONTEXTS = {
    "none",
    "surveillance_full",
    "surveillance_placebo",
    "surveillance_anonymous",
}
ALLOWED_DECISION_TASKS = {"coordination", "individual_bet"}
ALLOWED_BELIEF_TIMING = {"none", "pre", "post", "both"}
ALLOWED_BELIEF_INFORMATION = {"none", "messages_excluded", "messages_included"}
ALLOWED_MESSAGE_SOURCES = {"none", "live", "message_bank", "fixed_output"}


@dataclass(frozen=True)
class Manifest:
    """A validated experiment manifest with defaults applied to each arm."""

    path: Path
    raw: dict[str, Any]
    arms: list[dict[str, Any]]

    @property
    def run_group(self) -> str:
        return str(self.raw.get("run_group", self.path.stem))

    @property
    def api_base_url(self) -> str:
        return str(self.raw.get("api_base_url", "https://openrouter.ai/api/v1"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def load_manifest(path: str | Path) -> Manifest:
    """Load a YAML manifest and apply top-level defaults to every arm."""

    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()
    with open(manifest_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.get("defaults", {}) or {}
    arms = []
    for arm in raw.get("arms", []) or []:
        resolved = _deep_merge(defaults, arm)
        if "output_dir" in resolved:
            resolved["output_dir"] = _resolve_path(resolved["output_dir"])
        if "message_bank_path" in resolved and resolved["message_bank_path"]:
            resolved["message_bank_path"] = _resolve_path(resolved["message_bank_path"])
        arms.append(resolved)
    return Manifest(path=manifest_path, raw=raw, arms=arms)


def validate_manifest(manifest: Manifest) -> list[str]:
    """Return human-readable validation errors. Empty list means valid."""

    errors: list[str] = []
    seen_arm_ids: set[str] = set()

    if not manifest.arms:
        errors.append("manifest has no arms")
        return errors

    for index, arm in enumerate(manifest.arms):
        arm_id = arm.get("arm_id", f"<arm {index}>")
        missing = [field for field in REQUIRED_ARM_FIELDS if field not in arm]
        if missing:
            errors.append(f"{arm_id}: missing required fields: {', '.join(missing)}")
            continue

        if arm_id in seen_arm_ids:
            errors.append(f"{arm_id}: duplicate arm_id")
        seen_arm_ids.add(str(arm_id))

        for int_field in ("n_agents", "n_countries", "n_periods", "seed", "expected_rows"):
            try:
                value = int(arm[int_field])
            except (TypeError, ValueError):
                errors.append(f"{arm_id}: {int_field} must be an integer")
                continue
            if value <= 0:
                errors.append(f"{arm_id}: {int_field} must be positive")

        expected = int(arm["n_countries"]) * int(arm["n_periods"])
        if int(arm["expected_rows"]) != expected:
            errors.append(
                f"{arm_id}: expected_rows={arm['expected_rows']} but "
                f"n_countries*n_periods={expected}"
            )

        if arm["message_stage_context"] not in ALLOWED_MESSAGE_STAGE_CONTEXTS:
            errors.append(f"{arm_id}: invalid message_stage_context={arm['message_stage_context']!r}")
        if arm["decision_context"] not in ALLOWED_DECISION_CONTEXTS:
            errors.append(f"{arm_id}: invalid decision_context={arm['decision_context']!r}")
        if arm["decision_task"] not in ALLOWED_DECISION_TASKS:
            errors.append(f"{arm_id}: invalid decision_task={arm['decision_task']!r}")
        if arm["belief_timing"] not in ALLOWED_BELIEF_TIMING:
            errors.append(f"{arm_id}: invalid belief_timing={arm['belief_timing']!r}")
        if arm["belief_information"] not in ALLOWED_BELIEF_INFORMATION:
            errors.append(f"{arm_id}: invalid belief_information={arm['belief_information']!r}")
        if arm["message_source"] not in ALLOWED_MESSAGE_SOURCES:
            errors.append(f"{arm_id}: invalid message_source={arm['message_source']!r}")
        if not isinstance(arm["primary_outcomes"], list) or not arm["primary_outcomes"]:
            errors.append(f"{arm_id}: primary_outcomes must be a non-empty list")
        if not isinstance(arm["preregistered_comparisons"], list):
            errors.append(f"{arm_id}: preregistered_comparisons must be a list")
        if not isinstance(arm["exclusion_rules"], list):
            errors.append(f"{arm_id}: exclusion_rules must be a list")

        output_dir = Path(str(arm["output_dir"]))
        try:
            output_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"{arm_id}: output_dir must stay under the project root")

        if arm["message_source"] == "message_bank" and not arm.get("message_bank_path"):
            errors.append(f"{arm_id}: message_bank_path is required for message_bank arms")

    return errors


def require_valid_manifest(path: str | Path) -> Manifest:
    manifest = load_manifest(path)
    errors = validate_manifest(manifest)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid manifest {manifest.path}:\n{joined}")
    return manifest
