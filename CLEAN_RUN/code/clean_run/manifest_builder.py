"""Generate the canonical CLEAN_RUN main manifest from experiment templates."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import CLEAN_RUN_ROOT, validate_manifest, load_manifest


MODEL_ROSTER = [
    {
        "key": "qwen30",
        "model": "qwen/qwen3-30b-a3b-instruct-2507",
        "provider": {"only": ["siliconflow"], "allow_fallbacks": False},
    },
    {
        "key": "gemma4",
        "model": "google/gemma-4-26b-a4b-it-20260403",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
    },
    {
        "key": "deepseek",
        "model": "deepseek/deepseek-v4-flash-20260423",
        "provider": {"only": ["deepseek"], "allow_fallbacks": False},
    },
    {
        "key": "llama4",
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
    },
    {
        "key": "mistral",
        "model": "mistralai/mistral-small-2603",
        "provider": {"only": ["mistral"], "allow_fallbacks": False},
    },
    {
        "key": "glm51",
        "model": "z-ai/glm-5.1-20260406",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
    },
]


DEFAULTS = {
    "model": "deepseek/deepseek-v4-flash-20260423",
    "provider": {"only": ["deepseek"], "allow_fallbacks": False},
    "n_agents": 25,
    "n_countries": 5,
    "n_periods": 200,
    "seed": 5150,
    "theta_grid_mode": "random_normal",
    "sigma": 0.3,
    "benefit": 1.0,
    "cost": 1.0,
    "max_concurrent": 50,
    "llm_max_retries": 5,
    "llm_empty_retries": 12,
    "temperature": 0.7,
    "expected_rows": 1000,
    "power_target": "80pct_alpha_0.05_or_pilot_powered",
    "exclusion_rules": [
        "drop API-error and unparseable decisions from join_fraction_valid",
        "report parse and API error rates by arm",
        "trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point",
    ],
}


def build_main_manifest() -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    for model in MODEL_ROSTER:
        arms.extend(_signal_arms(model))
        arms.extend(_sender_surveillance_arms(model))

    for model in _models(["deepseek", "llama4", "qwen30"]):
        arms.extend(_message_control_arms(model))

    for model in _models(["deepseek", "llama4", "qwen30"]):
        arms.extend(_direct_coded_arms(model))

    for model in _models(["deepseek", "llama4"]):
        arms.extend(_prebelief_arms(model))

    for model in _models(["llama4", "mistral"]):
        arms.extend(_cross_task_arms(model))

    arms.extend(_cross_model_rotation_arms())

    return {
        "schema_version": 1,
        "run_group": "main",
        "api_base_url": "https://openrouter.ai/api/v1",
        "model_roster": [
            {"model": item["model"], "provider": item["provider"]}
            for item in MODEL_ROSTER
        ],
        "defaults": DEFAULTS,
        "arms": arms,
    }


def _models(keys: list[str]) -> list[dict[str, Any]]:
    lookup = {item["key"]: item for item in MODEL_ROSTER}
    return [lookup[key] for key in keys]


def _base_arm(model: dict[str, Any], arm_id: str, claim: str) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "claim": claim,
        "model": model["model"],
        "provider": model["provider"],
        "output_dir": f"CLEAN_RUN/output/main/{arm_id}",
        "preregistered_comparisons": [],
    }


def _signal_arms(model: dict[str, Any]) -> list[dict[str, Any]]:
    arms = []
    for mode, suffix in [("normal", "pure"), ("scramble", "scramble"), ("flip", "flip")]:
        arm = _base_arm(model, f"signal_{suffix}_{model['key']}", "language_signal")
        arm.update(
            {
                "role": "reader_only",
                "message_source": "none",
                "message_transform": "none",
                "signal_mode": mode,
                "decision_task": "coordination",
                "message_stage_context": "none",
                "decision_context": "none",
                "belief_timing": "none",
                "belief_information": "none",
                "primary_outcomes": ["join_fraction_valid"],
            }
        )
        if suffix != "pure":
            arm["preregistered_comparisons"] = [
                {"treatment": arm["arm_id"], "control": f"signal_pure_{model['key']}"}
            ]
        arms.append(arm)
    return arms


def _sender_surveillance_arms(model: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = _base_arm(model, f"comm_baseline_{model['key']}", "sender_side_surveillance")
    baseline.update(_live_comm_fields("none", "none"))
    surveillance = _base_arm(model, f"surv_sender_only_{model['key']}", "sender_side_surveillance")
    surveillance.update(_live_comm_fields("surveillance_full", "none"))
    surveillance["preregistered_comparisons"] = [
        {"treatment": surveillance["arm_id"], "control": baseline["arm_id"]}
    ]
    return [baseline, surveillance]


def _message_control_arms(model: dict[str, Any]) -> list[dict[str, Any]]:
    arms = []
    specs = [
        ("no_peer_messages", "message_value_control", "no_peer", "none"),
        ("degraded_messages", "generic_message_degradation", "degraded", "none"),
        ("monitored_for_research", "observation_placebo", "none", "surveillance_placebo"),
        ("anonymous_aggregation", "observation_placebo", "none", "surveillance_anonymous"),
    ]
    for prefix, claim, transform, stage_context in specs:
        arm = _base_arm(model, f"{prefix}_{model['key']}", claim)
        arm.update(_live_comm_fields(stage_context, "none"))
        arm["message_transform"] = transform
        arm["preregistered_comparisons"] = [
            {"treatment": arm["arm_id"], "control": f"comm_baseline_{model['key']}"}
        ]
        arms.append(arm)

    receiver = _base_arm(model, f"receiver_warning_{model['key']}", "direct_receiver_warning")
    receiver.update(
        _message_bank_fields(
            "CLEAN_RUN/message_banks/baseline_messages.parquet",
            "original",
            "coordination",
            "none",
            "surveillance_full",
            "none",
            "none",
        )
    )
    receiver["primary_outcomes"] = ["join_fraction_valid"]
    arms.append(receiver)
    return arms


def _direct_coded_arms(model: dict[str, Any]) -> list[dict[str, Any]]:
    direct = _base_arm(model, f"direct_replay_{model['key']}", "direct_coded_mechanism")
    direct.update(
        _message_bank_fields(
            "CLEAN_RUN/message_banks/direct_coded_pairs.parquet",
            "direct",
            "coordination",
            "none",
            "none",
            "pre",
            "messages_included",
        )
    )
    coded = deepcopy(direct)
    coded["arm_id"] = f"coded_replay_{model['key']}"
    coded["output_dir"] = f"CLEAN_RUN/output/main/{coded['arm_id']}"
    coded["message_transform"] = "coded"
    coded["preregistered_comparisons"] = [
        {"treatment": coded["arm_id"], "control": direct["arm_id"]}
    ]
    return [direct, coded]


def _prebelief_arms(model: dict[str, Any]) -> list[dict[str, Any]]:
    arms = []
    for source, stage_context in [("comm", "none"), ("surv", "surveillance_full")]:
        for info in ["messages_excluded", "messages_included"]:
            arm = _base_arm(
                model,
                f"prebelief_{source}_{info}_{model['key']}",
                "pre_decision_belief_mechanism",
            )
            arm.update(_live_comm_fields(stage_context, "none"))
            arm["belief_timing"] = "pre"
            arm["belief_information"] = info
            arm["primary_outcomes"] = _belief_outcomes()
            if source == "surv":
                arm["preregistered_comparisons"] = [
                    {
                        "treatment": arm["arm_id"],
                        "control": f"prebelief_comm_{info}_{model['key']}",
                    }
                ]
            arms.append(arm)
    return arms


def _cross_task_arms(model: dict[str, Any]) -> list[dict[str, Any]]:
    arms = []
    for task_prefix, task in [("coord", "coordination"), ("bet", "individual_bet")]:
        for source, transform in [("baseline", "direct"), ("surveillance", "coded")]:
            arm = _base_arm(
                model,
                f"{task_prefix}_with_{source}_messages_{model['key']}",
                "cross_task_decomposition",
            )
            arm.update(
                _message_bank_fields(
                    "CLEAN_RUN/message_banks/baseline_surveillance_matched.parquet",
                    transform,
                    task,
                    "none",
                    "none",
                    "pre",
                    "messages_included",
                )
            )
            arm["primary_outcomes"] = ["join_fraction_valid"]
            if source == "surveillance":
                arm["preregistered_comparisons"] = [
                    {
                        "treatment": arm["arm_id"],
                        "control": f"{task_prefix}_with_baseline_messages_{model['key']}",
                    }
                ]
            arms.append(arm)
    return arms


def _cross_model_rotation_arms() -> list[dict[str, Any]]:
    pairs = [
        ("llama4", "qwen30"),
        ("qwen30", "llama4"),
        ("mistral", "deepseek"),
        ("deepseek", "mistral"),
    ]
    lookup = {item["key"]: item for item in MODEL_ROSTER}
    arms = []
    for writer_key, reader_key in pairs:
        writer = lookup[writer_key]
        reader = lookup[reader_key]
        baseline = _base_arm(
            reader,
            f"xmodel_{writer_key}_writes_{reader_key}_reads_baseline",
            "cross_model_generalization",
        )
        baseline.update(_live_comm_fields("none", "none"))
        baseline["writer_model"] = writer["model"]
        baseline["reader_model"] = reader["model"]
        surveillance = _base_arm(
            reader,
            f"xmodel_{writer_key}_writes_{reader_key}_reads_surv",
            "cross_model_generalization",
        )
        surveillance.update(_live_comm_fields("surveillance_full", "none"))
        surveillance["writer_model"] = writer["model"]
        surveillance["reader_model"] = reader["model"]
        surveillance["preregistered_comparisons"] = [
            {"treatment": surveillance["arm_id"], "control": baseline["arm_id"]}
        ]
        arms.extend([baseline, surveillance])
    return arms


def _live_comm_fields(stage_context: str, decision_context: str) -> dict[str, Any]:
    return {
        "role": "writer_and_reader",
        "message_source": "live",
        "message_transform": "none",
        "decision_task": "coordination",
        "message_stage_context": stage_context,
        "decision_context": decision_context,
        "belief_timing": "none",
        "belief_information": "none",
        "primary_outcomes": ["join_fraction_valid", "message_sent"],
    }


def _message_bank_fields(
    path: str,
    transform: str,
    task: str,
    stage_context: str,
    decision_context: str,
    belief_timing: str,
    belief_information: str,
) -> dict[str, Any]:
    return {
        "role": "reader_with_fixed_messages",
        "message_source": "message_bank",
        "message_bank_path": path,
        "message_transform": transform,
        "decision_task": task,
        "message_stage_context": stage_context,
        "decision_context": decision_context,
        "belief_timing": belief_timing,
        "belief_information": belief_information,
        "primary_outcomes": _belief_outcomes() if belief_timing == "pre" else ["join_fraction_valid"],
    }


def _belief_outcomes() -> list[str]:
    return [
        "join_fraction_valid",
        "belief_pre_success",
        "belief_pre_join_share",
        "belief_pre_shared_understanding",
        "belief_pre_others_expect_join",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CLEAN_RUN/plans/main.yaml")
    parser.add_argument("--output", default=str(CLEAN_RUN_ROOT / "plans" / "main.yaml"))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_main_manifest()
    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, width=120)

    loaded = load_manifest(output)
    errors = validate_manifest(loaded)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"wrote {output} ({len(loaded.arms)} arms)")


if __name__ == "__main__":
    main()
