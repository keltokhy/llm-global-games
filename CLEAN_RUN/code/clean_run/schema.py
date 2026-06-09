"""Canonical clean-run parquet schemas and row normalization."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import pandas as pd


PERIOD_COLUMNS = [
    "run_id",
    "arm_id",
    "model",
    "country",
    "period",
    "seed",
    "theta",
    "z_public",
    "sigma",
    "benefit",
    "cost",
    "theta_star",
    "n_agents",
    "n_valid",
    "n_join",
    "join_fraction_valid",
    "coup_success",
    "theoretical_attack",
    "message_stage_context",
    "decision_context",
    "decision_task",
    "message_source_arm",
    "message_transform",
    "belief_timing",
    "belief_information",
    "api_error_rate",
    "unparseable_rate",
]


AGENT_COLUMNS = [
    "run_id",
    "arm_id",
    "model",
    "country",
    "period",
    "agent_id",
    "theta",
    "signal",
    "z_score",
    "briefing_text",
    "message_sent",
    "messages_received",
    "decision",
    "join",
    "reasoning",
    "belief_pre_success",
    "belief_pre_join_share",
    "belief_pre_shared_understanding",
    "belief_pre_others_expect_join",
    "belief_post_success",
    "belief_post_join_share",
    "belief_post_shared_understanding",
    "belief_post_others_expect_join",
    "punishment_risk",
    "api_error",
    "parse_error",
    "prompt_hash",
    "response_hash",
]


MESSAGE_BANK_COLUMNS = [
    "message_id",
    "source_arm_id",
    "country",
    "period",
    "sender_agent_id",
    "theta",
    "sender_signal",
    "sender_z_score",
    "original_message",
    "direct_message",
    "coded_message",
    "factual_summary",
    "valence",
    "sentiment_score",
    "hedge_density",
    "specificity_score",
    "verbosity_tokens",
    "syntactic_complexity",
    "embedding_vector_id",
    "first_order_similarity",
    "directness_score",
    "codedness_score",
    "risk_salience_score",
    "factual_equivalence_pass",
    "style_balance_pass",
    "qc_notes",
]


def sha256_text(value: Any) -> str:
    payload = "" if value is None else str(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_dumps(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _build_prompt_hash(agent_row: dict[str, Any], arm: dict[str, Any]) -> str:
    payload = {
        "briefing_text": agent_row.get("briefing_text", ""),
        "messages_received": agent_row.get("messages_received", []),
        "decision_task": arm["decision_task"],
        "decision_context": arm["decision_context"],
        "message_stage_context": arm["message_stage_context"],
        "belief_timing": arm["belief_timing"],
        "belief_information": arm["belief_information"],
    }
    return sha256_text(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def period_rows(results: Iterable[Any], arm: dict[str, Any], run_id: str) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "run_id": run_id,
                "arm_id": arm["arm_id"],
                "model": arm["model"],
                "country": result.country,
                "period": result.period,
                "seed": arm["seed"],
                "theta": result.theta,
                "z_public": result.z,
                "sigma": arm.get("sigma", 0.3),
                "benefit": result.benefit,
                "cost": arm.get("cost", 1.0),
                "theta_star": result.theta_star,
                "n_agents": result.n_agents,
                "n_valid": result.n_valid,
                "n_join": result.n_join,
                "join_fraction_valid": result.join_fraction_valid,
                "coup_success": result.coup_success,
                "theoretical_attack": result.theoretical_attack,
                "message_stage_context": arm["message_stage_context"],
                "decision_context": arm["decision_context"],
                "decision_task": arm["decision_task"],
                "message_source_arm": arm["message_source"],
                "message_transform": arm["message_transform"],
                "belief_timing": arm["belief_timing"],
                "belief_information": arm["belief_information"],
                "api_error_rate": result.api_error_rate,
                "unparseable_rate": result.unparseable_rate,
            }
        )
    return pd.DataFrame(rows, columns=PERIOD_COLUMNS)


def agent_rows(results: Iterable[Any], arm: dict[str, Any], run_id: str) -> pd.DataFrame:
    rows = []
    for result in results:
        for agent in result.agents:
            response = agent.get("reasoning", "")
            decision = agent.get("decision")
            rows.append(
                {
                    "run_id": run_id,
                    "arm_id": arm["arm_id"],
                    "model": agent.get("model") or arm["model"],
                    "country": result.country,
                    "period": result.period,
                    "agent_id": agent.get("id"),
                    "theta": result.theta,
                    "signal": agent.get("signal"),
                    "z_score": agent.get("z_score"),
                    "briefing_text": agent.get("briefing_text", ""),
                    "message_sent": agent.get("message_sent", ""),
                    "messages_received": _json_dumps(agent.get("messages_received", [])),
                    "decision": decision,
                    "join": True if decision == "JOIN" else False if decision == "STAY" else pd.NA,
                    "reasoning": response,
                    "belief_pre_success": agent.get("belief_pre"),
                    "belief_pre_join_share": agent.get("second_order_belief_pre"),
                    "belief_pre_shared_understanding": agent.get("shared_understanding_belief_pre"),
                    "belief_pre_others_expect_join": agent.get("others_expect_join_belief_pre"),
                    "belief_post_success": agent.get("belief"),
                    "belief_post_join_share": agent.get("second_order_belief"),
                    "belief_post_shared_understanding": agent.get("shared_understanding_belief"),
                    "belief_post_others_expect_join": agent.get("others_expect_join_belief"),
                    "punishment_risk": agent.get("punishment_risk"),
                    "api_error": bool(agent.get("api_error", False)),
                    "parse_error": decision not in ("JOIN", "STAY"),
                    "prompt_hash": _build_prompt_hash(agent, arm),
                    "response_hash": sha256_text(response),
                }
            )
    return pd.DataFrame(rows, columns=AGENT_COLUMNS)


def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [column for column in required if column not in df.columns]
