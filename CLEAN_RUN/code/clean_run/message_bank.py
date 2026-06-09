"""Message-bank utilities for direct/coded replay arms."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_based_simulation.briefing import BriefingGenerator
from agent_based_simulation.experiment import Agent
from agent_based_simulation.runtime import deterministic_hash, parse_float_list

from .config import CLEAN_RUN_ROOT, require_valid_manifest
from .grids import task_cells
from .schema import MESSAGE_BANK_COLUMNS


DIRECT_TERMS = {
    "regime",
    "uprising",
    "protest",
    "security",
    "military",
    "join",
    "streets",
    "fall",
    "weak",
    "strong",
}
CODED_TERMS = {
    "weather",
    "market",
    "noise",
    "lights",
    "doors",
    "season",
    "signals",
    "neighbors",
    "quiet",
}
HEDGE_TERMS = {
    "may",
    "might",
    "could",
    "seems",
    "appears",
    "maybe",
    "possibly",
    "unclear",
    "perhaps",
}
RISK_TERMS = {
    "punish",
    "punishment",
    "danger",
    "risk",
    "monitored",
    "surveillance",
    "security",
    "consequences",
    "trap",
}
URGENCY_TERMS = {
    "now",
    "urgent",
    "immediately",
    "today",
    "tonight",
    "moment",
    "breaking",
}
POSITIVE_TERMS = {"weak", "fracturing", "fall", "dissent", "momentum", "opening"}
NEGATIVE_TERMS = {"strong", "stable", "loyal", "quiet", "orderly", "control"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", str(text).lower())


def token_count(text: str) -> int:
    return len(tokenize(text))


def density(text: str, vocabulary: set[str]) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in vocabulary) / len(tokens)


def sentiment_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    positive = sum(1 for token in tokens if token in POSITIVE_TERMS)
    negative = sum(1 for token in tokens if token in NEGATIVE_TERMS)
    return (positive - negative) / len(tokens)


def specificity_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    numbers = len(re.findall(r"\b\d+(?:\.\d+)?\b", str(text)))
    named = len(re.findall(r"\b[A-Z][a-z]{3,}\b", str(text)))
    return min(1.0, (numbers + named + len(set(tokens)) / 8.0) / max(1.0, len(tokens) / 4.0))


def syntactic_complexity(text: str) -> float:
    clauses = len(re.findall(r"[,;:]", str(text))) + len(re.findall(r"\b(and|but|while|because|though)\b", str(text).lower()))
    return clauses / max(1, token_count(text))


def score_message(text: str) -> dict[str, float | int]:
    return {
        "sentiment_score": sentiment_score(text),
        "hedge_density": density(text, HEDGE_TERMS),
        "specificity_score": specificity_score(text),
        "verbosity_tokens": token_count(text),
        "syntactic_complexity": syntactic_complexity(text),
        "directness_score": density(text, DIRECT_TERMS),
        "codedness_score": density(text, CODED_TERMS),
        "risk_salience_score": density(text, RISK_TERMS),
        "urgency_score": density(text, URGENCY_TERMS),
    }


def validate_bank(df: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in MESSAGE_BANK_COLUMNS if column not in df.columns]
    result: dict[str, Any] = {
        "rows": int(len(df)),
        "missing_columns": missing,
        "factual_equivalence_fail_rate": None,
        "style_balance_fail_rate": None,
        "accepted_rows": 0,
        "pass": False,
    }
    if missing or df.empty:
        return result

    factual = df["factual_equivalence_pass"].fillna(False).astype(bool)
    style = df["style_balance_pass"].fillna(False).astype(bool)
    result["factual_equivalence_fail_rate"] = float((~factual).mean())
    result["style_balance_fail_rate"] = float((~style).mean())

    original_replay = _is_original_replay(df)
    if original_replay:
        acceptance = factual & style & (df["first_order_similarity"].astype(float) >= 0.88)
    else:
        acceptance = _accepted_mask(df, factual=factual, style=style)
    result["accepted_rows"] = int(acceptance.sum())
    result["pass"] = bool(
        result["rows"] > 0
        and result["factual_equivalence_fail_rate"] <= 0.05
        and result["accepted_rows"] == result["rows"]
    )
    return result


def _is_original_replay(df: pd.DataFrame) -> bool:
    return bool(
        (
            df["original_message"].fillna("").eq(df["direct_message"].fillna(""))
            & df["original_message"].fillna("").eq(df["coded_message"].fillna(""))
        ).all()
    )


def _accepted_mask(
    df: pd.DataFrame,
    *,
    factual: pd.Series | None = None,
    style: pd.Series | None = None,
) -> pd.Series:
    """Return the row-level direct/coded acceptance mask."""

    if df.empty:
        return pd.Series([], dtype=bool)
    factual = factual if factual is not None else df["factual_equivalence_pass"].fillna(False).astype(bool)
    style = style if style is not None else df["style_balance_pass"].fillna(False).astype(bool)
    direct_scores = pd.DataFrame([score_message(text) for text in df["direct_message"].fillna("")], index=df.index)
    coded_scores = pd.DataFrame([score_message(text) for text in df["coded_message"].fillna("")], index=df.index)
    direct_is_explicit = direct_scores["directness_score"] >= direct_scores["codedness_score"]
    coded_is_coded = coded_scores["codedness_score"] > coded_scores["directness_score"]
    return (
        factual
        & style
        & (df["first_order_similarity"].astype(float) >= 0.88)
        & direct_is_explicit
        & coded_is_coded
    )


def accepted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that pass the direct/coded row-level QC gate."""

    if _is_original_replay(df):
        mask = (
            df["factual_equivalence_pass"].fillna(False).astype(bool)
            & df["style_balance_pass"].fillna(False).astype(bool)
            & (df["first_order_similarity"].astype(float) >= 0.88)
        )
    else:
        mask = _accepted_mask(df)
    return df.loc[mask].copy()


BALANCE_DIMENSIONS = [
    "sentiment_score",
    "hedge_density",
    "specificity_score",
    "risk_salience_score",
    "urgency_score",
    "verbosity_tokens",
    "syntactic_complexity",
    "directness_score",
    "codedness_score",
]


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute direct-vs-coded balance diagnostics from message text."""

    if df.empty:
        return pd.DataFrame(
            columns=[
                "dimension",
                "direct_mean",
                "coded_mean",
                "difference",
                "pooled_sd",
                "standardized_difference",
                "passes_0_10sd_rule",
            ]
        )

    direct_scores = pd.DataFrame([score_message(text) for text in df["direct_message"].fillna("")])
    coded_scores = pd.DataFrame([score_message(text) for text in df["coded_message"].fillna("")])
    rows = []
    for dimension in BALANCE_DIMENSIONS:
        direct = direct_scores[dimension].astype(float)
        coded = coded_scores[dimension].astype(float)
        direct_mean = float(direct.mean())
        coded_mean = float(coded.mean())
        pooled_sd = float((((direct.var(ddof=1) + coded.var(ddof=1)) / 2) ** 0.5))
        diff = coded_mean - direct_mean
        standardized = float(diff / pooled_sd) if pooled_sd > 0 else 0.0
        rows.append(
            {
                "dimension": dimension,
                "direct_mean": direct_mean,
                "coded_mean": coded_mean,
                "difference": diff,
                "pooled_sd": pooled_sd,
                "standardized_difference": standardized,
                "passes_0_10sd_rule": abs(standardized) <= 0.10,
            }
        )
    return pd.DataFrame(rows)


def manual_audit_sample(
    df: pd.DataFrame,
    *,
    n: int = 50,
    seed: int = 5150,
) -> pd.DataFrame:
    """Draw a reproducible manual-audit sample from a message bank."""

    cols = [
        "message_id",
        "source_arm_id",
        "country",
        "period",
        "sender_agent_id",
        "theta",
        "sender_z_score",
        "original_message",
        "direct_message",
        "coded_message",
        "first_order_similarity",
        "factual_equivalence_pass",
        "style_balance_pass",
        "qc_notes",
    ]
    available = [col for col in cols if col in df.columns]
    sample_n = min(n, len(df))
    if sample_n == 0:
        return pd.DataFrame(columns=cols + _manual_audit_columns())
    sample = df.sample(n=sample_n, random_state=seed)[available].copy()
    for col in _manual_audit_columns():
        sample[col] = ""
    return sample


def _manual_audit_columns() -> list[str]:
    return [
        "manual_same_regime_strength",
        "manual_coded_less_direct",
        "manual_extra_punishment_risk",
        "manual_confusing_or_nonsensical",
        "manual_length_hedge_specificity_sentiment_urgency_balanced",
        "manual_notes",
    ]


def summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return compact summary statistics for a message bank."""

    if df.empty:
        return {"rows": 0, "numeric": {}, "counts": {}}

    numeric_columns = [
        "theta",
        "sender_signal",
        "sender_z_score",
        "sentiment_score",
        "hedge_density",
        "specificity_score",
        "verbosity_tokens",
        "syntactic_complexity",
        "first_order_similarity",
        "directness_score",
        "codedness_score",
        "risk_salience_score",
    ]
    numeric = {}
    for column in numeric_columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        numeric[column] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
        }

    counts = {}
    for column in ["source_arm_id", "valence", "factual_equivalence_pass", "style_balance_pass"]:
        if column in df.columns:
            counts[column] = {
                str(key): int(value)
                for key, value in df[column].fillna("<missing>").value_counts(dropna=False).to_dict().items()
            }

    return {
        "rows": int(len(df)),
        "numeric": numeric,
        "counts": counts,
        "qc": validate_bank(df),
    }


def build_original_bank_from_log(
    log_path: str | Path,
    *,
    source_arm_id: str,
    theta_min: float | None = None,
    theta_max: float | None = None,
) -> pd.DataFrame:
    """Build an original-message replay bank from one real communication log."""

    entries = _load_log_entries(log_path)
    rows = []
    for entry in entries:
        theta = float(entry["theta"])
        if theta_min is not None and theta < theta_min:
            continue
        if theta_max is not None and theta > theta_max:
            continue
        for agent in entry.get("agents", []):
            message = str(agent.get("message_sent") or "").strip()
            if not message:
                continue
            scores = score_message(message)
            rows.append(
                _message_bank_row(
                    message_id=_message_id(source_arm_id, entry, agent),
                    source_arm_id=source_arm_id,
                    entry=entry,
                    agent=agent,
                    original_message=message,
                    direct_message=message,
                    coded_message=message,
                    first_order_similarity=1.0,
                    factual_equivalence_pass=True,
                    style_balance_pass=True,
                    qc_notes="original replay bank from real communication log",
                    score_values=scores,
                )
            )
    return pd.DataFrame(rows, columns=MESSAGE_BANK_COLUMNS)


def build_paired_bank_from_logs(
    direct_log_path: str | Path,
    coded_log_path: str | Path,
    *,
    direct_source_arm_id: str,
    coded_source_arm_id: str,
    theta_min: float | None = None,
    theta_max: float | None = None,
) -> pd.DataFrame:
    """Build a matched real-message pair bank from two communication logs.

    This does not rewrite messages. It pairs the recorded baseline message with
    the recorded surveilled message for the same country, period, and sender.
    QC fields make clear whether the result passes the final direct/coded gate.
    """

    direct_entries = _entries_by_cell(_load_log_entries(direct_log_path))
    coded_entries = _entries_by_cell(_load_log_entries(coded_log_path))
    rows = []
    for key in sorted(set(direct_entries) & set(coded_entries)):
        direct_entry = direct_entries[key]
        coded_entry = coded_entries[key]
        theta = float(direct_entry["theta"])
        if theta_min is not None and theta < theta_min:
            continue
        if theta_max is not None and theta > theta_max:
            continue
        direct_agents = _agents_by_id(direct_entry)
        coded_agents = _agents_by_id(coded_entry)
        for agent_id in sorted(set(direct_agents) & set(coded_agents)):
            direct_agent = direct_agents[agent_id]
            coded_agent = coded_agents[agent_id]
            direct_message = str(direct_agent.get("message_sent") or "").strip()
            coded_message = str(coded_agent.get("message_sent") or "").strip()
            if not direct_message or not coded_message:
                continue

            direct_scores = score_message(direct_message)
            coded_scores = score_message(coded_message)
            similarity = _text_similarity(direct_message, coded_message)
            factual_pass = similarity >= 0.88
            style_pass = _style_balance_pass(direct_scores, coded_scores)
            row_scores = {
                "sentiment_score": coded_scores["sentiment_score"],
                "hedge_density": coded_scores["hedge_density"],
                "specificity_score": coded_scores["specificity_score"],
                "verbosity_tokens": coded_scores["verbosity_tokens"],
                "syntactic_complexity": coded_scores["syntactic_complexity"],
                "directness_score": direct_scores["directness_score"],
                "codedness_score": coded_scores["codedness_score"],
                "risk_salience_score": coded_scores["risk_salience_score"],
            }
            qc_notes = (
                "paired real baseline/surveillance source messages; "
                "not an LLM-rewritten factual-equivalence bank"
            )
            rows.append(
                _message_bank_row(
                    message_id=_paired_message_id(direct_source_arm_id, coded_source_arm_id, direct_entry, direct_agent),
                    source_arm_id=f"{direct_source_arm_id}|{coded_source_arm_id}",
                    entry=direct_entry,
                    agent=direct_agent,
                    original_message=direct_message,
                    direct_message=direct_message,
                    coded_message=coded_message,
                    first_order_similarity=similarity,
                    factual_equivalence_pass=factual_pass,
                    style_balance_pass=style_pass,
                    qc_notes=qc_notes,
                    score_values=row_scores,
                )
            )
    return pd.DataFrame(rows, columns=MESSAGE_BANK_COLUMNS)


def build_source_bank_from_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    arm_id: str | None = None,
    claim: str = "direct_coded_mechanism",
    max_cells: int | None = None,
    max_agents: int | None = None,
) -> pd.DataFrame:
    """Build source rows from the exact task cells a manifest arm will use.

    The source message is the rendered private briefing for the sender-agent in
    that task cell. This uses the real project signal/briefing pipeline and is
    meant as input to the LLM direct/coded rewrite step.
    """

    manifest = require_valid_manifest(manifest_path)
    arm = _select_manifest_arm(manifest.arms, arm_id=arm_id, claim=claim)
    briefing_gen = BriefingGenerator(**_briefing_kwargs_from_arm(arm))
    cells = task_cells(arm)
    if max_cells is not None:
        cells = cells[:max_cells]
    n_agents = int(arm["n_agents"])
    if max_agents is not None:
        n_agents = min(n_agents, int(max_agents))

    rows = []
    for cell in cells:
        country = int(cell["country"])
        period = int(cell["period"])
        theta = float(cell["theta"])
        z_public = float(cell["z_public"])
        sigma = float(arm.get("sigma", 0.3))
        rng = np.random.default_rng(deterministic_hash((country, period, "signals")) % 2**32)
        entry = {
            "country": country,
            "period": period,
            "theta": theta,
            "theta_star": float(arm.get("theta_star", 0.5)),
        }
        for agent_id in range(n_agents):
            agent = Agent(agent_id=agent_id)
            signal = theta + rng.normal(0, sigma)
            z_score = (signal - z_public) / sigma
            briefing = briefing_gen.generate(z_score, agent_id, period)
            source = briefing.render()
            scores = score_message(source)
            rows.append(
                _message_bank_row(
                    message_id=_manifest_source_message_id(str(arm["arm_id"]), country, period, agent_id),
                    source_arm_id=str(arm["arm_id"]),
                    entry=entry,
                    agent={"id": agent.agent_id, "signal": signal, "z_score": z_score},
                    original_message=source,
                    direct_message=source,
                    coded_message=source,
                    first_order_similarity=1.0,
                    factual_equivalence_pass=True,
                    style_balance_pass=True,
                    qc_notes="manifest-derived private briefing source row",
                    score_values=scores,
                )
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=MESSAGE_BANK_COLUMNS)
    df.to_parquet(output, index=False)
    return df


def derive_original_replay_bank(
    source_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str = "direct_message",
) -> pd.DataFrame:
    """Create an original-replay bank from one text column of a passing bank."""

    source = load_bank(source_path)
    if text_column not in source.columns:
        raise ValueError(f"Missing text column: {text_column}")
    rows = []
    for _, row in source.iterrows():
        text = str(row[text_column])
        scores = score_message(text)
        entry = {
            "country": row["country"],
            "period": row["period"],
            "theta": row["theta"],
        }
        agent = {
            "id": row["sender_agent_id"],
            "signal": row["sender_signal"],
            "z_score": row["sender_z_score"],
        }
        rows.append(
            _message_bank_row(
                message_id=str(row["message_id"]),
                source_arm_id=str(row["source_arm_id"]),
                entry=entry,
                agent=agent,
                original_message=text,
                direct_message=text,
                coded_message=text,
                first_order_similarity=1.0,
                factual_equivalence_pass=True,
                style_balance_pass=True,
                qc_notes=f"original replay derived from {text_column}",
                score_values=scores,
            )
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=MESSAGE_BANK_COLUMNS)
    df.to_parquet(output, index=False)
    return df


def combine_bank_batches(
    batch_dir: str | Path,
    output_path: str | Path,
    *,
    pattern: str = "batch_*.parquet",
    require_pass: bool = True,
) -> pd.DataFrame:
    """Combine batch parquet files into one bank, optionally requiring QC pass."""

    root = Path(batch_dir)
    paths = sorted(root.glob(pattern))
    if not paths:
        raise RuntimeError(f"Dependency needed: no batch files matching {root / pattern}")
    frames = []
    failed = []
    for path in paths:
        df = load_bank(path)
        qc = validate_bank(df)
        if require_pass and not qc["pass"]:
            failed.append({"path": str(path), "qc": qc})
        frames.append(df)
    if failed:
        raise RuntimeError(f"Batch QC failed; not combining: {failed[:5]}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(["message_id"], keep="last")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    if require_pass:
        qc = validate_bank(combined)
        if not qc["pass"]:
            raise RuntimeError(f"Combined bank failed QC: {qc}")
    return combined


def _select_manifest_arm(
    arms: list[dict[str, Any]],
    *,
    arm_id: str | None,
    claim: str,
) -> dict[str, Any]:
    if arm_id:
        matches = [arm for arm in arms if arm["arm_id"] == arm_id]
    else:
        matches = [arm for arm in arms if arm["claim"] == claim]
    if not matches:
        raise RuntimeError(f"Dependency needed: no manifest arm for arm_id={arm_id!r} claim={claim!r}")
    return matches[0]


def _briefing_kwargs_from_arm(arm: dict[str, Any]) -> dict[str, Any]:
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


def _manifest_source_message_id(arm_id: str, country: int, period: int, agent_id: int) -> str:
    payload = f"{arm_id}:manifest-source:{country}:{period}:{agent_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _load_log_entries(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list log payload in {path}")
    return payload


def _entries_by_cell(entries: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(entry["country"]), int(entry["period"])): entry
        for entry in entries
    }


def _agents_by_id(entry: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(agent["id"]): agent
        for agent in entry.get("agents", [])
        if "id" in agent
    }


def _message_id(source_arm_id: str, entry: dict[str, Any], agent: dict[str, Any]) -> str:
    payload = f"{source_arm_id}:{entry['country']}:{entry['period']}:{agent['id']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _paired_message_id(
    direct_source_arm_id: str,
    coded_source_arm_id: str,
    entry: dict[str, Any],
    agent: dict[str, Any],
) -> str:
    payload = f"{direct_source_arm_id}:{coded_source_arm_id}:{entry['country']}:{entry['period']}:{agent['id']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _message_bank_row(
    *,
    message_id: str,
    source_arm_id: str,
    entry: dict[str, Any],
    agent: dict[str, Any],
    original_message: str,
    direct_message: str,
    coded_message: str,
    first_order_similarity: float,
    factual_equivalence_pass: bool,
    style_balance_pass: bool,
    qc_notes: str,
    score_values: dict[str, Any],
) -> dict[str, Any]:
    return {
        "message_id": message_id,
        "source_arm_id": source_arm_id,
        "country": int(entry["country"]),
        "period": int(entry["period"]),
        "sender_agent_id": int(agent["id"]),
        "theta": float(entry["theta"]),
        "sender_signal": _safe_float(agent.get("signal")),
        "sender_z_score": _safe_float(agent.get("z_score")),
        "original_message": original_message,
        "direct_message": direct_message,
        "coded_message": coded_message,
        "factual_summary": pd.NA,
        "valence": _theta_valence(entry),
        "sentiment_score": score_values["sentiment_score"],
        "hedge_density": score_values["hedge_density"],
        "specificity_score": score_values["specificity_score"],
        "verbosity_tokens": int(score_values["verbosity_tokens"]),
        "syntactic_complexity": score_values["syntactic_complexity"],
        "embedding_vector_id": pd.NA,
        "first_order_similarity": float(first_order_similarity),
        "directness_score": score_values["directness_score"],
        "codedness_score": score_values["codedness_score"],
        "risk_salience_score": score_values["risk_salience_score"],
        "factual_equivalence_pass": bool(factual_equivalence_pass),
        "style_balance_pass": bool(style_balance_pass),
        "qc_notes": qc_notes,
    }


def _safe_float(value: Any) -> float | Any:
    if value is None or pd.isna(value):
        return pd.NA
    return float(value)


def _theta_valence(entry: dict[str, Any]) -> str:
    theta = float(entry["theta"])
    theta_star = float(entry.get("theta_star", 0.5))
    if theta < theta_star:
        return "weakness_indicating"
    if theta > theta_star:
        return "strength_indicating"
    return "marginal"


def _text_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens and not right_tokens:
        return 1.0
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    sequence = SequenceMatcher(None, left, right).ratio()
    return float((jaccard + sequence) / 2.0)


def _style_balance_pass(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_tokens = max(1, int(left["verbosity_tokens"]))
    right_tokens = max(1, int(right["verbosity_tokens"]))
    token_ratio = abs(left_tokens - right_tokens) / max(left_tokens, right_tokens)
    return bool(
        token_ratio <= 0.20
        and abs(float(left["sentiment_score"]) - float(right["sentiment_score"])) <= 0.10
        and abs(float(left["hedge_density"]) - float(right["hedge_density"])) <= 0.10
        and abs(float(left["risk_salience_score"]) - float(right["risk_salience_score"])) <= 0.10
    )


def load_bank(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def promote_valid_bank(
    candidate_path: str | Path,
    target_path: str | Path,
    *,
    backup_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically promote a candidate bank only if it passes QC."""

    candidate = Path(candidate_path).resolve()
    target = Path(target_path).resolve()
    if candidate == target:
        raise RuntimeError("Candidate and target paths are identical; nothing to promote.")
    if not candidate.exists():
        raise RuntimeError(f"Dependency needed: {candidate}")

    qc = validate_bank(load_bank(candidate))
    if not qc["pass"]:
        raise RuntimeError(f"Candidate bank failed QC and was not promoted: {qc}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if backup_path and target.exists():
        backup = Path(backup_path).resolve()
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)

    tmp_target = target.with_name(f"{target.name}.tmp")
    try:
        shutil.copy2(candidate, tmp_target)
        os.replace(tmp_target, target)
    finally:
        if tmp_target.exists():
            tmp_target.unlink()
    return qc


REWRITE_SYSTEM_PROMPT = (
    "You rewrite experimental citizen messages for a global-games study. "
    "Return strict JSON only. Create two natural 1-2 sentence messages with "
    "the same first-order facts, valence, uncertainty, urgency, and risk "
    "content. The direct version should be plain and explicit. The coded "
    "version should be indirect and deniable. Keep the two versions within "
    "20 percent of each other in length; ideally they should differ by no more "
    "than five words. Use the same number of sentences and roughly the same "
    "number of concrete claims. Keep hedge density, specificity, sentiment, "
    "urgency, and risk language similar. The direct version "
    "must use plain political terms such as regime, security, weak, protest, "
    "streets, join, or fall when they fit the source. The coded version must "
    "avoid those direct political terms and use at least two stable code words "
    "from this lexicon when they fit naturally: weather, market, lights, doors, "
    "season, signals, neighbors, quiet. Do not add facts, punishment threats, "
    "surveillance warnings, or new uncertainty."
)


async def rewrite_direct_coded_bank(
    source_path: str | Path,
    output_path: str | Path,
    *,
    model: str,
    api_base_url: str = "https://openrouter.ai/api/v1",
    provider: dict[str, Any] | None = None,
    max_rows: int | None = None,
    max_concurrent: int = 10,
    temperature: float = 0.2,
    allow_overwrite: bool = False,
    retry_failed: int = 0,
    request_timeout: float = 60.0,
) -> pd.DataFrame:
    """Rewrite real source messages into a candidate direct/coded bank."""

    source_file = Path(source_path).resolve()
    output = Path(output_path).resolve()
    replacing_source = source_file == output
    if replacing_source and not allow_overwrite:
        raise RuntimeError(
            "Refusing to overwrite the source message bank. "
            "Write to a new path, or pass --allow-overwrite for an explicit atomic replacement."
        )

    if "openrouter.ai" in api_base_url and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Dependency needed: OPENROUTER_API_KEY")

    from openai import AsyncOpenAI

    source = load_bank(source_path)
    if max_rows is not None:
        source = source.head(max_rows).copy()

    client = AsyncOpenAI(
        base_url=api_base_url,
        api_key=os.environ.get("OPENROUTER_API_KEY", "") or "not-needed",
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rewrite_one(row: pd.Series, *, attempt: int = 1) -> dict[str, Any]:
        user_prompt = (
            "Source message:\n"
            f"{row['original_message']}\n\n"
            f"Rewrite attempt: {attempt}. If this is greater than 1, the prior "
            "candidate failed automated row-level balance/QC, so be especially "
            "strict about matched length, matched risk/urgency, explicit direct "
            "language, and deniable coded language.\n\n"
            "Return JSON with keys: direct_message, coded_message, factual_summary, "
            "factual_equivalence_score, style_balance_score, qc_notes."
        )
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 900,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if provider:
            request["extra_body"] = {"provider": provider}
        async with semaphore:
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**request),
                    timeout=request_timeout,
                )
                content = response.choices[0].message.content or ""
                payload = _parse_json_object(content)
            except Exception as exc:
                return _failed_rewrite_row(row, f"rewrite parse/API failure attempt={attempt}: {exc}")

        direct_message = str(payload.get("direct_message") or "").strip()
        coded_message = str(payload.get("coded_message") or "").strip()
        if not direct_message or not coded_message:
            return _failed_rewrite_row(row, f"rewrite returned empty message attempt={attempt}")

        direct_scores = score_message(direct_message)
        coded_scores = score_message(coded_message)
        factual_score = float(payload.get("factual_equivalence_score", _text_similarity(direct_message, coded_message)))
        style_score = float(payload.get("style_balance_score", 0.0))
        row_scores = {
            "sentiment_score": coded_scores["sentiment_score"],
            "hedge_density": coded_scores["hedge_density"],
            "specificity_score": coded_scores["specificity_score"],
            "verbosity_tokens": coded_scores["verbosity_tokens"],
            "syntactic_complexity": coded_scores["syntactic_complexity"],
            "directness_score": direct_scores["directness_score"],
            "codedness_score": coded_scores["codedness_score"],
            "risk_salience_score": coded_scores["risk_salience_score"],
        }
        entry = {
            "country": row["country"],
            "period": row["period"],
            "theta": row["theta"],
        }
        agent = {
            "id": row["sender_agent_id"],
            "signal": row["sender_signal"],
            "z_score": row["sender_z_score"],
        }
        rewritten_row = _message_bank_row(
            message_id=str(row["message_id"]),
            source_arm_id=str(row["source_arm_id"]),
            entry=entry,
            agent=agent,
            original_message=str(row["original_message"]),
            direct_message=direct_message,
            coded_message=coded_message,
            first_order_similarity=factual_score,
            factual_equivalence_pass=factual_score >= 0.88,
            style_balance_pass=style_score >= 0.90 and _style_balance_pass(direct_scores, coded_scores),
            qc_notes=str(payload.get("qc_notes") or "LLM rewrite candidate"),
            score_values=row_scores,
        )
        summary = payload.get("factual_summary")
        rewritten_row["factual_summary"] = str(summary).strip() if summary else pd.NA
        rewritten_row["embedding_vector_id"] = pd.NA
        return rewritten_row

    accepted_by_id: dict[str, dict[str, Any]] = {}
    latest_by_id: dict[str, dict[str, Any]] = {}
    remaining = source.copy()
    source_order = {str(row.message_id): i for i, row in enumerate(source.itertuples(index=False))}
    try:
        for attempt in range(1, int(retry_failed) + 2):
            rows = await asyncio.gather(*[rewrite_one(row, attempt=attempt) for _, row in remaining.iterrows()])
            attempt_df = pd.DataFrame(rows, columns=MESSAGE_BANK_COLUMNS)
            for row in rows:
                latest_by_id[str(row["message_id"])] = row
            passing = accepted_rows(attempt_df)
            for row in passing.to_dict("records"):
                accepted_by_id[str(row["message_id"])] = row

            failed_ids = [
                str(message_id)
                for message_id in attempt_df["message_id"].astype(str).tolist()
                if str(message_id) not in accepted_by_id
            ]
            if not failed_ids:
                break
            remaining = source[source["message_id"].astype(str).isin(failed_ids)].copy()
    finally:
        await client.close()

    combined = []
    for message_id in source["message_id"].astype(str).tolist():
        combined.append(accepted_by_id.get(message_id) or latest_by_id[message_id])
    combined.sort(key=lambda row: source_order[str(row["message_id"])])
    rewritten = pd.DataFrame(combined, columns=MESSAGE_BANK_COLUMNS)
    qc = validate_bank(rewritten)
    if replacing_source and not qc["pass"]:
        raise RuntimeError(f"Rewritten candidate failed QC and source bank was not replaced: {qc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp")
    try:
        rewritten.to_parquet(tmp_output, index=False)
        os.replace(tmp_output, output)
    finally:
        if tmp_output.exists():
            tmp_output.unlink()
    return rewritten


async def rewrite_direct_coded_batches(
    source_path: str | Path,
    batch_dir: str | Path,
    *,
    model: str,
    api_base_url: str = "https://openrouter.ai/api/v1",
    provider: dict[str, Any] | None = None,
    batch_size: int = 250,
    start_batch: int = 0,
    max_batches: int | None = None,
    max_concurrent: int = 10,
    temperature: float = 0.2,
    retry_failed: int = 0,
    skip_existing: bool = True,
    request_timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Rewrite a source bank in resumable row batches."""

    source = load_bank(source_path)
    output_root = Path(batch_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    total_batches = (len(source) + batch_size - 1) // batch_size
    stop_batch = total_batches if max_batches is None else min(total_batches, start_batch + max_batches)
    for batch_idx in range(start_batch, stop_batch):
        start = batch_idx * batch_size
        end = min(len(source), start + batch_size)
        output = output_root / f"batch_{batch_idx:05d}_{start:07d}_{end:07d}.parquet"
        if skip_existing and output.exists():
            qc = validate_bank(load_bank(output))
            summaries.append({"batch": batch_idx, "start": start, "end": end, "skipped": True, "qc": qc})
            if qc["pass"]:
                continue
        tmp_source = output_root / f".source_{batch_idx:05d}.parquet"
        source.iloc[start:end].to_parquet(tmp_source, index=False)
        try:
            df = await rewrite_direct_coded_bank(
                tmp_source,
                output,
                model=model,
                api_base_url=api_base_url,
                provider=provider,
                max_concurrent=max_concurrent,
                temperature=temperature,
                retry_failed=retry_failed,
                request_timeout=request_timeout,
            )
            summaries.append(
                {
                    "batch": batch_idx,
                    "start": start,
                    "end": end,
                    "skipped": False,
                    "path": str(output),
                    "qc": validate_bank(df),
                }
            )
        finally:
            if tmp_source.exists():
                tmp_source.unlink()
    return summaries


def _failed_rewrite_row(row: pd.Series, reason: str) -> dict[str, Any]:
    source = str(row.get("original_message") or "")
    scores = score_message(source)
    entry = {
        "country": row["country"],
        "period": row["period"],
        "theta": row["theta"],
    }
    agent = {
        "id": row["sender_agent_id"],
        "signal": row["sender_signal"],
        "z_score": row["sender_z_score"],
    }
    return _message_bank_row(
        message_id=str(row["message_id"]),
        source_arm_id=str(row["source_arm_id"]),
        entry=entry,
        agent=agent,
        original_message=source,
        direct_message=source,
        coded_message=source,
        first_order_similarity=0.0,
        factual_equivalence_pass=False,
        style_balance_pass=False,
        qc_notes=reason,
        score_values=scores,
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def messages_for_period(
    bank: pd.DataFrame,
    *,
    country: int,
    period: int,
    transform: str,
) -> tuple[dict[int, str], tuple[int, int] | None]:
    """Return fixed messages for one period from a frozen message bank."""

    if transform not in {"direct", "coded", "original"}:
        raise ValueError(f"message_bank replay requires transform direct/coded/original, got {transform!r}")
    column = {
        "direct": "direct_message",
        "coded": "coded_message",
        "original": "original_message",
    }[transform]
    matches = bank[(bank["country"] == country) & (bank["period"] == period)]
    if matches.empty:
        return {}, None
    return (
        {
            int(row.sender_agent_id): str(getattr(row, column))
            for row in matches.itertuples(index=False)
            if pd.notna(getattr(row, column))
        },
        (country, period),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate clean-run message banks")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("path")

    p_score = sub.add_parser("score")
    p_score.add_argument("path")
    p_score.add_argument("--text-column", default="message")

    p_balance = sub.add_parser("balance")
    p_balance.add_argument("path")
    p_balance.add_argument("--output", required=True)

    p_manual = sub.add_parser("manual-audit-sample")
    p_manual.add_argument("path")
    p_manual.add_argument("--output", required=True)
    p_manual.add_argument("--n", type=int, default=50)
    p_manual.add_argument("--seed", type=int, default=5150)

    p_summary = sub.add_parser("summary-stats")
    p_summary.add_argument("paths", nargs="+")
    p_summary.add_argument("--output", required=True)

    p_original = sub.add_parser("build-original")
    p_original.add_argument("--log", required=True)
    p_original.add_argument("--source-arm-id", required=True)
    p_original.add_argument("--output", required=True)
    p_original.add_argument("--theta-min", type=float, default=None)
    p_original.add_argument("--theta-max", type=float, default=None)

    p_paired = sub.add_parser("build-paired")
    p_paired.add_argument("--direct-log", required=True)
    p_paired.add_argument("--coded-log", required=True)
    p_paired.add_argument("--direct-source-arm-id", required=True)
    p_paired.add_argument("--coded-source-arm-id", required=True)
    p_paired.add_argument("--output", required=True)
    p_paired.add_argument("--theta-min", type=float, default=None)
    p_paired.add_argument("--theta-max", type=float, default=None)

    p_manifest_source = sub.add_parser("build-manifest-source")
    p_manifest_source.add_argument("--manifest", required=True)
    p_manifest_source.add_argument("--output", required=True)
    p_manifest_source.add_argument("--arm-id", default=None)
    p_manifest_source.add_argument("--claim", default="direct_coded_mechanism")
    p_manifest_source.add_argument("--max-cells", type=int, default=None)
    p_manifest_source.add_argument("--max-agents", type=int, default=None)

    p_derive_original = sub.add_parser("derive-original")
    p_derive_original.add_argument("--source", required=True)
    p_derive_original.add_argument("--output", required=True)
    p_derive_original.add_argument("--text-column", default="direct_message")

    p_combine = sub.add_parser("combine-batches")
    p_combine.add_argument("--batch-dir", required=True)
    p_combine.add_argument("--output", required=True)
    p_combine.add_argument("--pattern", default="batch_*.parquet")
    p_combine.add_argument("--allow-failed", action="store_true")

    p_rewrite = sub.add_parser("rewrite-pairs")
    p_rewrite.add_argument("--source", required=True)
    p_rewrite.add_argument("--output", required=True)
    p_rewrite.add_argument("--model", required=True)
    p_rewrite.add_argument("--api-base-url", default="https://openrouter.ai/api/v1")
    p_rewrite.add_argument("--provider-json", default=None)
    p_rewrite.add_argument("--max-rows", type=int, default=None)
    p_rewrite.add_argument("--max-concurrent", type=int, default=10)
    p_rewrite.add_argument("--temperature", type=float, default=0.2)
    p_rewrite.add_argument("--allow-overwrite", action="store_true")
    p_rewrite.add_argument("--retry-failed", type=int, default=0)
    p_rewrite.add_argument("--request-timeout", type=float, default=60.0)

    p_rewrite_batches = sub.add_parser("rewrite-pairs-batches")
    p_rewrite_batches.add_argument("--source", required=True)
    p_rewrite_batches.add_argument("--batch-dir", required=True)
    p_rewrite_batches.add_argument("--model", required=True)
    p_rewrite_batches.add_argument("--api-base-url", default="https://openrouter.ai/api/v1")
    p_rewrite_batches.add_argument("--provider-json", default=None)
    p_rewrite_batches.add_argument("--batch-size", type=int, default=250)
    p_rewrite_batches.add_argument("--start-batch", type=int, default=0)
    p_rewrite_batches.add_argument("--max-batches", type=int, default=None)
    p_rewrite_batches.add_argument("--max-concurrent", type=int, default=10)
    p_rewrite_batches.add_argument("--temperature", type=float, default=0.2)
    p_rewrite_batches.add_argument("--retry-failed", type=int, default=0)
    p_rewrite_batches.add_argument("--request-timeout", type=float, default=60.0)
    p_rewrite_batches.add_argument("--rerun-existing", action="store_true")

    p_promote = sub.add_parser("promote-bank")
    p_promote.add_argument("--candidate", required=True)
    p_promote.add_argument("--target", required=True)
    p_promote.add_argument("--backup", default=None)

    args = parser.parse_args()

    if args.command == "validate":
        df = load_bank(args.path)
        print(validate_bank(df))
    elif args.command == "score":
        df = load_bank(args.path)
        if args.text_column not in df.columns:
            raise SystemExit(f"Missing text column: {args.text_column}")
        scores = pd.DataFrame([score_message(text) for text in df[args.text_column].fillna("")])
        print(scores.describe().to_string())
    elif args.command == "balance":
        df = load_bank(args.path)
        table = balance_table(df)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        print(f"wrote {output}")
        print(table.to_string(index=False))
    elif args.command == "manual-audit-sample":
        df = load_bank(args.path)
        sample = manual_audit_sample(df, n=args.n, seed=args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(output, index=False)
        print(f"wrote {output} ({len(sample)} rows)")
    elif args.command == "summary-stats":
        summary = {
            Path(path).name: summary_stats(load_bank(path))
            for path in args.paths
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"wrote {output}")
    elif args.command == "build-original":
        df = build_original_bank_from_log(
            args.log,
            source_arm_id=args.source_arm_id,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        print(f"wrote {output} ({len(df)} rows)")
        print(validate_bank(df))
    elif args.command == "build-paired":
        df = build_paired_bank_from_logs(
            args.direct_log,
            args.coded_log,
            direct_source_arm_id=args.direct_source_arm_id,
            coded_source_arm_id=args.coded_source_arm_id,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        print(f"wrote {output} ({len(df)} rows)")
        print(validate_bank(df))
    elif args.command == "build-manifest-source":
        df = build_source_bank_from_manifest(
            args.manifest,
            args.output,
            arm_id=args.arm_id,
            claim=args.claim,
            max_cells=args.max_cells,
            max_agents=args.max_agents,
        )
        print(f"wrote {args.output} ({len(df)} rows)")
        print(validate_bank(df))
    elif args.command == "derive-original":
        df = derive_original_replay_bank(
            args.source,
            args.output,
            text_column=args.text_column,
        )
        print(f"wrote {args.output} ({len(df)} rows)")
        print(validate_bank(df))
    elif args.command == "combine-batches":
        df = combine_bank_batches(
            args.batch_dir,
            args.output,
            pattern=args.pattern,
            require_pass=not args.allow_failed,
        )
        print(f"wrote {args.output} ({len(df)} rows)")
        print(validate_bank(df))
    elif args.command == "rewrite-pairs":
        provider = json.loads(args.provider_json) if args.provider_json else None
        df = asyncio.run(
            rewrite_direct_coded_bank(
                args.source,
                args.output,
                model=args.model,
                api_base_url=args.api_base_url,
                provider=provider,
                max_rows=args.max_rows,
                max_concurrent=args.max_concurrent,
                temperature=args.temperature,
                allow_overwrite=args.allow_overwrite,
                retry_failed=args.retry_failed,
                request_timeout=args.request_timeout,
            )
        )
        print(f"wrote {args.output} ({len(df)} rows)")
        print(validate_bank(df))
    elif args.command == "rewrite-pairs-batches":
        provider = json.loads(args.provider_json) if args.provider_json else None
        summaries = asyncio.run(
            rewrite_direct_coded_batches(
                args.source,
                args.batch_dir,
                model=args.model,
                api_base_url=args.api_base_url,
                provider=provider,
                batch_size=args.batch_size,
                start_batch=args.start_batch,
                max_batches=args.max_batches,
                max_concurrent=args.max_concurrent,
                temperature=args.temperature,
                retry_failed=args.retry_failed,
                skip_existing=not args.rerun_existing,
                request_timeout=args.request_timeout,
            )
        )
        print(json.dumps(summaries, indent=2, sort_keys=True))
    elif args.command == "promote-bank":
        qc = promote_valid_bank(args.candidate, args.target, backup_path=args.backup)
        print(f"promoted {args.candidate} -> {args.target}")
        print(qc)


if __name__ == "__main__":
    main()
