"""Regime-analyst pilot for "The Intelligence Cost of Surveillance" (paper 2, Phase A).

Feeds archived citizen messages (written with vs. without a surveillance
warning) to "regime analyst" LLMs and records how well the analysts recover
the state (theta / coup outcome), the cell's join fraction, and each sender's
JOIN/STAY decision. The paired design holds theta, senders, and signals fixed
across arms, so any accuracy gap is caused by the surveillance-induced change
in the messages themselves.

Blinding note: the analyst prompt is identical across arms — no arm labels,
no theta, no signals, no briefings, and never the sender `reasoning` field
(which leaks decisions). The only arm-varying content is the message text.
An analyst inferring the arm from message *style* is not a confound: guarded
style is the equilibrium object the regime must read through.

Usage:
  uv run python -m agent_based_simulation.analyst \
      --analyst-model deepseek/deepseek-v4-flash-20260423 \
      --corpus nested --n-cells 150 --temperature 0.0

Outputs (per analyst model):
  output/analyst-pilot/<analyst_slug>/experiment_analyst_<corpus>_<arm>_summary.csv
  output/analyst-pilot/<analyst_slug>/experiment_analyst_<corpus>_<arm>_log.json
  output/analyst-pilot/<analyst_slug>/analyst_run_config.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .runtime import (
    PROJECT_ROOT,
    build_cache_key_and_request,
    deterministic_hash,
    get_cache,
    model_slug,
)

ANALYST_PROMPT_VERSION = "v1"

SYSTEM_ANALYST = (
    "You are an intelligence analyst working for the security services of an "
    "authoritarian regime. Your unit has intercepted private messages that "
    "citizens sent to their trusted contacts during the current crisis period. "
    "Citizens were deciding whether to JOIN an uprising or STAY home. Use only "
    "the intercepted messages below. Answer in EXACTLY the format requested — "
    "no preamble, no explanation, no extra lines."
)

NESTED_BASELINE_LOG = (
    "output/revision-nested-comm/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json"
)
NESTED_SURVEILLANCE_LOG = (
    "output/revision-nested-surv/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json"
)
CODED_PAIRS_PARQUET = "CLEAN_RUN/message_banks/direct_coded_pairs.parquet"
CLEAN_QWEN30_BASELINE = (
    "CLEAN_RUN/output/random_theta_pilot_large/large_random_comm_baseline_qwen30/agents.parquet"
)
CLEAN_QWEN30_SURVEILLANCE = (
    "CLEAN_RUN/output/random_theta_pilot_large/large_random_surv_sender_only_qwen30/agents.parquet"
)

MIN_SENDERS_PER_CELL = 20
MAX_SENDERS_PER_CELL = 25


@dataclass
class Sender:
    agent_id: int
    message: str
    true_decision: str | None  # "JOIN" / "STAY" / None (coded_pairs bank)


@dataclass
class CellItem:
    """One analyst task: a cell's intercepted messages under one arm."""

    corpus: str
    arm: str
    country: int
    period: int
    theta: float
    theta_star: float | None
    coup_success: bool | None
    senders: list[Sender] = field(default_factory=list)  # display order

    @property
    def jf_true_shown(self) -> float | None:
        decisions = [s.true_decision for s in self.senders if s.true_decision]
        if not decisions:
            return None
        return sum(d == "JOIN" for d in decisions) / len(decisions)


# ── Dataset assembly ──────────────────────────────────────────────────


def _clean_message(text: str) -> str:
    """Collapse whitespace; strip stray wrapping quotes."""
    msg = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(msg) >= 2 and msg[0] == msg[-1] and msg[0] in {'"', "'"}:
        msg = msg[1:-1].strip()
    return msg


def _degenerate(msg: str) -> bool:
    """Sender-model glitch output (symbol spam / runaway length); ~0.3% of corpus.

    Excluded at the usability layer so the intersection rule drops the sender
    from BOTH arms, keeping the paired sender sets identical.
    """
    if not msg:
        return True
    if len(msg) > 1500:
        return True
    alpha = sum(c.isalpha() or c.isspace() for c in msg) / len(msg)
    return alpha < 0.75


def _usable(agent: dict) -> bool:
    msg = _clean_message(agent.get("message_sent", ""))
    return (
        not agent.get("api_error")
        and agent.get("decision") in ("JOIN", "STAY")
        and bool(msg)
        and not _degenerate(msg)
    )


def _shuffle_senders(senders: list[Sender], country: int, period: int, arm: str) -> list[Sender]:
    rng = np.random.default_rng(
        deterministic_hash(("analyst_order", country, period, arm)) % (2**32)
    )
    order = rng.permutation(len(senders))
    return [senders[i] for i in order]


def load_nested_cells(
    repo_root: Path, surv_log: str | None = None
) -> list[tuple[CellItem, CellItem]]:
    """Load the paired nested-Llama corpus: (baseline_item, surveillance_item) per cell.

    surv_log overrides the surveillance-arm source (dose-response variants:
    mild/severe arms on the identical theta grid, paired against the same
    baseline).
    """
    with open(repo_root / NESTED_BASELINE_LOG) as f:
        base_entries = json.load(f)
    with open(repo_root / (surv_log or NESTED_SURVEILLANCE_LOG)) as f:
        surv_entries = json.load(f)

    base_by_key = {(e["country"], e["period"]): e for e in base_entries}
    surv_by_key = {(e["country"], e["period"]): e for e in surv_entries}

    pairs: list[tuple[CellItem, CellItem]] = []
    for key in sorted(base_by_key):
        if key not in surv_by_key:
            continue
        be, se = base_by_key[key], surv_by_key[key]
        if abs(be["theta"] - se["theta"]) > 1e-12:
            raise ValueError(f"theta mismatch at cell {key}: {be['theta']} vs {se['theta']}")
        base_agents = {a["id"]: a for a in be["agents"] if _usable(a)}
        surv_agents = {a["id"]: a for a in se["agents"] if _usable(a)}
        common_ids = sorted(set(base_agents) & set(surv_agents))[:MAX_SENDERS_PER_CELL]
        if len(common_ids) < MIN_SENDERS_PER_CELL:
            continue

        def _item(entry: dict, agents: dict, arm: str) -> CellItem:
            senders = [
                Sender(
                    agent_id=i,
                    message=_clean_message(agents[i]["message_sent"]),
                    true_decision=agents[i]["decision"],
                )
                for i in common_ids
            ]
            return CellItem(
                corpus="nested",
                arm=arm,
                country=entry["country"],
                period=entry["period"],
                theta=entry["theta"],
                theta_star=entry.get("theta_star"),
                coup_success=bool(entry.get("coup_success")),
                senders=_shuffle_senders(senders, entry["country"], entry["period"], arm),
            )

        pairs.append((_item(be, base_agents, "baseline"), _item(se, surv_agents, "surveillance")))
    return pairs


def load_coded_pairs(repo_root: Path) -> list[tuple[CellItem, CellItem]]:
    """Load direct/coded message bank: (direct_item, coded_item) per cell. Theta truth only."""
    import pandas as pd

    df = pd.read_parquet(repo_root / CODED_PAIRS_PARQUET)
    pairs: list[tuple[CellItem, CellItem]] = []
    for (country, period), grp in df.groupby(["country", "period"], sort=True):
        grp = grp.sort_values("sender_agent_id").head(MAX_SENDERS_PER_CELL)
        theta = float(grp["theta"].iloc[0])

        def _item(col: str, arm: str) -> CellItem:
            senders = [
                Sender(
                    agent_id=int(r.sender_agent_id),
                    message=_clean_message(getattr(r, col)),
                    true_decision=None,
                )
                for r in grp.itertuples()
                if _clean_message(getattr(r, col))
            ]
            return CellItem(
                corpus="coded",
                arm=arm,
                country=int(country),
                period=int(period),
                theta=theta,
                theta_star=None,
                coup_success=None,
                senders=_shuffle_senders(senders, int(country), int(period), arm),
            )

        direct, coded = _item("direct_message", "direct"), _item("coded_message", "coded")
        if len(direct.senders) >= MIN_SENDERS_PER_CELL and len(coded.senders) >= MIN_SENDERS_PER_CELL:
            pairs.append((direct, coded))
    return pairs


def load_clean_qwen30_cells(repo_root: Path) -> list[tuple[CellItem, CellItem]]:
    """Secondary corpus: CLEAN_RUN qwen30 agents.parquet pair (250 cells x <=10 senders)."""
    import pandas as pd

    def _cells(path: str, arm: str) -> dict[tuple[int, int], CellItem]:
        df = pd.read_parquet(repo_root / path)
        out: dict[tuple[int, int], CellItem] = {}
        for (country, period), grp in df.groupby(["country", "period"], sort=True):
            senders = [
                Sender(
                    agent_id=int(r.agent_id),
                    message=_clean_message(r.message_sent),
                    true_decision=r.decision if r.decision in ("JOIN", "STAY") else None,
                )
                for r in grp.sort_values("agent_id").itertuples()
                if not r.api_error
                and r.decision in ("JOIN", "STAY")
                and _clean_message(r.message_sent)
            ]
            if not senders:
                continue
            out[(int(country), int(period))] = CellItem(
                corpus="clean_qwen30",
                arm=arm,
                country=int(country),
                period=int(period),
                theta=float(grp["theta"].iloc[0]),
                theta_star=None,
                coup_success=None,
                senders=_shuffle_senders(senders, int(country), int(period), arm),
            )
        return out

    base = _cells(CLEAN_QWEN30_BASELINE, "baseline")
    surv = _cells(CLEAN_QWEN30_SURVEILLANCE, "surveillance")
    pairs = []
    for key in sorted(set(base) & set(surv)):
        b, s = base[key], surv[key]
        common = sorted(
            {x.agent_id for x in b.senders} & {x.agent_id for x in s.senders}
        )
        if len(common) < 8:  # 10-sender cells; require near-complete overlap
            continue
        b.senders = [x for x in b.senders if x.agent_id in common]
        s.senders = [x for x in s.senders if x.agent_id in common]
        pairs.append((b, s))
    return pairs


def sample_cells(
    pairs: list[tuple[CellItem, CellItem]], n_cells: int, seed: int
) -> list[tuple[CellItem, CellItem]]:
    """Theta-decile-stratified sample of matched cell pairs (deterministic)."""
    if n_cells <= 0 or n_cells >= len(pairs):
        return pairs
    order = np.argsort([p[0].theta for p in pairs], kind="stable")
    n_strata = 10
    strata: list[list[int]] = [[] for _ in range(n_strata)]
    for rank, idx in enumerate(order):
        strata[min(rank * n_strata // len(pairs), n_strata - 1)].append(int(idx))
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    base, extra = divmod(n_cells, n_strata)
    for k, stratum in enumerate(strata):
        take = min(base + (1 if k < extra else 0), len(stratum))
        picked.extend(rng.choice(stratum, size=take, replace=False).tolist())
    picked = sorted(picked)
    return [pairs[i] for i in picked]


# ── Prompts and parsing ───────────────────────────────────────────────


def build_analyst_prompt(item: CellItem) -> tuple[str, str]:
    n = len(item.senders)
    lines = [f'Sender {k + 1}: "{s.message}"' for k, s in enumerate(item.senders)]
    user = (
        "INTERCEPTED MESSAGES (one per sender, order is arbitrary):\n\n"
        + "\n".join(lines)
        + "\n\nYOUR TASKS — answer in exactly this format, one line each:\n"
        "FALL: <integer 0-100>   (probability the regime falls within the month: "
        "0 = certainly survives, 100 = certainly falls)\n"
        f"JOIN_PERCENT: <integer 0-100>   (percentage of these {n} senders who chose to JOIN)\n"
        "Then one line per sender, in the order shown:\n"
        "SENDER 1: <JOIN or STAY> <integer 0-100>   "
        "(your verdict, then probability this sender chose JOIN)\n"
        "SENDER 2: ...\n"
        f"...continue through SENDER {n}."
    )
    return SYSTEM_ANALYST, user


_FALL_RE = re.compile(r"^\W*FALL\W*?[:\-]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_JOINPCT_RE = re.compile(r"^\W*JOIN[\s_]?PERCENT\W*?[:\-]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_SENDER_RE = re.compile(
    r"^\W*SENDER\s*(\d+)\W*?[:\-]?\s*(JOIN|STAY)\b(.*)$", re.IGNORECASE
)
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_SENDER_NUM_ONLY_RE = re.compile(r"^\W*SENDER\s*(\d+)\W*?[:\-]?\s*(\d+(?:\.\d+)?)\s*\W*$", re.IGNORECASE)


def _clamp01_100(value: float) -> float:
    return max(0.0, min(100.0, value))


def parse_analyst_response(text: str, n_senders: int) -> dict:
    """Parse the strict line format; tolerate markdown wrapping and truncation.

    Returns dict with: fall_est, join_pct_est (floats or None),
    senders: list of n_senders dicts {verdict, p_join, p_imputed, parse_ok},
    parse_ok_fall, parse_ok_join_pct, api_error (bool).
    """
    out = {
        "fall_est": None,
        "join_pct_est": None,
        "senders": [
            {"verdict": None, "p_join": None, "p_imputed": False, "parse_ok": False}
            for _ in range(n_senders)
        ],
        "parse_ok_fall": False,
        "parse_ok_join_pct": False,
        "api_error": False,
    }
    if not text or text.startswith("[API Error:") or text.startswith("[Empty response"):
        out["api_error"] = True
        return out

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if out["fall_est"] is None:
            m = _FALL_RE.match(line)
            if m:
                out["fall_est"] = _clamp01_100(float(m.group(1)))
                out["parse_ok_fall"] = True
                continue
        if out["join_pct_est"] is None:
            m = _JOINPCT_RE.match(line)
            if m:
                out["join_pct_est"] = _clamp01_100(float(m.group(1)))
                out["parse_ok_join_pct"] = True
                continue
        m = _SENDER_RE.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_senders and not out["senders"][idx]["parse_ok"]:
                verdict = m.group(2).upper()
                num = _NUM_RE.search(m.group(3))
                if num is not None:
                    p = _clamp01_100(float(num.group(1)))
                    imputed = False
                else:
                    p = 75.0 if verdict == "JOIN" else 25.0
                    imputed = True
                out["senders"][idx] = {
                    "verdict": verdict, "p_join": p, "p_imputed": imputed, "parse_ok": True,
                }
            continue
        m = _SENDER_NUM_ONLY_RE.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_senders and not out["senders"][idx]["parse_ok"]:
                p = _clamp01_100(float(m.group(2)))
                out["senders"][idx] = {
                    "verdict": "JOIN" if p >= 50.0 else "STAY",
                    "p_join": p, "p_imputed": False, "parse_ok": True,
                }

    # Loose fallbacks for cell-level lines buried in prose.
    if out["fall_est"] is None:
        m = re.search(r"\bFALL\b\D{0,20}?(\d+(?:\.\d+)?)", str(text), re.IGNORECASE)
        if m:
            out["fall_est"] = _clamp01_100(float(m.group(1)))
            out["parse_ok_fall"] = True
    if out["join_pct_est"] is None:
        m = re.search(r"\bJOIN[\s_]?PERCENT\b\D{0,20}?(\d+(?:\.\d+)?)", str(text), re.IGNORECASE)
        if m:
            out["join_pct_est"] = _clamp01_100(float(m.group(1)))
            out["parse_ok_join_pct"] = True
    return out


# ── LLM call (cache-aware; parameterized max_tokens, unlike _call_llm) ─


# Per-run usage accumulator (single event loop; reset in run_pilot).
_USAGE = {"calls": 0, "cost_usd": 0.0, "completion_tokens": 0, "prompt_tokens": 0}


def _record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    _USAGE["calls"] += 1
    _USAGE["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    _USAGE["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = (getattr(usage, "model_extra", None) or {}).get("cost")
    if isinstance(cost, (int, float)):
        _USAGE["cost_usd"] += float(cost)


async def _call_analyst_llm(
    client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_retries: int = 5,
    max_empty_retries: int = 3,
    request_timeout: int = 180,
) -> str:
    """Cache-aware analyst call.

    Reasoning models (e.g. deepseek-v4-flash) spend completion tokens on a
    hidden reasoning channel; if content comes back empty with
    finish_reason == "length", the token budget is escalated (x2, twice)
    instead of retrying the identical request. The response is cached under
    the ORIGINAL request key so reruns stay idempotent at the item level.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    cache = get_cache()
    cache_key = cache_req = None
    if cache is not None:
        cache_key, cache_req = build_cache_key_and_request(
            model=model_name, messages=messages, max_tokens=max_tokens, temperature=temperature,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    api_attempts = empty_attempts = timeout_attempts = escalations = 0
    effective_max_tokens = max_tokens
    while True:
        try:
            async with semaphore:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=effective_max_tokens,
                        temperature=temperature,
                    ),
                    timeout=request_timeout,
                )
            _record_usage(response)
            content = ""
            finish_reason = None
            if response and getattr(response, "choices", None):
                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length" and escalations < 2:
                # Reasoning channel (or verbosity) exhausted the budget;
                # content may be empty OR truncated mid-answer. Escalate.
                escalations += 1
                effective_max_tokens *= 2
                continue
            if len(content) < 3 or re.search(r"[A-Za-z0-9]", content) is None:
                empty_attempts += 1
                if empty_attempts >= max_empty_retries:
                    return "[Empty response after retries]"
                await asyncio.sleep(min(6.0, 0.5 * (2 ** (empty_attempts - 1))))
                continue
            if finish_reason == "length":
                # Escalations exhausted: return what we have (parser salvages
                # cell-level lines) but never cache a truncated response.
                return content
            if cache is not None and cache_key is not None:
                cache.set(cache_key, cache_req, content)
            return content
        except (asyncio.TimeoutError, TimeoutError):
            timeout_attempts += 1
            if timeout_attempts >= 3:
                return "[API Error: request timed out after 3 retries]"
            await asyncio.sleep(min(5.0, 2.0 * timeout_attempts))
        except Exception as e:  # rate limits, network, provider errors
            api_attempts += 1
            if api_attempts >= max_retries:
                return f"[API Error: {e}]"
            await asyncio.sleep(min(10.0, 2 ** (api_attempts - 1)))


# ── Runner ────────────────────────────────────────────────────────────


def _summary_row(item: CellItem, analyst_model: str, parsed: dict) -> dict:
    senders = parsed["senders"]
    parsed_senders = [
        (s, p) for s, p in zip(item.senders, senders) if p["parse_ok"]
    ]
    truth_pairs = [(s, p) for s, p in parsed_senders if s.true_decision is not None]
    sender_accuracy = (
        sum(p["verdict"] == s.true_decision for s, p in truth_pairs) / len(truth_pairs)
        if truth_pairs
        else None
    )
    n_top5_join_true = None
    if truth_pairs and len(truth_pairs) >= 5:
        ranked = sorted(
            range(len(item.senders)),
            key=lambda i: (-(senders[i]["p_join"] if senders[i]["parse_ok"] else -1.0), i),
        )
        top5 = [i for i in ranked if senders[i]["parse_ok"]][:5]
        n_top5_join_true = sum(item.senders[i].true_decision == "JOIN" for i in top5)
    return {
        "analyst_model": analyst_model,
        "corpus": item.corpus,
        "arm": item.arm,
        "country": item.country,
        "period": item.period,
        "theta": item.theta,
        "theta_star": item.theta_star,
        "coup_success": item.coup_success,
        "n_senders_shown": len(item.senders),
        "jf_true_shown": item.jf_true_shown,
        "n_join_true_shown": (
            sum(s.true_decision == "JOIN" for s in item.senders)
            if item.jf_true_shown is not None
            else None
        ),
        "fall_est": parsed["fall_est"],
        "join_pct_est": parsed["join_pct_est"],
        "n_sender_parsed": sum(p["parse_ok"] for p in senders),
        "n_sender_imputed_p": sum(p["p_imputed"] for p in senders),
        "sender_accuracy": sender_accuracy,
        "n_top5_join_true": n_top5_join_true,
        "parse_ok_fall": parsed["parse_ok_fall"],
        "parse_ok_join_pct": parsed["parse_ok_join_pct"],
        "api_error": parsed["api_error"],
    }


SUMMARY_COLUMNS = [
    "analyst_model", "corpus", "arm", "country", "period", "theta", "theta_star",
    "coup_success", "n_senders_shown", "jf_true_shown", "n_join_true_shown",
    "fall_est", "join_pct_est", "n_sender_parsed", "n_sender_imputed_p",
    "sender_accuracy", "n_top5_join_true", "parse_ok_fall", "parse_ok_join_pct",
    "api_error",
]


async def run_pilot(args) -> None:
    repo_root = Path(args.repo_root) if args.repo_root else PROJECT_ROOT

    if args.corpus == "nested":
        pairs = load_nested_cells(repo_root, surv_log=args.surv_log)
    elif args.corpus == "coded_pairs":
        pairs = load_coded_pairs(repo_root)
    elif args.corpus == "clean_qwen30":
        pairs = load_clean_qwen30_cells(repo_root)
    else:
        raise ValueError(f"unknown corpus: {args.corpus}")

    if args.holdout:
        # Complement of the main sample: cells never touched by prior runs.
        main_keys = {
            (a.country, a.period) for a, _ in sample_cells(pairs, args.n_cells, args.seed)
        }
        pairs = [p for p in pairs if (p[0].country, p[0].period) not in main_keys]
    else:
        pairs = sample_cells(pairs, args.n_cells, args.seed)
    if args.n_messages > 0:
        rng = np.random.default_rng(args.seed + 1)
        for a, b in pairs:
            common = sorted({s.agent_id for s in a.senders} & {s.agent_id for s in b.senders})
            keep = set(
                rng.choice(common, size=min(args.n_messages, len(common)), replace=False).tolist()
            )
            a.senders = [s for s in a.senders if s.agent_id in keep]
            b.senders = [s for s in b.senders if s.agent_id in keep]

    items = [item for pair in pairs for item in pair]
    arms = sorted({i.arm for i in items})
    print(
        f"[analyst] corpus={args.corpus} cells={len(pairs)} items={len(items)} "
        f"arms={arms} analyst={args.analyst_model}"
    )

    if args.dry_run:
        for item in items[: 2 * len(arms)]:
            system, user = build_analyst_prompt(item)
            print(f"\n===== DRY RUN: {item.corpus}/{item.arm} cell=({item.country},{item.period}) "
                  f"theta={item.theta:.3f} n={len(item.senders)} =====")
            print("--- system ---\n" + system)
            print("--- user ---\n" + user)
        return

    from openai import AsyncOpenAI

    for k in _USAGE:
        _USAGE[k] = 0
    api_key = os.environ.get("OPENROUTER_API_KEY", "") or "not-needed"
    client = AsyncOpenAI(base_url=args.api_base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    out_dir = Path(args.output_dir) / "analyst-pilot" / model_slug(args.analyst_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "analyst_model": args.analyst_model,
        "corpus": args.corpus,
        "run_label": args.run_label,
        "n_cells": len(pairs),
        "n_messages": args.n_messages,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_version": ANALYST_PROMPT_VERSION,
        "source_paths": {
            "nested": [NESTED_BASELINE_LOG, NESTED_SURVEILLANCE_LOG],
            "coded_pairs": [CODED_PAIRS_PARQUET],
            "clean_qwen30": [CLEAN_QWEN30_BASELINE, CLEAN_QWEN30_SURVEILLANCE],
        }[args.corpus],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    config_path = out_dir / "analyst_run_config.json"
    existing = json.loads(config_path.read_text()) if config_path.exists() else []
    if not isinstance(existing, list):
        existing = [existing]
    existing.append(run_config)
    config_path.write_text(json.dumps(existing, indent=2))

    async def _one(item: CellItem) -> tuple[CellItem, dict, str]:
        system, user = build_analyst_prompt(item)
        response = await _call_analyst_llm(
            client, args.analyst_model, system, user, semaphore,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        return item, parse_analyst_response(response, len(item.senders)), response

    results = await asyncio.gather(*[_one(item) for item in items])

    for arm in arms:
        arm_results = [(i, p, r) for i, p, r in results if i.arm == arm]
        corpus_label = f"{args.corpus}-{args.run_label}" if args.run_label else args.corpus
        label = f"{corpus_label}_{arm}"
        summary_path = out_dir / f"experiment_analyst_{label}_summary.csv"
        log_path = out_dir / f"experiment_analyst_{label}_log.json"

        rows = [_summary_row(item, args.analyst_model, parsed) for item, parsed, _ in arm_results]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        log_entries = []
        for (item, parsed, response), row in zip(arm_results, rows):
            log_entries.append({
                **row,
                "prompt_version": ANALYST_PROMPT_VERSION,
                "response_text": response,
                "senders": [
                    {
                        "display_idx": k + 1,
                        "agent_id": s.agent_id,
                        "true_decision": s.true_decision,
                        **parsed["senders"][k],
                    }
                    for k, s in enumerate(item.senders)
                ],
            })
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=1)

        n_err = sum(r["api_error"] for r in rows)
        fall_rate = float(np.mean([r["parse_ok_fall"] for r in rows])) if rows else 0.0
        sender_rate = (
            float(np.mean([r["n_sender_parsed"] / r["n_senders_shown"] for r in rows]))
            if rows else 0.0
        )
        print(
            f"[analyst] arm={arm}: {len(rows)} cells -> {summary_path.name} | "
            f"api_errors={n_err} parse(FALL)={fall_rate:.0%} parse(senders)={sender_rate:.0%}"
        )

    print(
        f"[analyst] usage: {_USAGE['calls']} API calls "
        f"({_USAGE['prompt_tokens']:,} in / {_USAGE['completion_tokens']:,} out tokens), "
        f"cost=${_USAGE['cost_usd']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-analyst pilot (paper 2, Phase A)")
    parser.add_argument("--analyst-model", type=str, required=True)
    parser.add_argument("--corpus", type=str, default="nested",
                        choices=["nested", "coded_pairs", "clean_qwen30"])
    parser.add_argument("--n-cells", type=int, default=150,
                        help="Matched cells to sample (0 = all)")
    parser.add_argument("--n-messages", type=int, default=0,
                        help="Subsample k senders per cell (0 = all usable; difficulty knob)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=5150)
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Completion budget incl. hidden reasoning channels; "
                             "auto-escalates x2 (twice) if reasoning exhausts it")
    parser.add_argument("--api-base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--surv-log", type=str, default=None,
                        help="Override surveillance-arm log path (repo-relative); "
                             "for dose-response arms on the same theta grid")
    parser.add_argument("--holdout", action="store_true",
                        help="Use the COMPLEMENT of the --n-cells/--seed sample "
                             "(cells untouched by prior runs)")
    parser.add_argument("--run-label", type=str, default="",
                        help="Suffix for output filenames (e.g. k10) so variant runs "
                             "do not overwrite the main corpus outputs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sample prompts; no API calls")
    args = parser.parse_args()
    asyncio.run(run_pilot(args))


if __name__ == "__main__":
    main()
