#!/usr/bin/env python3
r"""Experiment B transforms: separate inferred-risk salience from coordination-cue loss.

Builds two fixed-message logs on the nested 500-cell Llama grid:

  risk-stripped : the surveilled messages with all danger/monitoring/risk/fear
                  language removed (LLM rewrite, Qwen3 30B) while the factual
                  regime-strength assessment AND any actionability/coordination
                  cue are preserved. If joining stays depressed when risk
                  language is gone, the surveillance effect is not carried by
                  inferred risk.

  risk-only     : the baseline (communication) messages with one explicit
                  punishment-risk clause appended (deterministic, no
                  coordination content). If joining drops toward the
                  surveillance level, inferred risk alone suffices.

Each output log keeps the (country, period, agent id) structure so it can be
replayed verbatim through `run comm --fixed-messages ... --fixed-messages-mode
exact`.

Usage:
  OPENROUTER_API_KEY=... uv run python analysis/exp_b_transform.py strip [--limit N] [--audit-only]
  uv run python analysis/exp_b_transform.py riskonly
  uv run python analysis/exp_b_transform.py audit
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "output" / "expB-risk-stripped" / "rewrite_cache.json"
_CACHE: dict[str, str] = {}


def _cache_load() -> None:
    global _CACHE
    if CACHE_PATH.exists():
        try:
            _CACHE = json.loads(CACHE_PATH.read_text())
        except Exception:
            _CACHE = {}


def _cache_save() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_CACHE))


def _ckey(prompt: str) -> str:
    return hashlib.sha256((TRANSLATOR + "\n" + prompt).encode()).hexdigest()
MODEL_DIR = "meta-llama--llama-3.3-70b-instruct"
SURV = ROOT / "output" / "revision-nested-surv" / MODEL_DIR / "experiment_comm_log.json"
COMM = ROOT / "output" / "revision-nested-comm" / MODEL_DIR / "experiment_comm_log.json"
STRIPPED = ROOT / "output" / "expB-risk-stripped" / "risk_stripped_log.json"
RISKONLY = ROOT / "output" / "expB-risk-only" / "risk_only_log.json"
TRANSLATOR = "qwen/qwen3-30b-a3b-instruct-2507"

STRIP_PROMPT = (
    "You will rewrite a short message that one citizen privately sent to another "
    "during a political crisis. Remove EVERY hint of danger, surveillance, being "
    "watched, monitored or tracked, secrecy, risk, fear, caution, vigilance, "
    "alertness, or consequences of acting. Do NOT use any of these words or their "
    "synonyms: careful, cautious, caution, watch, watching, alert, attentive, "
    "vigilant, quiet, discreet, safe, safety, risk, danger, monitor, surveillance, "
    "consequence, secret, lay low, keep your head down. KEEP the factual assessment "
    "of the government/regime's strength or weakness, and KEEP any statement about "
    "what people are doing, planning, or whether they are ready to act -- but state "
    "it plainly and openly, with no hedging about danger. Do not add new "
    "information. Keep it to 1-2 sentences and natural. Reply with ONLY the "
    "rewritten message.\n\nMESSAGE:\n{msg}"
)

# Deterministic punishment-risk clauses (no coordination/actionability content),
# rotated by agent id so receivers do not see one identical templated sentence.
RISK_CLAUSES = [
    " Just be careful, though -- I hear the security services are watching closely these days.",
    " Watch yourself: people say the authorities are monitoring everything, and there are real consequences for anyone who gets involved.",
    " Be cautious -- word is the regime is cracking down hard on anyone who steps out of line.",
    " One warning -- they are said to be tracking communications now, and it can go badly for people who act.",
    " Stay safe; the security forces are reportedly keeping a very close eye on things.",
]

# ── Dictionaries for the manipulation-check audit ──────────────────────
# STRONG = explicit repression risk (the reviewer's "inferred punishment risk").
STRONG_RISK_TERMS = [
    "monitor", "surveillance", "security service", "security forces", "consequence",
    "danger", "fear", "afraid", "crackdown", "crack down", "arrest", "traced",
    "tracking", "reprisal", "punish", "regime security", "authorities are watching",
]
# SOFT = caution / vigilance flavour that can shade into situational awareness.
SOFT_RISK_TERMS = [
    "careful", "caution", "cautious", "watch", "watched", "watching", "alert",
    "attentive", "vigilant", "quiet", "discreet", "safe", "safety", "secret",
    "lay low", "keep your head", "risk",
]
RISK_TERMS = STRONG_RISK_TERMS + SOFT_RISK_TERMS
ACTION_TERMS = [
    "ready", "join", "act", "action", "move", "together", "everyone", "prepare",
    "preparation", "plan", "time to", "stand", "rise", "streets", "step up",
    "make our", "contingency", "shift", "change is coming", "people are",
]
WEAK_TERMS = [
    "weak", "unstable", "instability", "cracking", "crumbl", "collapse", "falling",
    "shift in", "balance of power", "lose control", "losing control", "end is",
    "status quo", "change", "wall", "ground shifting",
]


def _count(text: str, terms: list[str]) -> int:
    t = text.lower()
    return sum(t.count(term) for term in terms)


def _rate(logentries: list[dict], terms: list[str]) -> float:
    """Share of messages containing >=1 term."""
    msgs = [a.get("message_sent") or "" for e in logentries for a in e.get("agents", [])]
    msgs = [m for m in msgs if m.strip()]
    if not msgs:
        return float("nan")
    hit = sum(1 for m in msgs if _count(m, terms) > 0)
    return 100.0 * hit / len(msgs)


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


# ── risk-only (deterministic) ──────────────────────────────────────────
def build_risk_only() -> None:
    entries = _load(COMM)
    n = 0
    for e in entries:
        for a in e.get("agents", []):
            msg = a.get("message_sent") or ""
            if not msg.strip():
                continue
            clause = RISK_CLAUSES[int(a.get("id", 0)) % len(RISK_CLAUSES)]
            a["message_sent"] = msg.rstrip() + clause
            n += 1
    RISKONLY.parent.mkdir(parents=True, exist_ok=True)
    RISKONLY.write_text(json.dumps(entries))
    print(f"risk-only: wrote {RISKONLY} ({n} messages, risk clause appended)")


# ── risk-stripped (LLM rewrite) ────────────────────────────────────────
async def _translate(client, msg: str, sem) -> str:
    async with sem:
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=TRANSLATOR,
                    messages=[{"role": "user", "content": STRIP_PROMPT.format(msg=msg)}],
                    temperature=0.3,
                    max_tokens=200,
                )
                txt = (r.choices[0].message.content or "").strip()
                txt = txt.strip('"').strip()
                if txt:
                    return txt
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return msg


async def build_stripped(limit: int | None, audit_only: bool) -> None:
    from openai import AsyncOpenAI

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    sem = asyncio.Semaphore(60)

    entries = _load(SURV)
    jobs: list[tuple[int, int, str]] = []
    for ei, e in enumerate(entries):
        for ai, a in enumerate(e.get("agents", [])):
            msg = a.get("message_sent") or ""
            if msg.strip():
                jobs.append((ei, ai, msg))

    if limit is not None:
        # Sample evenly across the log for a representative pilot.
        step = max(1, len(jobs) // limit)
        jobs = jobs[::step][:limit]

    print(f"risk-stripped: rewriting {len(jobs)} messages via {TRANSLATOR} "
          f"({'PILOT/audit-only' if audit_only else 'full -> log'}) ...")
    results = await asyncio.gather(*[_translate(client, m, sem) for _, _, m in jobs])

    # Pilot audit on the sampled originals vs rewrites.
    orig = [m for _, _, m in jobs]
    n_fallback = sum(1 for o, r in zip(orig, results) if r == o)
    print("\n  Manipulation check (sampled messages):")
    print(f"    n = {len(jobs)}, fallbacks (unchanged) = {n_fallback}")
    for label, terms in [("STRONG risk", STRONG_RISK_TERMS), ("soft risk", SOFT_RISK_TERMS),
                         ("actionability", ACTION_TERMS), ("weakness", WEAK_TERMS)]:
        o_hit = 100.0 * sum(1 for m in orig if _count(m, terms) > 0) / len(orig)
        r_hit = 100.0 * sum(1 for m in results if _count(m, terms) > 0) / len(results)
        print(f"    {label:14s}: surveilled {o_hit:5.1f}%  ->  stripped {r_hit:5.1f}%")
    print("\n  Examples:")
    for (_, _, o), r in list(zip(jobs, results))[:4]:
        print(f"    SURV : {o[:150]}")
        print(f"    STRIP: {r[:150]}\n")

    if audit_only:
        return

    for (ei, ai, _), new in zip(jobs, results):
        entries[ei]["agents"][ai]["message_sent"] = new
    STRIPPED.parent.mkdir(parents=True, exist_ok=True)
    STRIPPED.write_text(json.dumps(entries))
    print(f"risk-stripped: wrote {STRIPPED} ({len(jobs)} messages, {n_fallback} fallbacks)")


# ── audit across all four message logs ─────────────────────────────────
def audit() -> None:
    logs = {
        "baseline (comm)": COMM,
        "surveillance": SURV,
        "risk-stripped": STRIPPED,
        "risk-only": RISKONLY,
    }
    print(f"{'log':18s} {'risk/fear%':>11} {'action%':>9} {'weakness%':>10}")
    for name, path in logs.items():
        if not path.exists():
            print(f"{name:18s}  (missing: {path})")
            continue
        e = _load(path)
        print(f"{name:18s} {_rate(e, RISK_TERMS):11.1f} {_rate(e, ACTION_TERMS):9.1f} {_rate(e, WEAK_TERMS):10.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strip")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--audit-only", action="store_true")
    sub.add_parser("riskonly")
    sub.add_parser("audit")
    args = ap.parse_args()

    if args.cmd == "strip":
        asyncio.run(build_stripped(args.limit, args.audit_only))
    elif args.cmd == "riskonly":
        build_risk_only()
    elif args.cmd == "audit":
        audit()


if __name__ == "__main__":
    main()
