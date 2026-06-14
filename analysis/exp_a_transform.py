#!/usr/bin/env python3
r"""Experiment A: willingness x style 2x2 factorial on baseline messages.

Rewrites each baseline (communication) message into four variants, holding the
factual regime-strength assessment fixed:

  w1_direct : willingness-to-act cue PRESENT, plain/direct style
  w1_coded  : willingness-to-act cue PRESENT, indirect/metaphorical style
  w0_direct : willingness cue ABSENT (situation only), plain/direct style
  w0_coded  : willingness cue ABSENT, indirect/metaphorical style

Replaying each variant to fresh Llama receivers identifies the receiver's
response to the willingness factor conditional on factual content, and whether
it is separable from coded style. This is the positive-identification test the
abstract currently can only "point to".

Usage:
  OPENROUTER_API_KEY=... uv run python analysis/exp_a_transform.py build [--limit N] [--audit-only]
  uv run python analysis/exp_a_transform.py audit
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "output" / "expA-cache" / "rewrite_cache.json"
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
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_CACHE))
    tmp.replace(CACHE_PATH)
MODEL_DIR = "meta-llama--llama-3.3-70b-instruct"
COMM = ROOT / "output" / "revision-nested-comm" / MODEL_DIR / "experiment_comm_log.json"
OUT = ROOT / "output"
TRANSLATOR = "qwen/qwen3-30b-a3b-instruct-2507"

CELLS = {
    "w1_direct": ("present", "direct"),
    "w1_coded": ("present", "coded"),
    "w0_direct": ("absent", "direct"),
    "w0_coded": ("absent", "coded"),
}
LOGPATH = {k: OUT / f"expA-{k.replace('_','-')}" / f"{k}_log.json" for k in CELLS}

WILL = {
    "present": ("Clearly convey that the people you know are ready and willing to act "
                "now -- that they will join in / take part if others do."),
    "absent": ("Say NOTHING about whether anyone is ready or willing to act, and "
               "nothing about joining or taking part; describe only the situation "
               "itself."),
}
STYLE = {
    "direct": "Use plain, direct, explicit language; name things openly.",
    "coded": ("Use indirect, metaphorical, deniable language -- allusion and "
              "euphemism rather than explicit statements."),
}
PROMPT = (
    "Rewrite the message below, which one citizen sent to another during a "
    "political crisis. Keep the factual assessment of the government/regime's "
    "strength or weakness EXACTLY the same -- do not make the regime sound "
    "stronger or weaker than the original does. {will} {style} Do not add facts "
    "beyond the original's assessment. Keep it to 1-2 sentences. Reply with ONLY "
    "the rewritten message.\n\nMESSAGE:\n{msg}"
)

WILLINGNESS_TERMS = [
    "ready", "willing", "will join", "join in", "join us", "take part", "act now",
    "with us", "on board", "count on", "stand with", "rise", "take to the street",
    "in if", "ready to", "prepared to act", "move together", "we move",
]
CODED_TERMS = [
    "wall", "crack", "ground shif", "winds", "tide", "storm", "season", "harvest",
    "garden", "weather", "shadow", "whisper", "current", "horizon", "chapter",
    "the air", "feels different", "things are moving", "something",
]
WEAK_TERMS = [
    "weak", "unstable", "instability", "cracking", "crumbl", "collapse", "falling",
    "shift in", "balance of power", "lose control", "losing control", "status quo",
    "in control", "stable", "grip", "firmly",
]


def _count(t: str, terms: list[str]) -> int:
    tl = t.lower()
    return sum(tl.count(x) for x in terms)


def _rate(entries: list[dict], terms: list[str]) -> float:
    msgs = [a.get("message_sent") or "" for e in entries for a in e.get("agents", [])]
    msgs = [m for m in msgs if m.strip()]
    return 100.0 * sum(1 for m in msgs if _count(m, terms) > 0) / len(msgs) if msgs else float("nan")


async def _rewrite(client, msg, will, style, sem):
    p = PROMPT.format(will=WILL[will], style=STYLE[style], msg=msg)
    ck = hashlib.sha256((TRANSLATOR + "\n" + p).encode()).hexdigest()
    if ck in _CACHE:
        return _CACHE[ck]
    async with sem:
        if ck in _CACHE:
            return _CACHE[ck]
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=TRANSLATOR,
                    messages=[{"role": "user", "content": p}],
                    temperature=0.4, max_tokens=200,
                )
                txt = (r.choices[0].message.content or "").strip().strip('"').strip()
                if txt:
                    _CACHE[ck] = txt
                    return txt
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return msg


async def build(limit, audit_only):
    from openai import AsyncOpenAI
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    sem = asyncio.Semaphore(80)
    _cache_load()
    print(f"  cache: {len(_CACHE)} entries loaded")

    base = json.loads(COMM.read_text())
    jobs = [(ei, ai, a.get("message_sent") or "")
            for ei, e in enumerate(base) for ai, a in enumerate(e.get("agents", []))
            if (a.get("message_sent") or "").strip()]
    if limit is not None:
        step = max(1, len(jobs) // limit)
        jobs = jobs[::step][:limit]

    for cellname, (will, style) in CELLS.items():
        print(f"\n=== {cellname}  (willingness={will}, style={style}) ===")
        res = await asyncio.gather(*[_rewrite(client, m, will, style, sem) for _, _, m in jobs])
        _cache_save()
        orig = [m for _, _, m in jobs]
        w_o = 100.0 * sum(1 for m in orig if _count(m, WILLINGNESS_TERMS) > 0) / len(orig)
        w_r = 100.0 * sum(1 for m in res if _count(m, WILLINGNESS_TERMS) > 0) / len(res)
        c_o = 100.0 * sum(1 for m in orig if _count(m, CODED_TERMS) > 0) / len(orig)
        c_r = 100.0 * sum(1 for m in res if _count(m, CODED_TERMS) > 0) / len(res)
        k_o = 100.0 * sum(1 for m in orig if _count(m, WEAK_TERMS) > 0) / len(orig)
        k_r = 100.0 * sum(1 for m in res if _count(m, WEAK_TERMS) > 0) / len(res)
        print(f"  willingness: base {w_o:5.1f}% -> {w_r:5.1f}%   "
              f"coded: base {c_o:5.1f}% -> {c_r:5.1f}%   "
              f"weakness(any): base {k_o:5.1f}% -> {k_r:5.1f}%")
        for (_, _, o), r in list(zip(jobs, res))[:2]:
            print(f"    BASE: {o[:130]}")
            print(f"    {cellname}: {r[:130]}")
        if not audit_only:
            entries = json.loads(COMM.read_text())
            by = {}
            for (ei, ai, _), r in zip(jobs, res):
                by[(ei, ai)] = r
            for ei, e in enumerate(entries):
                for ai, a in enumerate(e.get("agents", [])):
                    if (ei, ai) in by:
                        a["message_sent"] = by[(ei, ai)]
            LOGPATH[cellname].parent.mkdir(parents=True, exist_ok=True)
            LOGPATH[cellname].write_text(json.dumps(entries))
            print(f"  wrote {LOGPATH[cellname]}")


def audit():
    print(f"{'arm':12s} {'willing%':>9} {'coded%':>8} {'weak%':>7}")
    print(f"{'baseline':12s} {_rate(json.loads(COMM.read_text()), WILLINGNESS_TERMS):9.1f} "
          f"{_rate(json.loads(COMM.read_text()), CODED_TERMS):8.1f} "
          f"{_rate(json.loads(COMM.read_text()), WEAK_TERMS):7.1f}")
    for k, p in LOGPATH.items():
        if p.exists():
            e = json.loads(p.read_text())
            print(f"{k:12s} {_rate(e, WILLINGNESS_TERMS):9.1f} {_rate(e, CODED_TERMS):8.1f} {_rate(e, WEAK_TERMS):7.1f}")
        else:
            print(f"{k:12s}  (missing)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--audit-only", action="store_true")
    sub.add_parser("audit")
    a = ap.parse_args()
    if a.cmd == "build":
        asyncio.run(build(a.limit, a.audit_only))
    else:
        audit()


if __name__ == "__main__":
    main()
