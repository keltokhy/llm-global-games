#!/usr/bin/env python3
r"""
Decode surveilled peer messages back into direct language (review Required #2,
part ii). Reads the nested surveillance arm's comm log, asks a different-family
model (Qwen3 30B) to translate each message into plain, direct language with the
same factual content, and writes a fixed-messages log with decoded text
substituted. The decoded log is then replayed to receivers via
--fixed-messages ... --fixed-messages-mode exact.

Usage: OPENROUTER_API_KEY=... uv run python analysis/decode_messages.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "output" / "revision-nested-surv" / "meta-llama--llama-3.3-70b-instruct" / "experiment_comm_log.json"
DST = ROOT / "output" / "revision-nested-decoded" / "decoded_messages_log.json"
TRANSLATOR = "qwen/qwen3-30b-a3b-instruct-2507"

PROMPT = (
    "Rewrite the message below in plain, direct language. Keep exactly the same "
    "factual content and the same overall assessment, but remove euphemism, "
    "metaphor, and allusion: name the regime, security forces, and actions "
    "directly where the original gestures at them. Keep it to 1-2 sentences. "
    "Reply with only the rewritten message.\n\nMESSAGE:\n{msg}"
)


async def translate(client: AsyncOpenAI, msg: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=TRANSLATOR,
                    messages=[{"role": "user", "content": PROMPT.format(msg=msg)}],
                    temperature=0.3,
                    max_tokens=200,
                )
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return txt
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return msg  # fall back to original on persistent failure


async def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    sem = asyncio.Semaphore(40)

    entries = json.loads(SRC.read_text())
    jobs: list[tuple[int, int, str]] = []  # (entry_idx, agent_idx, msg)
    for ei, e in enumerate(entries):
        for ai, ag in enumerate(e.get("agents", [])):
            msg = ag.get("message_sent") or ""
            if msg.strip():
                jobs.append((ei, ai, msg))

    print(f"Translating {len(jobs)} messages via {TRANSLATOR} ...")
    results = await asyncio.gather(*(translate(client, m, sem) for _, _, m in jobs))
    n_fallback = sum(1 for (_, _, orig), new in zip(jobs, results) if new == orig)
    for (ei, ai, _), new in zip(jobs, results):
        entries[ei]["agents"][ai]["message_sent"] = new

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(entries))
    print(f"Wrote {DST} ({len(jobs)} messages, {n_fallback} fallbacks)")


if __name__ == "__main__":
    asyncio.run(main())
