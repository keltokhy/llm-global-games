#!/usr/bin/env python3
r"""
Horton-style contamination/demand audit (fresh calls, post hoc).

For each accessible paper model, presents the actual decision prompt plus a
representative briefing and asks, in separate fresh calls:
  Q1  Does this resemble a formal model/game from economics? Name it.
  Q2  What hypothesis would a researcher running this be testing?
  Q3  If senders wrote messages under a monitoring warning, what effect on
      participation would you predict?

Writes analysis/contamination_audit_results.json and
paper/tables/stats_contamination.tex (\providecommand macros).

Usage: OPENROUTER_API_KEY=... uv run python analysis/contamination_audit.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_based_simulation.briefing import BriefingGenerator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "analysis" / "contamination_audit_results.json"
OUT_TEX = ROOT / "paper" / "tables" / "stats_contamination.tex"

MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "openai/gpt-oss-120b",
    "qwen/qwen3-235b-a22b-2507",
]
N_SAMPLES = 10

DECISION_PROMPT = (
    "You are a citizen deciding whether to JOIN an uprising or STAY home. "
    "You only have your private briefing below. You do not know how many others "
    "will JOIN. Start your first line with exactly JOIN or STAY, then give 1-2 "
    "concise sentences of reasoning."
)

Q1 = (
    "Below is a scenario a participant was given. Do not play the scenario. "
    "Instead, answer: does this setup resemble any formal model or game from "
    "economics or political science? If so, name the model and its authors as "
    "precisely as you can.\n\n--- SCENARIO ---\n{scenario}"
)
Q2 = (
    "Below is a scenario a participant was given. Do not play the scenario. "
    "Instead, answer: if a researcher constructed this scenario as an experiment, "
    "what hypothesis do you think they would be testing?\n\n--- SCENARIO ---\n{scenario}"
)
Q3 = (
    "Below is a scenario a participant was given. Do not play the scenario. "
    "Suppose that in one experimental condition, other citizens' messages were "
    "written after a warning that communications are monitored by regime security "
    "services, while the deciding citizen's own prompt was unchanged. What effect, "
    "if any, would you predict on participation, and why? Answer in 2-3 sentences, "
    "starting with one word: HIGHER, LOWER, or UNCHANGED.\n\n--- SCENARIO ---\n{scenario}"
)

GAME_PAT = re.compile(r"global game|morris.{0,15}shin|carlsson|van damme|regime.change game", re.I)
COORD_PAT = re.compile(r"coordination game|collective action|threshold model|tipping", re.I)
LOWER_PAT = re.compile(r"^\W*LOWER", re.I)


async def ask(client: AsyncOpenAI, model: str, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=400,
                )
                txt = r.choices[0].message.content or ""
                if txt.strip():
                    return txt
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return ""


async def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    sem = asyncio.Semaphore(20)

    gen = BriefingGenerator(seed=5150)
    briefing = gen.generate(0.0, agent_id=0, period=0).render()
    scenario = f"{DECISION_PROMPT}\n\nYOUR INTELLIGENCE BRIEFING:\n{briefing}"

    tasks = []
    for model in MODELS:
        for i in range(N_SAMPLES):
            for qname, q in (("q1", Q1), ("q2", Q2), ("q3", Q3)):
                tasks.append((model, i, qname, ask(client, model, q.format(scenario=scenario), sem)))

    results: dict = {m: {"q1": [], "q2": [], "q3": []} for m in MODELS}
    answers = await asyncio.gather(*(t[3] for t in tasks))
    for (model, i, qname, _), txt in zip(tasks, answers):
        results[model][qname].append(txt)

    summary = {}
    for m in MODELS:
        q1 = results[m]["q1"]
        q3 = results[m]["q3"]
        summary[m] = {
            "n": N_SAMPLES,
            "q1_names_global_game": sum(bool(GAME_PAT.search(t)) for t in q1),
            "q1_names_coordination": sum(bool(COORD_PAT.search(t)) for t in q1),
            "q3_predicts_lower": sum(bool(LOWER_PAT.search(t.strip())) for t in q3),
            "q3_valid": sum(bool(t.strip()) for t in q3),
        }

    OUT_JSON.write_text(json.dumps({"summary": summary, "raw": results}, indent=1))

    n_models = len(MODELS)
    tot = lambda k: sum(summary[m][k] for m in MODELS)  # noqa: E731
    n_total = n_models * N_SAMPLES
    game_pct = 100 * tot("q1_names_global_game") / n_total
    coord_pct = 100 * (tot("q1_names_global_game") + tot("q1_names_coordination")) / n_total
    lower_pct = 100 * tot("q3_predicts_lower") / max(tot("q3_valid"), 1)
    lines = [
        "% Auto-generated by analysis/contamination_audit.py -- do not edit.",
        f"\\providecommand{{\\HortonNModels}}{{{n_models}}}",
        f"\\providecommand{{\\HortonNSamples}}{{{N_SAMPLES}}}",
        f"\\providecommand{{\\HortonGameRecogPct}}{{{game_pct:.0f}\\%}}",
        f"\\providecommand{{\\HortonCoordRecogPct}}{{{min(coord_pct, 100):.0f}\\%}}",
        f"\\providecommand{{\\HortonPredictLowerPct}}{{{lower_pct:.0f}\\%}}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"game {game_pct:.0f}% | coord-or-game {coord_pct:.0f}% | predict-lower {lower_pct:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
