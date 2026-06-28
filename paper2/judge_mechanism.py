"""LLM-judge validation of the data-named mechanism (beatability axis).

A different-family judge scores each surveillance/comm source message 0-10 on
how confidently it asserts the regime is beatable (and collective framing). We
(a) validate against lexicon proxies and (b) test mediation of the replay join
drop within matched cells.

Per-message scores are cached to output/paper2_judge_cache/, so re-runs are cheap
and deterministic; aggregate results are written to result_<judge>_<mode>.json
for the macro emitter (emit_macros.py).

Env: SURV, COMM, MODE (replay|live), JUDGE, N_CELLS.
"""
import asyncio, hashlib, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
SURV = ROOT / os.environ.get("SURV", "output/expB-surv-replay/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json")
COMM = ROOT / os.environ.get("COMM", "output/expB-comm-replay/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json")
JUDGE = os.environ.get("JUDGE", "qwen/qwen3-30b-a3b-instruct-2507")
MODE = os.environ.get("MODE", "replay")  # replay=messages_received, live=message_sent
N_CELLS = int(os.environ.get("N_CELLS", "200"))
MAX_MSGS = 6
CACHE = ROOT / "output" / "paper2_judge_cache"
SAFE = JUDGE.replace("/", "--")
PREFIX = re.compile(r'^\s*(trusted contact|neighbor|colleague|friend|associate)\s*:\s*', re.I)
CERT = re.compile(r"\b(will|certain|clearly|inevitable|undoubtedly|definitely|surely|no doubt|bound to)\b", re.I)
WE = re.compile(r"\b(we|us|our|together|all of us|everyone|collective)\b", re.I)
PROMPT = (
    "A private message between citizens deciding whether to join an uprising. Rate two things 0-10.\n"
    "beatable: how confidently does it assert the regime is weak / vulnerable / about to fall? "
    "(0 = says regime is firmly in control; 10 = says regime is collapsing, beatable now)\n"
    "collective: how much collective 'we/us/together' framing vs purely individual? "
    "(0 = individual; 10 = strongly collective)\n"
    "Reply with ONLY a JSON object: {{\"beatable\": <int>, \"collective\": <int>}}\n\nMESSAGE:\n{msg}"
)


def clean(m): return PREFIX.sub("", m.strip().strip('"').strip()).strip().strip('"')


def cell_msgs(path, rng):
    d = json.loads(Path(path).read_text())
    out = {}
    for i, c in enumerate(d):
        seen = []
        if MODE == "live":
            cand = [ag.get("message_sent") for ag in c.get("agents", [])]
        else:
            cand = [m for ag in c.get("agents", []) for m in (ag.get("messages_received") or [])]
        for m in cand:
            cm = clean(m) if isinstance(m, str) else ""
            if cm and cm not in seen:
                seen.append(cm)
        if seen:
            rng.shuffle(seen)
            out[i] = {"theta": c["theta"], "join": c["join_fraction"], "msgs": seen[:MAX_MSGS]}
    return out


async def judge_one(client, msg, sem):
    async with sem:
        for k in range(4):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE, temperature=0,
                    messages=[{"role": "user", "content": PROMPT.format(msg=msg)}], max_tokens=40)
                mt = re.search(r"\{[^}]*\}", r.choices[0].message.content or "")
                if mt:
                    o = json.loads(mt.group(0))
                    return float(o["beatable"]), float(o["collective"])
            except Exception:
                await asyncio.sleep(1.5 * (k + 1))
        return np.nan, np.nan


async def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"scores_{SAFE}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    rng = np.random.default_rng(0)
    S, C = cell_msgs(SURV, rng), cell_msgs(COMM, rng)
    cells = [i for i in S if i in C][:N_CELLS]
    uniq = {m for i in cells for cond in (S, C) for m in cond[i]["msgs"]}
    todo = [m for m in uniq if hashlib.sha1(m.encode()).hexdigest() not in cache]
    print(f"{len(uniq)} unique messages, {len(todo)} to judge via {JUDGE} (mode={MODE}) ...", flush=True)
    if todo:
        key = os.environ.get("OPENROUTER_API_KEY") or next(
            (l.split("=", 1)[1].strip().strip('"').strip("'") for l in (ROOT / ".env").read_text().splitlines()
             if l.startswith("OPENROUTER_API_KEY=")), "")
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        sem = asyncio.Semaphore(40)
        scores = await asyncio.gather(*(judge_one(client, m, sem) for m in todo))
        for m, sc in zip(todo, scores):
            cache[hashlib.sha1(m.encode()).hexdigest()] = sc
        cache_path.write_text(json.dumps(cache))

    def look(m):
        return cache.get(hashlib.sha1(m.encode()).hexdigest(), (np.nan, np.nan))

    rec = []
    for i in cells:
        row = {"cell": i, "join_s": S[i]["join"], "join_c": C[i]["join"]}
        for cond, dd in (("s", S[i]), ("c", C[i])):
            b = [look(m)[0] for m in dd["msgs"]]
            cl = [look(m)[1] for m in dd["msgs"]]
            row[f"beat_{cond}"] = np.nanmean(b)
            row[f"coll_{cond}"] = np.nanmean(cl)
            row[f"cert_{cond}"] = np.nanmean([len(CERT.findall(m)) for m in dd["msgs"]])
            row[f"we_{cond}"] = np.nanmean([len(WE.findall(m)) for m in dd["msgs"]])
        rec.append(row)
    df = pd.DataFrame(rec).dropna()
    df["dj"] = df.join_s - df.join_c
    df["d"] = df.beat_s - df.beat_c
    fit = smf.ols("dj ~ d", df).fit()
    val_r = float(np.corrcoef(np.r_[df.beat_s, df.beat_c], np.r_[df.cert_s, df.cert_c])[0, 1])

    result = {
        "judge": JUDGE, "mode": MODE, "n_cells": int(len(df)),
        "beat_surv": float(df.beat_s.mean()), "beat_comm": float(df.beat_c.mean()),
        "beat_delta": float(df.beat_s.mean() - df.beat_c.mean()),
        "join_delta": float(df.dj.mean()),
        "med_slope": float(fit.params["d"]), "med_p": float(fit.pvalues["d"]),
        "med_share": float(fit.params["d"] * df.d.mean() / df.dj.mean()),
        "val_r_beat_cert": val_r,
    }
    (CACHE / f"result_{SAFE}_{MODE}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
