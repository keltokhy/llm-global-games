"""LLM-judge validation of the data-named mechanism.

A different-family judge (Qwen3-30B) scores each surveillance/comm source
message on two dimensions: (1) how confidently it asserts the regime is
beatable, (2) collective 'we' framing. We then (a) validate these against the
lexicon proxies and (b) test whether they mediate the replay join drop within
matched cells.
"""
import asyncio, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from openai import AsyncOpenAI

ROOT = Path("/Users/khaled/GitHub/llm-global-games")
SURV = ROOT / os.environ.get("SURV", "output/expB-surv-replay/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json")
COMM = ROOT / os.environ.get("COMM", "output/expB-comm-replay/meta-llama--llama-3.3-70b-instruct/experiment_comm_log.json")
JUDGE = os.environ.get("JUDGE", "qwen/qwen3-30b-a3b-instruct-2507")
N_CELLS = 200
MAX_MSGS = 6
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


MODE = os.environ.get("MODE", "replay")  # replay=messages_received, live=message_sent

def cell_msgs(path, rng):
    d = json.loads(Path(path).read_text())
    out = {}
    for i, c in enumerate(d):
        seen = []
        if MODE == "live":
            for ag in c.get("agents", []):
                m = ag.get("message_sent")
                cm = clean(m) if isinstance(m, str) else ""
                if cm and cm not in seen:
                    seen.append(cm)
        else:
            for ag in c.get("agents", []):
                for m in ag.get("messages_received", []) or []:
                    cm = clean(m) if isinstance(m, str) else ""
                    if cm and cm not in seen:
                        seen.append(cm)
        if seen:
            rng.shuffle(seen)
            out[i] = {"theta": c["theta"], "join": c["join_fraction"], "msgs": seen[:MAX_MSGS]}
    return out


async def judge(client, msg, sem):
    async with sem:
        for k in range(4):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE, temperature=0,
                    messages=[{"role": "user", "content": PROMPT.format(msg=msg)}], max_tokens=40)
                t = r.choices[0].message.content or ""
                mt = re.search(r"\{[^}]*\}", t)
                if mt:
                    o = json.loads(mt.group(0))
                    return float(o["beatable"]), float(o["collective"])
            except Exception:
                await asyncio.sleep(1.5 * (k + 1))
        return np.nan, np.nan


async def main():
    key = os.environ.get("OPENROUTER_API_KEY") or next(
        (l.split("=", 1)[1].strip().strip('"').strip("'") for l in (ROOT / ".env").read_text().splitlines()
         if l.startswith("OPENROUTER_API_KEY=")), "")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    sem = asyncio.Semaphore(40)
    rng = np.random.default_rng(0)
    S, C = cell_msgs(SURV, rng), cell_msgs(COMM, rng)
    cells = [i for i in S if i in C][:N_CELLS]

    # collect all (cell, cond, msg) jobs
    jobs = []
    for i in cells:
        for cond, dd in (("s", S[i]), ("c", C[i])):
            for m in dd["msgs"]:
                jobs.append((i, cond, m))
    print(f"judging {len(jobs)} messages across {len(cells)} matched cells via {JUDGE} ...", flush=True)
    res = await asyncio.gather(*(judge(client, m, sem) for _, _, m in jobs))

    rows = {}
    for (i, cond, m), (b, col) in zip(jobs, res):
        rows.setdefault((i, cond), []).append((b, col, len(CERT.findall(m)), len(WE.findall(m))))
    rec = []
    for i in cells:
        rcs = {}
        for cond in ("s", "c"):
            arr = np.array(rows[(i, cond)], dtype=float)
            rcs[cond] = np.nanmean(arr, axis=0)
        rec.append({"cell": i,
                    "beat_s": rcs["s"][0], "beat_c": rcs["c"][0],
                    "coll_s": rcs["s"][1], "coll_c": rcs["c"][1],
                    "cert_s": rcs["s"][2], "cert_c": rcs["c"][2],
                    "we_s": rcs["s"][3], "we_c": rcs["c"][3],
                    "join_s": S[i]["join"], "join_c": C[i]["join"]})
    df = pd.DataFrame(rec).dropna()
    print(f"\n=== judge means (surv vs comm), {len(df)} cells ===")
    print(f"beatable:   surv={df.beat_s.mean():.2f}  comm={df.beat_c.mean():.2f}  diff={df.beat_s.mean()-df.beat_c.mean():+.2f}")
    print(f"collective: surv={df.coll_s.mean():.2f}  comm={df.coll_c.mean():.2f}  diff={df.coll_s.mean()-df.coll_c.mean():+.2f}")
    print(f"join:       surv={df.join_s.mean():.3f}  comm={df.join_c.mean():.3f}  diff={df.join_s.mean()-df.join_c.mean():+.3f}")

    # validate judge vs lexicon (pool cell-level surv & comm)
    jb = np.r_[df.beat_s, df.beat_c]; ct = np.r_[df.cert_s, df.cert_c]
    jc = np.r_[df.coll_s, df.coll_c]; we = np.r_[df.we_s, df.we_c]
    print(f"\n=== validation: judge vs lexicon (cell-level corr) ===")
    print(f"judge beatable  ~ lexicon certainty: r={np.corrcoef(jb,ct)[0,1]:+.3f}")
    print(f"judge collective~ lexicon we:        r={np.corrcoef(jc,we)[0,1]:+.3f}")

    print(f"\n=== mediation within matched cells (Delta-join ~ Delta-judge) ===")
    df["dj"] = df.join_s - df.join_c
    for dim, a, b in [("beatable", "beat_s", "beat_c"), ("collective", "coll_s", "coll_c")]:
        df["d"] = df[a] - df[b]
        fit = smf.ols("dj ~ d", df).fit()
        print(f"{dim:11s}: surv-comm Delta={df.d.mean():+.3f}  slope(dj~d)={fit.params['d']:+.4f}  p={fit.pvalues['d']:.1e}")


if __name__ == "__main__":
    asyncio.run(main())
