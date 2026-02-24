"""
Render LaTeX tables for the paper from verified_stats.json.

This avoids manual copy/paste errors: the paper should `\\input{}` these files.

Usage:
    uv run python agent_based_simulation/render_paper_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATS_PATH = ROOT / "verified_stats.json"
OUT_DIR = ROOT / "tables"


def _fmt_num(x: float, nd: int = 3) -> str:
    if x is None:
        return "---"
    try:
        if x != x:  # nan
            return "---"
    except Exception:
        return "---"
    return f"{x:.{nd}f}"


def _fmt_r(x: float, nd: int = 2) -> str:
    if x is None:
        return "---"
    try:
        if x != x:
            return "---"
    except Exception:
        return "---"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{nd}f}"


def _fmt_mean(x: float, nd: int = 2) -> str:
    return _fmt_num(x, nd=nd)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load() -> dict:
    with open(STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def render_tab_models(stats: dict) -> str:
    part1 = stats["part1"]
    # Preserve paper order.
    models = [
        "Mistral Small Creative",
        "Llama 3.3 70B",
        "OLMo 3 7B",
        "Ministral 3B",
        "Qwen3 30B",
        "GPT-OSS 120B",
        "Qwen3 235B",
        "Trinity Large",
        "MiniMax M2-Her",
    ]
    arch = {
        "Mistral Small Creative": "Mistral",
        "Llama 3.3 70B": "Llama",
        "OLMo 3 7B": "OLMo",
        "Ministral 3B": "Mistral",
        "Qwen3 30B": "Qwen (MoE)",
        "GPT-OSS 120B": "GPT",
        "Qwen3 235B": "Qwen (MoE)",
        "Trinity Large": "Arcee",
        "MiniMax M2-Her": "MiniMax",
    }

    lines = []
    total_pure = 0
    total_comm = 0
    total_falsif = 0

    for m in models:
        entry = part1.get(m, {})
        pure_n = (entry.get("pure") or {}).get("n_obs")
        comm_n = (entry.get("comm") or {}).get("n_obs")
        scr_n = (entry.get("scramble") or {}).get("n_obs")
        flip_n = (entry.get("flip") or {}).get("n_obs")
        falsif_n = None
        if isinstance(scr_n, int) and isinstance(flip_n, int):
            falsif_n = scr_n + flip_n

        total_pure += int(pure_n or 0)
        total_comm += int(comm_n or 0)
        total_falsif += int(falsif_n or 0)

        falsif_cell = f"{falsif_n}" if falsif_n is not None else "---"
        lines.append(
            f"{m} & {arch.get(m,'')} & {pure_n} & {comm_n} & {falsif_cell} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Model summary. Columns report country-period counts in the pure, communication, and falsification (scramble+flip) suites. All runs use $N=25$ agents per period and $\sigma=0.3$.}
\label{tab:models}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccc}
\toprule
Model & Arch. & Pure & Comm & Falsif. \\
\midrule
"""
    tex += "\n".join(lines) + "\n"
    tex += r"""\midrule
\textbf{Total} & & \textbf{""" + f"{total_pure}" + r"""} & \textbf{""" + f"{total_comm}" + r"""} & \textbf{""" + f"{total_falsif}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_main_results(stats: dict) -> str:
    part1 = stats["part1"]
    models = [
        "Mistral Small Creative",
        "Llama 3.3 70B",
        "OLMo 3 7B",
        "Ministral 3B",
        "Qwen3 30B",
        "GPT-OSS 120B",
        "Qwen3 235B",
        "Trinity Large",
        "MiniMax M2-Her",
    ]

    def r_attack_val(m: str, t: str) -> float | None:
        d = part1.get(m, {}).get(t, {})
        if not isinstance(d, dict):
            return None
        return (d.get("r_vs_attack") or {}).get("r")

    def r_attack(m: str, t: str) -> str:
        r = r_attack_val(m, t)
        if r is None:
            return r"$\text{---}$"
        return f"${_fmt_r(r, nd=2)}$"

    def mean_r_attack(t: str) -> float | None:
        vals: list[float] = []
        for m in models:
            r = r_attack_val(m, t)
            if r is not None:
                vals.append(float(r))
        if not vals:
            return None
        return sum(vals) / len(vals)

    def r_cell(r: float | None, nd: int = 2) -> str:
        if r is None:
            return r"$\text{---}$"
        return f"${_fmt_r(float(r), nd=nd)}$"

    def mean_join(m: str) -> str:
        d = part1.get(m, {}).get("pure", {})
        return _fmt_mean(d.get("mean_join"), nd=2)

    def n_pure(m: str) -> str:
        d = part1.get(m, {}).get("pure", {})
        return str(d.get("n_obs") or "---")

    rows = []
    for m in models:
        rows.append(
            f"{m} & {r_attack(m,'pure')} & {r_attack(m,'comm')} & {r_attack(m,'scramble')} & {r_attack(m,'flip')} & {n_pure(m)} & {mean_join(m)} \\\\"
        )

    pooled = part1.get("_pooled_pure", {}).get("r_vs_attack", {}).get("r")
    pooled_comm = part1.get("_pooled_comm", {}).get("r_vs_attack", {}).get("r")
    pooled_scr = part1.get("_pooled_scramble", {}).get("r_vs_attack", {}).get("r")
    pooled_flip = part1.get("_pooled_flip", {}).get("r_vs_attack", {}).get("r")
    pooled_n = part1.get("_pooled_pure", {}).get("n_obs")
    pooled_mean = part1.get("_pooled_pure", {}).get("mean_join")

    mean_pure = part1.get("_mean_r_pure_vs_attack")
    mean_comm = mean_r_attack("comm")
    mean_scr = mean_r_attack("scramble")
    mean_flip = mean_r_attack("flip")

    tex = r"""\begin{table*}[t]
\centering
\caption{Equilibrium alignment by model and treatment. Cells report Pearson $r$ between the empirical join fraction and the theoretical attack mass $A(\theta)$.}
\label{tab:main_results}
\small
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Main treatments} & \multicolumn{2}{c}{Falsification} & & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Model & Pure & Comm & Scramble & Flip & $n_{\text{pure}}$ & Mean join \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\midrule
\textbf{Pooled} & $""" + _fmt_r(pooled, 2) + r"""$ & $""" + _fmt_r(pooled_comm, 2) + r"""$ & $""" + _fmt_r(pooled_scr, 2) + r"""$ & $""" + _fmt_r(pooled_flip, 2) + r"""$ & """ + f"{pooled_n}" + r""" & """ + _fmt_mean(pooled_mean, 2) + r""" \\
\textbf{Mean across models} & """ + r_cell(mean_pure, 2) + r""" & """ + r_cell(mean_comm, 2) + r""" & """ + r_cell(mean_scr, 2) + r""" & """ + r_cell(mean_flip, 2) + r""" & --- & --- \\
\bottomrule
\end{tabular}
\end{table*}
"""
    return tex


def render_tab_infodesign(stats: dict) -> str:
    info = stats["infodesign"]
    designs = [
        ("baseline", "Baseline"),
        ("stability", "Stability"),
        ("instability", "Instability"),
        ("public_signal", "Public signal"),
        ("scramble", "Scramble"),
        ("flip", "Flip"),
    ]

    rows = []
    for key, label in designs:
        d = info.get(key, {})
        mean = d.get("mean_join")
        r = (d.get("r_vs_theta") or {}).get("r")
        delta = d.get("delta_vs_baseline")
        n = d.get("n_obs")
        delta_cell = "---" if delta is None else _fmt_r(delta, nd=3).replace("+", "+")
        rows.append(
            f"{label} & {_fmt_num(mean,3)} & ${_fmt_r(r,3)}$ & {delta_cell} & {n} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Information design treatment summary (primary model: Mistral Small Creative). $r$ is the Pearson correlation between $\theta$ and join fraction.}
\label{tab:infodesign_summary}
\small
\begin{tabular}{lcccc}
\toprule
Design & Mean & $r$ & $\Delta$ & $N$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_surveillance_propaganda(stats: dict) -> str:
    part1 = stats["part1"]
    regime = stats["regime_control"]

    # Baseline comm (Mistral) from Part I
    base = part1["Mistral Small Creative"]["comm"]
    base_mean = base["mean_join"]
    base_r = base["r_vs_theta"]["r"]

    def row_prop(k: int) -> tuple[str, dict]:
        d = regime["propaganda"][f"k={k}"]["Mistral Small Creative"]
        return f"Prop $k={k}$", d

    prop_rows = [row_prop(2), row_prop(5), row_prop(10)]
    surv = regime["surveillance"]["Mistral Small Creative"]
    ps = regime["propaganda_surveillance"]["Mistral Small Creative"]

    lines = []
    lines.append(f"Comm (baseline) & {_fmt_num(base_mean,3).lstrip('0')} & {_fmt_num(base_mean,3).lstrip('0')} & ${_fmt_r(base_r,3)}$ & --- \\\\")
    lines.append(r"\midrule")

    for label, d in prop_rows:
        mean_all = d["mean_join_all"]
        mean_real = d.get("mean_join_real")
        r = d["r_vs_theta_all"]["r"]
        delta_real = d.get("delta_real_vs_baseline_pp")
        delta_cell = "---" if delta_real is None else f"{_fmt_r(delta_real/100,3)}"
        lines.append(
            f"{label} & {_fmt_num(mean_all,3).lstrip('0')} & {_fmt_num(mean_real,3).lstrip('0')} & ${_fmt_r(r,3)}$ & {delta_cell} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(
        f"Surveillance & {_fmt_num(surv['mean_join'],3).lstrip('0')} & {_fmt_num(surv['mean_join'],3).lstrip('0')} & ${_fmt_r(surv['r_vs_theta']['r'],3)}$ & {_fmt_r(surv['delta_vs_baseline_pp']/100,3)} \\\\"
    )
    lines.append(
        f"Prop+Surv & {_fmt_num(ps['mean_join_all'],3).lstrip('0')} & --- & ${_fmt_r(ps['r_vs_theta_all']['r'],3)}$ & --- \\\\"
    )

    tex = r"""\begin{table}[t]
\centering
\caption{Propaganda and surveillance effects (primary model: Mistral Small Creative). ``All'' includes propaganda agents; ``Real'' excludes them (computed from logs). $\Delta$ is the change in real-agent mean join vs.\ baseline communication.}
\label{tab:surveillance_propaganda}
\small
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Mean join} & & \\
\cmidrule(lr){2-3}
Treatment & All & Real & $r$ & $\Delta$ \\
\midrule
"""
    tex += "\n".join(lines) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_surv_censor(stats: dict) -> str:
    info = stats["infodesign"]
    sxc = stats["regime_control"]["surveillance_x_censorship"]["Mistral Small Creative"]

    def m(key: str) -> float:
        return float(info[key]["mean_join"])

    baseline = m("baseline")
    up = m("censor_upper")
    lo = m("censor_lower")

    lines = []
    for label, key in [("Baseline", "baseline"), ("Upper cens.", "censor_upper"), ("Lower cens.", "censor_lower")]:
        no = {"baseline": baseline, "censor_upper": up, "censor_lower": lo}[key]
        yes = float(sxc[key])
        delta = yes - no
        lines.append(f"{label} & {_fmt_num(no,3)} & {_fmt_num(yes,3)} & {_fmt_r(delta,nd=3)} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Surveillance $\times$ censorship interaction (primary model: Mistral Small Creative).}
\label{tab:surv_censor}
\small
\begin{tabular}{lccc}
\toprule
Design & No Surv. & Surv. & $\Delta$ \\
\midrule
"""
    tex += "\n".join(lines) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_bandwidth(stats: dict) -> str:
    info = stats["infodesign"]
    bw = stats["robustness"]["bandwidth"]

    # Baseline bandwidth=0.15 is the main infodesign run.
    def main_mean(design: str) -> float:
        return float(info[design]["mean_join"])

    b005 = bw["bandwidth-005"]["Mistral Small Creative"]
    b030 = bw["bandwidth-030"]["Mistral Small Creative"]

    rows = []
    for design, label in [
        ("baseline", "Baseline"),
        ("stability", "Stability"),
        ("censor_upper", "Upper cens."),
        ("censor_lower", "Lower cens."),
    ]:
        v005 = float(b005[design])
        v015 = main_mean(design)
        v030 = float(b030[design])
        rows.append(f"{label} & {_fmt_num(v005,3)} & {_fmt_num(v015,3)} & {_fmt_num(v030,3)} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Bandwidth robustness: mean join rates (primary model: Mistral Small Creative).}
\label{tab:bandwidth}
\small
\begin{tabular}{lccc}
\toprule
Design & BW=0.05 & BW=0.15 & BW=0.30 \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_crossmodel(stats: dict) -> str:
    cross = stats["infodesign"].get("_cross_model", {})
    order = [
        "Mistral Small Creative",
        "GPT-OSS 120B",
        "Llama 3.3 70B",
        "Ministral 3B",
        "Qwen3 30B",
        "Qwen3 235B",
        "OLMo 3 7B",
    ]

    def cell(model: str, design: str, field: str):
        d = cross.get(model, {}).get(design, {})
        if not d:
            return "---"
        if field == "mean":
            return _fmt_num(d.get("mean_join"), 3)
        if field == "r":
            return f"${_fmt_r((d.get('r_vs_theta') or {}).get('r'), 3)}$"
        return "---"

    rows = []
    for model in order:
        rows.append(
            f"{model} & {cell(model,'baseline','mean')} & {cell(model,'baseline','r')} & "
            f"{cell(model,'scramble','mean')} & {cell(model,'scramble','r')} & "
            f"{cell(model,'flip','mean')} & {cell(model,'flip','r')} \\\\"
        )

    tex = r"""\begin{table*}[t]
\centering
\caption{Cross-model replication of key information design conditions. $r$ is the correlation between $\theta$ and join fraction.}
\label{tab:crossmodel}
\small
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Baseline} & \multicolumn{2}{c}{Scramble} & \multicolumn{2}{c}{Flip} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Model & Mean & $r$ & Mean & $r$ & Mean & $r$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    return tex


def render_tab_decomposition(stats: dict) -> str:
    info = stats["infodesign"]
    rows = []
    for key, label in [
        ("stability", "Full stability"),
        ("stability_clarity", "Clarity only"),
        ("stability_direction", "Direction only"),
        ("stability_dissent", "Dissent only"),
    ]:
        d = info.get(key, {})
        mean = d.get("mean_join")
        r = (d.get("r_vs_theta") or {}).get("r")
        delta = d.get("delta_vs_baseline")
        rows.append(f"{label} & {_fmt_num(mean,3)} & ${_fmt_r(r,3)}$ & {_fmt_r(delta,3)} \\\\")

    # Sum of single-channel deltas vs full delta
    deltas = [info.get(k, {}).get("delta_vs_baseline") for k in ["stability_clarity", "stability_direction", "stability_dissent"]]
    sum_delta = sum(float(x) for x in deltas if x is not None)
    full_delta = float(info["stability"]["delta_vs_baseline"])

    rows.append(r"\midrule")
    rows.append(f"Sum of channels & --- & --- & {_fmt_r(sum_delta,3)} \\\\")
    rows.append(f"Full design & --- & --- & {_fmt_r(full_delta,3)} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Single-channel decomposition of the stability design (primary model: Mistral Small Creative).}
\label{tab:decomposition}
\small
\begin{tabular}{lccc}
\toprule
Channel & Mean & $r$ & $\Delta$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def main() -> None:
    stats = _load()

    tables = {
        "tab_models.tex": render_tab_models(stats),
        "tab_main_results.tex": render_tab_main_results(stats),
        "tab_infodesign_summary.tex": render_tab_infodesign(stats),
        "tab_surveillance_propaganda.tex": render_tab_surveillance_propaganda(stats),
        "tab_surv_censor.tex": render_tab_surv_censor(stats),
        "tab_bandwidth.tex": render_tab_bandwidth(stats),
        "tab_crossmodel.tex": render_tab_crossmodel(stats),
        "tab_decomposition.tex": render_tab_decomposition(stats),
    }

    for name, content in tables.items():
        _write(OUT_DIR / name, content)

    print(f"Wrote {len(tables)} table(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
