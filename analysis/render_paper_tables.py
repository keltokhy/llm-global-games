"""
Render LaTeX tables for the paper from verified_stats.json.

This avoids manual copy/paste errors: the paper should `\\input{}` these files.

Usage:
    uv run python agent_based_simulation/render_paper_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
STATS_PATH = ANALYSIS_DIR / "verified_stats.json"
OUT_DIR = ANALYSIS_DIR.parent / "paper" / "tables"


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

    def r_attack_ci(m: str, t: str) -> tuple[float | None, float | None]:
        d = part1.get(m, {}).get(t, {})
        if not isinstance(d, dict):
            return None, None
        ra = d.get("r_vs_attack") or {}
        return ra.get("ci_lo"), ra.get("ci_hi")

    def r_attack_with_ci(m: str, t: str) -> str:
        r = r_attack_val(m, t)
        if r is None:
            return r"$\text{---}$"
        ci_lo, ci_hi = r_attack_ci(m, t)
        if ci_lo is not None and ci_hi is not None:
            return f"${_fmt_r(r, 2)}$ {{\\scriptsize $[{ci_lo:.2f},{ci_hi:.2f}]$}}"
        return f"${_fmt_r(r, 2)}$"

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
            f"{m} & {r_attack_with_ci(m,'pure')} & {r_attack_with_ci(m,'comm')} & {r_attack(m,'scramble')} & {r_attack(m,'flip')} & {n_pure(m)} & {mean_join(m)} \\\\"
        )

    pooled = part1.get("_pooled_pure", {}).get("r_vs_attack", {}).get("r")
    pooled_comm = part1.get("_pooled_comm", {}).get("r_vs_attack", {}).get("r")
    pooled_scr = part1.get("_pooled_scramble", {}).get("r_vs_attack", {}).get("r")
    pooled_flip = part1.get("_pooled_flip", {}).get("r_vs_attack", {}).get("r")
    pooled_n = part1.get("_pooled_pure", {}).get("n_obs")
    pooled_mean = part1.get("_pooled_pure", {}).get("mean_join")

    pooled_pure_ci = part1.get("_pooled_pure", {}).get("r_vs_attack", {})
    pooled_comm_ci = part1.get("_pooled_comm", {}).get("r_vs_attack", {})

    mean_pure = part1.get("_mean_r_pure_vs_attack")
    mean_comm = mean_r_attack("comm")
    mean_scr = mean_r_attack("scramble")
    mean_flip = mean_r_attack("flip")

    tex = r"""\begin{table*}[t]
\centering
\caption{Equilibrium alignment by model and treatment. Cells report Pearson $r$ between the empirical join fraction and the theoretical attack mass $A(\theta)$; 95\% Fisher-$z$ confidence intervals in brackets for main treatments.}
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
    # Build pooled cells with CIs for main treatments (Pure, Comm)
    def _r_cell_with_ci(r_val, ci_dict):
        cell = f"${_fmt_r(r_val, 2)}$"
        ci_lo, ci_hi = ci_dict.get("ci_lo"), ci_dict.get("ci_hi")
        if ci_lo is not None and ci_hi is not None:
            cell += f" {{\\scriptsize $[{ci_lo:.2f},{ci_hi:.2f}]$}}"
        return cell

    pooled_pure_cell = _r_cell_with_ci(pooled, pooled_pure_ci)
    pooled_comm_cell = _r_cell_with_ci(pooled_comm, pooled_comm_ci)

    tex += r"""\midrule
\textbf{Pooled} & """ + pooled_pure_cell + r""" & """ + pooled_comm_cell + r""" & $""" + _fmt_r(pooled_scr, 2) + r"""$ & $""" + _fmt_r(pooled_flip, 2) + r"""$ & """ + f"{pooled_n}" + r""" & """ + _fmt_mean(pooled_mean, 2) + r""" \\
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
\vspace{0.25em}
\footnotesize\emph{Notes:} Data from \texttt{output/mistralai--mistral-small-creative/experiment\_infodesign\_\{design\}\_summary.csv} (pure treatment; $\theta \in [0.20, 0.80]$ on a 9-point grid; $N{=}25$ agents per period). Mean join uses \texttt{join\_fraction\_valid}; $r$ is Pearson $r(\theta,\text{join})$ across rep-level periods.
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
    info_comm = stats.get("infodesign_comm") or {}
    sxc = stats["regime_control"]["surveillance_x_censorship"]["Mistral Small Creative"]

    def m(key: str) -> float:
        d = info_comm.get(key, {})
        if not d:
            raise KeyError(
                f"Missing infodesign_comm['{key}'] in verified_stats.json. "
                "Re-run: uv run python analysis/verify_paper_stats.py"
            )
        return float(d["mean_join"])

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
\caption{Surveillance $\times$ censorship interaction in the communication game (primary model: Mistral Small Creative).}
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
\vspace{0.25em}
\footnotesize\emph{Notes:} ``No Surv.'' uses the communication infodesign grid (\texttt{output/mistralai--mistral-small-creative-infodesign-comm/}.) ``Surv.'' uses the same grid with surveillance active during messaging (\texttt{output/surveillance-x-censorship/}.) All entries are means of \texttt{join\_fraction\_valid}.
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
\vspace{0.25em}
\footnotesize\emph{Notes:} Each row is a separate infodesign run for Mistral Small Creative on the same $\theta$ grid as Table~\ref{tab:infodesign_summary}. $\Delta$ reports the mean difference vs.\ the baseline infodesign mean (Table~\ref{tab:infodesign_summary}).
\end{table}
"""
    return tex


def render_tab_uncalibrated(stats: dict) -> str:
    uncal = stats.get("uncalibrated", {})
    if not uncal:
        return "% No uncalibrated robustness data available.\n"

    models = [
        "Mistral Small Creative",
        "Llama 3.3 70B",
        "Qwen3 235B",
    ]

    rows = []
    for m in models:
        d = uncal.get(m, {})
        if not d or d.get("status") == "missing":
            rows.append(f"{m} & --- & --- & --- & --- \\\\")
            continue
        n = d.get("n_obs", "---")
        mean_j = _fmt_num(d.get("mean_join"), 3)
        r = (d.get("r_vs_theta") or {}).get("r")
        p = (d.get("r_vs_theta") or {}).get("p")
        r_cell = f"${_fmt_r(r, 3)}$" if r is not None else "---"
        p_cell = _fmt_num(p, 4) if p is not None else "---"
        rows.append(f"{m} & {n} & {mean_j} & {r_cell} & {p_cell} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Uncalibrated robustness: models run without calibration adjustment. $r$ is the Pearson correlation between $\theta$ and join fraction.}
\label{tab:uncalibrated}
\small
\begin{tabular}{lcccc}
\toprule
Model & $N$ & Mean join & $r(\theta, J)$ & $p$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_surv_censor_crossmodel(stats: dict) -> str:
    sxc = stats.get("regime_control", {}).get("surveillance_x_censorship", {})
    if not sxc:
        return "% No cross-model surveillance x censorship data available.\n"

    models = [
        "Mistral Small Creative",
        "Llama 3.3 70B",
        "GPT-OSS 120B",
        "Qwen3 235B",
    ]

    rows = []
    for m in models:
        d = sxc.get(m, {})
        if not d:
            rows.append(f"{m} & --- & --- & --- & --- & --- \\\\")
            continue
        bl = d.get("baseline")
        cu = d.get("censor_upper")
        cl = d.get("censor_lower")
        bl_cell = _fmt_num(bl, 3)
        cu_cell = _fmt_num(cu, 3)
        cl_cell = _fmt_num(cl, 3)
        delta_u = (cu - bl) if bl is not None and cu is not None else None
        delta_l = (cl - bl) if bl is not None and cl is not None else None
        du_cell = _fmt_r(delta_u, 3) if delta_u is not None else "---"
        dl_cell = _fmt_r(delta_l, 3) if delta_l is not None else "---"
        rows.append(f"{m} & {bl_cell} & {cu_cell} & {cl_cell} & {du_cell} & {dl_cell} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Cross-model surveillance $\times$ censorship interaction. All conditions run under surveillance with communication. $\Delta$ columns show the change relative to the surveilled baseline.}
\label{tab:surv_censor_crossmodel}
\small
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{Mean join (surv.)} & \multicolumn{2}{c}{$\Delta$ vs baseline} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-6}
Model & Baseline & Upper cens. & Lower cens. & $\Delta$ upper & $\Delta$ lower \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_logistic_params(stats: dict) -> str:
    fits = stats.get("logistic_fits", {})
    if not fits:
        return "% No logistic fit data available.\n"

    MODEL_ORDER = [
        "Mistral Small Creative",
        "Llama 3.3 70B",
        "Ministral 3B",
        "Qwen3 30B",
        "GPT-OSS 120B",
        "Qwen3 235B",
        "Trinity Large",
        "MiniMax M2-Her",
    ]

    def _cell(fit: dict | None, key: str, se_key: str) -> str:
        if fit is None:
            return "---"
        val = fit.get(key)
        se = fit.get(se_key)
        if val is None or (isinstance(val, float) and (val != val or abs(val) > 50)):
            return "---"
        sign = "+" if val >= 0 else ""
        if se is not None and isinstance(se, float) and se == se and se < 50:
            return f"${sign}{val:.2f}$ ({se:.2f})"
        return f"${sign}{val:.2f}$"

    rows = []
    for m in MODEL_ORDER:
        if m not in fits:
            continue
        p = fits[m].get("pure")
        c = fits[m].get("comm")
        rows.append(
            f"{m} & {_cell(p, 'cutoff', 'se_cutoff')} & {_cell(p, 'b1', 'se_b1')}"
            f" & {_cell(c, 'cutoff', 'se_cutoff')} & {_cell(c, 'b1', 'se_b1')} \\\\"
        )

    tex = r"""\begin{table*}[t]
\centering
\caption{Logistic fit parameters by model and treatment. $\hat{\theta}^*$ is the estimated cutoff ($-b_0/b_1$); $\beta$ is the logistic slope. Standard errors from the covariance matrix of the nonlinear fit; cutoff SE by delta method.}
\label{tab:logistic_params}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Pure} & \multicolumn{2}{c}{Communication} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Model & $\hat{\theta}^*$ (SE) & $\beta$ (SE) & $\hat{\theta}^*$ (SE) & $\beta$ (SE) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    return tex


def render_tab_surveillance_variants(stats: dict) -> str:
    sv = stats.get("surveillance_variants", {})
    if not sv:
        return "% No surveillance variant data available.\n"

    rows = []
    for variant in ["placebo", "anonymous"]:
        d = sv.get(variant, {})
        if "mean_join" not in d:
            continue
        mean_j = _fmt_num(d["mean_join"], 3)
        r_val = _fmt_r(d["r_vs_theta"]["r"], 2) if "r_vs_theta" in d else "---"
        delta = _fmt_r(d.get("delta_vs_comm_pp", 0) / 100, 3) if d.get("delta_vs_comm_pp") is not None else "---"
        t_test = d.get("t_test_vs_comm", {})
        p_val = _fmt_num(t_test.get("p_value"), 3) if t_test else "---"
        label = "Placebo" if variant == "placebo" else "Anonymous"
        rows.append(f"{label} & {d['n_obs']} & {mean_j} & {r_val} & {delta} & {p_val} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Surveillance isolation checks. Placebo: monitored for research, no consequences. Anonymous: messages aggregated anonymously. Neither deviates significantly from the communication baseline.}
\label{tab:surveillance_variants}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Variant & $N$ & Mean join & $r(\theta, J)$ & $\Delta$ & $p$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_bc_statics(stats: dict) -> str:
    """B/C comparative statics: cutoff shifts under cost/benefit narratives."""
    info = stats.get("infodesign", {})
    designs = ["baseline", "bc_high_cost", "bc_low_cost"]
    labels = {"baseline": "Baseline", "bc_high_cost": "High cost", "bc_low_cost": "Low cost"}

    rows = []
    for d in designs:
        di = info.get(d, {})
        if not di:
            continue
        mean_j = _fmt_num(di["mean_join"], 3)
        r_val = _fmt_r(di["r_vs_theta"]["r"], 2) if "r_vs_theta" in di else "---"
        fit = di.get("logistic_fit") or {}
        cutoff = fit.get("cutoff")
        se_cutoff = fit.get("se_cutoff")
        cutoff_cell = "---"
        if cutoff is not None and se_cutoff is not None:
            cutoff_cell = f"{cutoff:.2f} ({se_cutoff:.3f})"
        n = di.get("n_obs", "---")
        delta = ""
        if "delta_vs_baseline" in di:
            delta = _fmt_r(di["delta_vs_baseline"], 3)
        elif d == "baseline":
            delta = "---"
        rows.append(f"{labels[d]} & {n} & {mean_j} & {r_val} & {cutoff_cell} & {delta} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Cost/benefit narrative comparative statics. High cost: narrative emphasizes severe reprisals for failed action. Low cost: narrative emphasizes minimal consequences. Theory predicts higher perceived cost lowers the cutoff (less joining).}
\label{tab:bc_statics}
\small
\begin{tabular}{lccccc}
\toprule
Design & $N$ & Mean join & $r(\theta, J)$ & Cutoff $\hat{\theta}^*$ (SE) & $\Delta$ vs baseline \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_censor_ck(stats: dict) -> str:
    """Censorship with common knowledge comparison."""
    info = stats.get("infodesign", {})
    designs = ["baseline", "censor_upper", "censor_upper_known"]
    labels = {
        "baseline": "Baseline (no censorship)",
        "censor_upper": "Upper censorship (na\\\"ive)",
        "censor_upper_known": "Upper censorship (known)",
    }

    rows = []
    for d in designs:
        di = info.get(d, {})
        if not di:
            continue
        mean_j = _fmt_num(di["mean_join"], 3)
        r_val = _fmt_r(di["r_vs_theta"]["r"], 2) if "r_vs_theta" in di else "---"
        n = di.get("n_obs", "---")
        delta = ""
        if "delta_vs_baseline" in di:
            delta = _fmt_r(di["delta_vs_baseline"], 3)
        elif d == "baseline":
            delta = "---"
        rows.append(f"{labels[d]} & {n} & {mean_j} & {r_val} & {delta} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Censorship with and without common knowledge. Na\"ive: agents do not know censorship is active. Known: agents are told that regime censors suppress unfavorable intelligence above a severity threshold.}
\label{tab:censor_ck}
\small
\begin{tabular}{lcccc}
\toprule
Design & $N$ & Mean join & $r(\theta, J)$ & $\Delta$ vs baseline \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_temperature(stats: dict) -> str:
    """Temperature robustness table."""
    temp = stats.get("temperature_robustness", {})
    if not temp:
        return "% No temperature robustness data available.\n"

    rows = []
    for key in ["T=0.3", "T=0.7", "T=1.0"]:
        d = temp.get(key, {})
        if not d:
            continue
        mean_j = _fmt_num(d["mean_join"], 3)
        r_val = _fmt_r(d["r_vs_theta"]["r"], 2) if "r_vs_theta" in d else "---"
        n = d.get("n_obs", "---")
        fit = d.get("logistic_fit", {})
        cutoff = _fmt_num(fit.get("cutoff"), 3) if fit.get("cutoff") is not None else "---"
        slope = _fmt_num(fit.get("b1"), 2) if fit.get("b1") is not None else "---"
        rows.append(f"{key} & {n} & {mean_j} & {r_val} & {cutoff} & {slope} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Temperature robustness. The pure global game is run at three LLM decoding temperatures using Mistral Small Creative with calibrated parameters. The correlation $r(\theta, J)$ and logistic parameters are stable across temperatures.}
\label{tab:temperature}
\small
\begin{tabular}{lccccc}
\toprule
Temperature & $N$ & Mean join & $r(\theta, J)$ & Cutoff $\hat{\theta}^*$ & Slope $\hat{\beta}$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def _fmt_pct(x: float, nd: int = 1) -> str:
    """Format a fraction as a percent string for LaTeX."""
    if x is None:
        return "---"
    try:
        if x != x:  # nan
            return "---"
    except Exception:
        return "---"
    return f"{x * 100:.{nd}f}\\%"


def _fmt_pp(x: float, nd: int = 1) -> str:
    """Format a fraction difference as signed percentage points for LaTeX."""
    if x is None:
        return "---"
    try:
        if x != x:  # nan
            return "---"
    except Exception:
        return "---"
    val = x * 100
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{nd}f}"


def render_stats_macros(stats: dict) -> str:
    """Render LaTeX macros for key stats used in the paper text.

    Motivation: avoid copy/paste inconsistencies between text, tables, and figures.
    This file is meant to be `\\input{tables/stats_macros.tex}` near the top of
    each TeX document.
    """
    info = stats.get("infodesign", {})
    part1 = stats.get("part1", {})
    regime = stats.get("regime_control", {})

    def ig(design: str, field: str, default=None):
        return (info.get(design) or {}).get(field, default)

    # Communication: mean across models (equal-weight) + pooled unpaired.
    model_entries = [
        v for k, v in part1.items()
        if isinstance(k, str) and not k.startswith("_") and isinstance(v, dict)
    ]
    pure_means = [((m.get("pure") or {}).get("mean_join")) for m in model_entries]
    comm_means = [((m.get("comm") or {}).get("mean_join")) for m in model_entries]
    pure_means = [x for x in pure_means if x is not None]
    comm_means = [x for x in comm_means if x is not None]
    mean_pure_models = sum(pure_means) / len(pure_means) if pure_means else None
    mean_comm_models = sum(comm_means) / len(comm_means) if comm_means else None
    delta_models = (mean_comm_models - mean_pure_models) if (mean_pure_models is not None and mean_comm_models is not None) else None

    pooled_pure = (part1.get("_pooled_pure") or {}).get("mean_join")
    pooled_comm = (part1.get("_pooled_comm") or {}).get("mean_join")
    pooled_delta_pp = ((part1.get("_pooled_comm_effect") or {}).get("unpaired") or {}).get("delta_pp")
    pooled_pval = ((part1.get("_pooled_comm_effect") or {}).get("unpaired") or {}).get("p_value")

    # Surveillance × censorship: primary model (Mistral) join levels.
    sxc = ((regime.get("surveillance_x_censorship") or {}).get("Mistral Small Creative") or {})

    lines = []
    lines.append("% Auto-generated from analysis/verified_stats.json. Do not edit by hand.")
    lines.append("% Generated by: uv run python analysis/render_paper_tables.py")
    lines.append("")

    # Communication summary
    lines.append(r"\providecommand{\CommPureMeanModelAvg}{" + _fmt_num(mean_pure_models, 3) + "}")
    lines.append(r"\providecommand{\CommCommMeanModelAvg}{" + _fmt_num(mean_comm_models, 3) + "}")
    lines.append(r"\providecommand{\CommDeltaPPModelAvg}{" + _fmt_pp(delta_models, 1) + "}")
    lines.append(r"\providecommand{\CommPureMeanPooled}{" + _fmt_num(pooled_pure, 3) + "}")
    lines.append(r"\providecommand{\CommCommMeanPooled}{" + _fmt_num(pooled_comm, 3) + "}")
    if pooled_delta_pp is None:
        lines.append(r"\providecommand{\CommDeltaPPPooled}{---}")
    else:
        sign = "+" if pooled_delta_pp >= 0 else ""
        lines.append(r"\providecommand{\CommDeltaPPPooled}{" + f"{sign}{pooled_delta_pp:.2f}" + "}")
    lines.append(r"\providecommand{\CommPValueUnpaired}{" + _fmt_num(pooled_pval, 3) + "}")
    lines.append("")

    # Infodesign summary (primary model)
    for key, macro in [
        ("baseline", "InfodesignBaseline"),
        ("stability", "InfodesignStability"),
        ("instability", "InfodesignInstability"),
        ("public_signal", "InfodesignPublicSignal"),
        ("scramble", "InfodesignScramble"),
        ("flip", "InfodesignFlip"),
        ("censor_upper", "InfodesignCensorUpper"),
        ("censor_lower", "InfodesignCensorLower"),
        ("stability_clarity", "DecompClarityOnly"),
        ("stability_direction", "DecompDirectionOnly"),
        ("stability_dissent", "DecompDissentOnly"),
    ]:
        mean = ig(key, "mean_join")
        r = ((info.get(key) or {}).get("r_vs_theta") or {}).get("r")
        delta = ig(key, "delta_vs_baseline")
        lines.append(f"\\providecommand{{\\{macro}Mean}}{{{_fmt_num(mean, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}MeanPct}}{{{_fmt_pct(mean, 1)}}}")
        lines.append(f"\\providecommand{{\\{macro}RTheta}}{{{_fmt_r(r, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}DeltaPP}}{{{_fmt_pp(delta, 1)}}}")
        lines.append("")

    # Decomposition summary: sum of single-channel deltas vs full bundled delta
    decomp_keys = ["stability_clarity", "stability_direction", "stability_dissent"]
    sum_delta = 0.0
    for k in decomp_keys:
        d = ig(k, "delta_vs_baseline")
        if d is not None:
            sum_delta += float(d)
    full_delta = ig("stability", "delta_vs_baseline")
    lines.append(r"\providecommand{\DecompSumChannelsDeltaPP}{" + _fmt_pp(sum_delta, 1) + "}")
    lines.append(r"\providecommand{\DecompFullDeltaPP}{" + _fmt_pp(full_delta, 1) + "}")
    lines.append("")

    # Within-briefing falsification (observation shuffle, domain scrambles)
    for key, macro in [
        ("within_scramble", "WithinScramble"),
        ("domain_scramble_coord", "DomainScrambleCoord"),
        ("domain_scramble_state", "DomainScrambleState"),
    ]:
        mean = ig(key, "mean_join")
        r = ((info.get(key) or {}).get("r_vs_theta") or {}).get("r")
        delta = ig(key, "delta_vs_baseline")
        lines.append(f"\\providecommand{{\\{macro}Mean}}{{{_fmt_num(mean, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}MeanPct}}{{{_fmt_pct(mean, 1)}}}")
        lines.append(f"\\providecommand{{\\{macro}RTheta}}{{{_fmt_r(r, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}DeltaPP}}{{{_fmt_pp(delta, 1)}}}")
    lines.append("")

    # Surveillance × censorship levels (primary model)
    for dname, macro in [("baseline", "SXCBase"), ("censor_upper", "SXCUpper"), ("censor_lower", "SXCLower")]:
        surv_mean = sxc.get(dname)
        lines.append(f"\\providecommand{{\\{macro}SurvMean}}{{{_fmt_num(surv_mean, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}SurvMeanPct}}{{{_fmt_pct(surv_mean, 1)}}}")
    lines.append("")

    return "\n".join(lines) + "\n"


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
        "tab_uncalibrated.tex": render_tab_uncalibrated(stats),
        "tab_surv_censor_crossmodel.tex": render_tab_surv_censor_crossmodel(stats),
        "tab_logistic_params.tex": render_tab_logistic_params(stats),
        "tab_surveillance_variants.tex": render_tab_surveillance_variants(stats),
        "tab_bc_statics.tex": render_tab_bc_statics(stats),
        "tab_censor_ck.tex": render_tab_censor_ck(stats),
        "tab_temperature.tex": render_tab_temperature(stats),
        "stats_macros.tex": render_stats_macros(stats),
    }

    for name, content in tables.items():
        _write(OUT_DIR / name, content)

    print(f"Wrote {len(tables)} table(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
