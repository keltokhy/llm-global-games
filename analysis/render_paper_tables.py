"""
Render LaTeX tables for the paper from verified_stats.json.

This avoids manual copy/paste errors: the paper should `\\input{}` these files.

Usage:
    uv run python agent_based_simulation/render_paper_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

from models import DISPLAY_ORDER, DISPLAY_NAMES, PART1_SLUGS


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
    models = DISPLAY_ORDER
    arch = {
        "Mistral Small Creative": "Mistral",
        "Llama 3.3 70B": "Llama",
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
    models = DISPLAY_ORDER

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
\caption{Threshold-policy alignment by model and treatment. Cells report Pearson $r$ between the empirical join fraction and the benchmark attack mass $A(\theta)$ under $B=C=1$ (so $\theta^* = 0.50$); 95\% Fisher-$z$ confidence intervals in brackets for main treatments.}
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
\parbox{\columnwidth}{\footnotesize\emph{Notes:} Data from the primary model (pure treatment; $\theta \in [0.20, 0.80]$ on a 9-point grid; $N{=}25$ agents per period). Mean join uses valid decisions; $r$ is Pearson $r(\theta,\text{join})$ across rep-level periods.}
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
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Design & No Surv. & Surv. & $\Delta$ \\
\midrule
"""
    tex += "\n".join(lines) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\par\vspace{0.25em}
\parbox{\columnwidth}{\footnotesize\emph{Notes:} ``No Surv.'' uses the communication infodesign grid. ``Surv.'' uses the same grid with surveillance active during messaging. All entries are means of \texttt{join\_fraction\_valid}.}
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

    # Compute baselines for each bandwidth
    base_005 = float(b005["baseline"])
    base_015 = main_mean("baseline")
    base_030 = float(b030["baseline"])

    rows = []
    # First row: baseline levels for reference
    rows.append(f"Baseline (level) & {_fmt_num(base_005,3)} & {_fmt_num(base_015,3)} & {_fmt_num(base_030,3)} \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{4}{l}{\textit{Treatment effect $\Delta$ (treatment $-$ baseline):}} \\")
    for design, label in [
        ("stability", "Stability"),
        ("censor_upper", "Upper cens."),
        ("censor_lower", "Lower cens."),
    ]:
        d005 = float(b005[design]) - base_005
        d015 = main_mean(design) - base_015
        d030 = float(b030[design]) - base_030
        rows.append(f"{label} & {_fmt_r(d005,3)} & {_fmt_r(d015,3)} & {_fmt_r(d030,3)} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Bandwidth robustness: treatment effects $\Delta$ (treatment $-$ baseline) within each bandwidth condition (primary model: Mistral Small Creative). Top row shows baseline join rates for reference.}
\label{tab:bandwidth}
\small
\begin{tabular}{lccc}
\toprule
 & BW=0.05 & BW=0.15 & BW=0.30 \\
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
    # Subset: only models with cross-model infodesign data
    order = [
        "Mistral Small Creative",
        "GPT-OSS 120B",
        "Llama 3.3 70B",
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

    # Subset: only models with uncalibrated robustness data
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
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Model & $N$ & Mean join & $r(\theta, J)$ & $p$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


def render_tab_surv_censor_crossmodel(stats: dict) -> str:
    sxc = stats.get("regime_control", {}).get("surveillance_x_censorship", {})
    if not sxc:
        return "% No cross-model surveillance x censorship data available.\n"

    # Subset: only models with surveillance x censorship data
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
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{Mean join (surv.)} & \multicolumn{2}{c}{$\Delta$ vs baseline} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-6}
Model & Baseline & Upper cens. & Lower cens. & $\Delta$ upper & $\Delta$ lower \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


def render_tab_logistic_params(stats: dict) -> str:
    fits = stats.get("logistic_fits", {})
    if not fits:
        return "% No logistic fit data available.\n"

    MODEL_ORDER = DISPLAY_ORDER

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
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Design & $N$ & Mean join & $r(\theta, J)$ & Cutoff $\hat{\theta}^*$ (SE) & $\Delta$ vs baseline \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
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
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Design & $N$ & Mean join & $r(\theta, J)$ & $\Delta$ vs baseline \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
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
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Temperature & $N$ & Mean join & $r(\theta, J)$ & Cutoff $\hat{\theta}^*$ & Slope $\hat{\beta}$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
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


def render_tab_ck_2x2(stats: dict) -> str:
    """CK framing x coordination intensity 2x2 table."""
    ck = stats.get("ck_interaction", {})
    if not ck or ck.get("status") == "incomplete":
        return "% No CK interaction data available.\n"

    cm = ck.get("cell_means", {})
    priv_low = cm.get("priv_low_coord")
    priv_high = cm.get("priv_high_coord")
    ck_low = cm.get("ck_low_coord")
    ck_high = cm.get("ck_high_coord")

    def pct(v):
        return f"{v*100:.1f}\\%" if v is not None else "---"

    def pp(a, b):
        if a is None or b is None:
            return "---"
        d = (a - b) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}"

    interaction = ck.get("interaction", {})
    inter_beta = interaction.get("beta")
    inter_p = interaction.get("p")

    ck_main = ck.get("ck", {})
    coord_main = ck.get("high_coord", {})

    tex = r"""\begin{table}[t]
\centering
\caption{Common knowledge $\times$ coordination intensity. Each cell reports mean join rate (270 country--periods). The CK main effect is """ + f"{pp(ck_main.get('beta'), 0) if ck_main.get('beta') is not None else '---'}" + r"""~pp ($p = """ + f"{ck_main.get('p', '---'):.4f}" + r"""$); the interaction is """ + f"{pp(inter_beta, 0) if inter_beta is not None else '---'}" + r"""~pp ($p = """ + f"{inter_p:.2f}" + r"""$).}
\label{tab:ck_2x2}
\small
\begin{tabular}{lccc}
\toprule
& Low coord & High coord & $\Delta$ (coord) \\
\midrule
Private framing & """ + pct(priv_low) + " & " + pct(priv_high) + " & " + pp(priv_high, priv_low) + r"""~pp \\
CK framing      & """ + pct(ck_low) + " & " + pct(ck_high) + " & " + pp(ck_high, ck_low) + r"""~pp \\
\midrule
$\Delta$ (CK) & """ + pp(ck_low, priv_low) + "~pp & " + pp(ck_high, priv_high) + r"""~pp & \\
\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_classifiers(stats: dict) -> str:
    """Classifier baselines table."""
    cb = stats.get("classifier_baselines", {})
    if not cb or cb.get("status") == "missing":
        return "% No classifier baseline data available.\n"

    def _acc(clf, key="cv_pure"):
        d = cb.get(clf, {}).get(key, {})
        v = d.get("accuracy_mean") if key == "cv_pure" else d.get("accuracy")
        return f"{v*100:.1f}\\%" if v is not None else "---"

    def _auc(clf, key="cv_pure"):
        d = cb.get(clf, {}).get(key, {})
        v = d.get("auc_mean") if key == "cv_pure" else d.get("auc")
        return f"{v:.3f}" if v is not None else "---"

    def _pred(clf):
        d = cb.get(clf, {}).get("cross_pure_to_surv", {})
        v = d.get("predicted_join_rate")
        return f"{v*100:.1f}\\%" if v is not None else "---"

    def _actual(clf):
        d = cb.get(clf, {}).get("cross_pure_to_surv", {})
        v = d.get("actual_join_rate")
        return f"{v*100:.1f}\\%" if v is not None else "---"

    def _gap(clf):
        d = cb.get(clf, {}).get("cross_pure_to_surv", {})
        pred = d.get("predicted_join_rate")
        actual = d.get("actual_join_rate")
        if pred is not None and actual is not None:
            g = (pred - actual) * 100
            return f"{g:.1f}"
        return "---"

    rows = []
    for clf, label in [
        ("bow_tfidf", "BoW TF-IDF"),
        ("slider_logistic", "Slider logistic"),
        ("keyphrase_sentiment", "Keyphrase"),
    ]:
        rows.append(
            f"{label} & {_acc(clf)} & {_auc(clf)} & {_pred(clf)} & {_actual(clf)} & {_gap(clf)}~pp \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Classifier baselines. Accuracy and AUC are 5-fold CV on pure-treatment data. ``Pred.\ join (surv.)'' is the classifier's predicted join rate when applied to surveillance-treatment briefings; ``Actual'' is the LLM's observed rate. The gap measures the surveillance wedge invisible to text classifiers.}
\label{tab:classifiers}
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Classifier & Acc. & AUC & Pred.\ join (surv.) & Actual (surv.) & Gap \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


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

    def _mc(name, val, nd=3):
        """Helper: emit one \\providecommand line."""
        return f"\\providecommand{{\\{name}}}{{{_fmt_num(val, nd)}}}"

    def _mc_r(name, val, nd=2):
        return f"\\providecommand{{\\{name}}}{{{_fmt_r(val, nd)}}}"

    def _mc_pp(name, val, nd=1):
        return f"\\providecommand{{\\{name}}}{{{_fmt_pp(val, nd)}}}"

    def _mc_pct(name, val, nd=1):
        return f"\\providecommand{{\\{name}}}{{{_fmt_pct(val, nd)}}}"

    def _mc_raw(name, val_str):
        return f"\\providecommand{{\\{name}}}{{{val_str}}}"

    # Slug → macro-safe short name
    _MACRO_NAMES = {
        "mistralai--mistral-small-creative": "Mistral",
        "meta-llama--llama-3.3-70b-instruct": "Llama",
        "qwen--qwen3-30b-a3b-instruct-2507": "QwenS",
        "openai--gpt-oss-120b": "GptOss",
        "qwen--qwen3-235b-a22b-2507": "QwenL",
        "arcee-ai--trinity-large-preview_free": "Trinity",
        "minimax--minimax-m2-her": "MiniMax",
    }

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

    pooled_pure = (part1.get("_pooled_pure") or {})
    pooled_comm = (part1.get("_pooled_comm") or {})
    pooled_delta_pp = ((part1.get("_pooled_comm_effect") or {}).get("unpaired") or {}).get("delta_pp")
    pooled_pval = ((part1.get("_pooled_comm_effect") or {}).get("unpaired") or {}).get("p_value")

    # Surveillance × censorship: primary model (Mistral) join levels.
    sxc = ((regime.get("surveillance_x_censorship") or {}).get("Mistral Small Creative") or {})

    lines = []
    lines.append("% Auto-generated from analysis/verified_stats.json. Do not edit by hand.")
    lines.append("% Generated by: uv run python analysis/render_paper_tables.py")
    lines.append("")

    # ── Communication summary ─────────────────────────────────────
    lines.append("% Communication summary")
    lines.append(_mc("CommPureMeanModelAvg", mean_pure_models))
    lines.append(_mc("CommCommMeanModelAvg", mean_comm_models))
    lines.append(_mc_pp("CommDeltaPPModelAvg", delta_models))
    lines.append(_mc("CommPureMeanPooled", pooled_pure.get("mean_join")))
    lines.append(_mc("CommCommMeanPooled", pooled_comm.get("mean_join")))
    if pooled_delta_pp is None:
        lines.append(_mc_raw("CommDeltaPPPooled", "---"))
    else:
        sign = "+" if pooled_delta_pp >= 0 else ""
        lines.append(_mc_raw("CommDeltaPPPooled", f"{sign}{pooled_delta_pp:.2f}"))
    lines.append(_mc("CommPValueUnpaired", pooled_pval))
    lines.append("")

    # ── Pooled pure/comm aggregate statistics ─────────────────────
    lines.append("% Pooled pure/comm aggregates")
    lines.append(_mc_r("PooledPureRTheta", (pooled_pure.get("r_vs_theta") or {}).get("r")))
    lines.append(_mc_r("PooledPureRAttack", (pooled_pure.get("r_vs_attack") or {}).get("r")))
    lines.append(_mc("PooledPureMeanJoin", pooled_pure.get("mean_join")))
    lines.append(_mc_r("PooledCommRTheta", (pooled_comm.get("r_vs_theta") or {}).get("r")))
    lines.append(_mc_r("PooledCommRAttack", (pooled_comm.get("r_vs_attack") or {}).get("r")))
    lines.append(_mc("PooledCommMeanJoin", pooled_comm.get("mean_join")))
    lines.append(_mc_r("MeanModelRPureTheta", part1.get("_mean_r_pure_vs_theta")))
    lines.append(_mc_r("MeanModelRPureAttack", part1.get("_mean_r_pure_vs_attack")))
    # Mean-of-models flip r_attack
    flip_rs = [
        (v.get("flip", {}).get("r_vs_attack") or {}).get("r")
        for k, v in part1.items()
        if isinstance(k, str) and not k.startswith("_") and isinstance(v, dict) and "flip" in v
    ]
    flip_rs = [x for x in flip_rs if x is not None]
    mean_flip_r = sum(flip_rs) / len(flip_rs) if flip_rs else None
    lines.append(_mc_r("MeanModelRFlipAttack", mean_flip_r))
    # Pooled flip
    pooled_flip = part1.get("_pooled_flip", {})
    lines.append(_mc_r("PooledFlipRAttack", (pooled_flip.get("r_vs_attack") or {}).get("r")))
    lines.append("")

    # ── Pooled OLS ────────────────────────────────────────────────
    ols = stats.get("pooled_ols", {})
    lines.append("% Pooled OLS (join_fraction ~ theta)")
    lines.append(_mc("PooledOLSIntercept", ols.get("intercept")))
    lines.append(_mc("PooledOLSSlope", ols.get("slope"), 4))
    lines.append(_mc("PooledOLSRSq", ols.get("r_squared"), 4))
    lines.append(_mc_raw("PooledOLSNObs", str(ols.get("n_obs", "---"))))
    # Display-format OLS (2dp for inline equation)
    lines.append(_mc("PooledOLSInterceptDisp", ols.get("intercept"), 2))
    lines.append(_mc("PooledOLSSlopeDisp", ols.get("slope"), 2))
    lines.append(_mc("PooledOLSRSqDisp", ols.get("r_squared"), 2))
    lines.append("")

    # ── Clustered standard errors ─────────────────────────────────
    clust = stats.get("clustered_ses", {})
    lines.append("% Clustered standard errors on slope")
    for cluster_type, macro_prefix in [
        ("homoskedastic", "ClusteredSEHomo"),
        ("hc1", "ClusteredSEHCOne"),
        ("clustered_country", "ClusteredSECountry"),
        ("clustered_model", "ClusteredSEModel"),
    ]:
        se = (clust.get(cluster_type) or {}).get("se_slope")
        lines.append(_mc(f"{macro_prefix}Slope", se, 4))
    lines.append("")

    # ── Per-model pure r-values and mean join ─────────────────────
    lines.append("% Per-model pure statistics")
    for slug in PART1_SLUGS:
        display = DISPLAY_NAMES.get(slug, slug)
        mname = _MACRO_NAMES.get(slug, slug.split("--")[-1].title())
        entry = part1.get(display, {})
        pure = entry.get("pure", {})
        r_theta = (pure.get("r_vs_theta") or {}).get("r")
        r_attack = (pure.get("r_vs_attack") or {}).get("r")
        mean_join = pure.get("mean_join")
        lines.append(_mc_r(f"{mname}PureRTheta", r_theta))
        lines.append(_mc_r(f"{mname}PureRAttack", r_attack))
        lines.append(_mc(f"{mname}PureMeanJoin", mean_join))
    lines.append("")

    # ── Fisher z-tests (pure vs scramble/flip) ────────────────────
    fisher_scr = part1.get("_fisher_pure_vs_scramble_attack", {})
    fisher_flip = part1.get("_fisher_pure_vs_flip_attack", {})
    lines.append("% Fisher z-tests")
    lines.append(_mc("FisherPureVsScrambleZ", fisher_scr.get("z"), 2))
    lines.append(_mc("FisherPureVsScrambleP", fisher_scr.get("p"), 4))
    lines.append(_mc("FisherPureVsFlipZ", fisher_flip.get("z"), 2))
    lines.append(_mc("FisherPureVsFlipP", fisher_flip.get("p"), 4))
    lines.append("")

    # ── Infodesign summary (primary model) ────────────────────────
    lines.append("% Infodesign summary (primary model)")
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

    # ── Surveillance × censorship levels (primary model) ──────────
    lines.append("% Surveillance x censorship levels (primary model)")
    for dname, macro in [("baseline", "SXCBase"), ("censor_upper", "SXCUpper"), ("censor_lower", "SXCLower")]:
        surv_mean = sxc.get(dname)
        lines.append(f"\\providecommand{{\\{macro}SurvMean}}{{{_fmt_num(surv_mean, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}SurvMeanPct}}{{{_fmt_pct(surv_mean, 1)}}}")
    lines.append("")

    # ── Surveillance per-model ────────────────────────────────────
    surv_data = regime.get("surveillance", {})
    lines.append("% Surveillance per-model delta vs baseline")
    for slug in PART1_SLUGS:
        display = DISPLAY_NAMES.get(slug, slug)
        mname = _MACRO_NAMES.get(slug, slug.split("--")[-1].title())
        entry = surv_data.get(display, {})
        lines.append(_mc_pp(f"Surv{mname}DeltaPP", entry.get("delta_vs_baseline_pp")))
        lines.append(_mc(f"Surv{mname}MeanJoin", entry.get("mean_join")))
    lines.append("")

    # ── Surveillance variants (placebo, anonymous) ────────────────
    sv = stats.get("surveillance_variants", {})
    lines.append("% Surveillance variants")
    for variant, macro in [("placebo", "PlaceboSurv"), ("anonymous", "AnonSurv")]:
        vd = sv.get(variant, {})
        lines.append(_mc_pp(f"{macro}DeltaPP", vd.get("delta_vs_comm_pp")))
        tt = vd.get("t_test_vs_comm", {})
        lines.append(_mc(f"{macro}PValue", tt.get("p_value") if isinstance(tt, dict) else None, 4))
        lines.append(_mc(f"{macro}MeanJoin", vd.get("mean_join")))
    lines.append("")

    # ── Fixed messages test ───────────────────────────────────────
    fm = stats.get("fixed_messages_test", {})
    lines.append("% Fixed messages surveillance test")
    lines.append(_mc_pp("FixedMsgDeltaPP", fm.get("delta_pp")))
    lines.append(_mc("FixedMsgTStat", fm.get("ttest_t"), 2))
    lines.append(_mc("FixedMsgPValue", fm.get("ttest_p"), 4))
    lines.append(_mc("FixedMsgBaselineMean", fm.get("baseline_mean_join")))
    lines.append(_mc("FixedMsgSurvMean", fm.get("surv_mean_join")))
    lines.append("")

    # ── Beliefs v2 ────────────────────────────────────────────────
    beliefs = stats.get("beliefs_v2", {})
    lines.append("% Beliefs v2")
    for treatment, macro in [("pure", "BeliefPure"), ("comm", "BeliefComm"), ("surveillance", "BeliefSurv")]:
        bd = beliefs.get(treatment, {})
        r_post = bd.get("r_posterior_belief")
        r_dec = bd.get("r_belief_decision")
        lines.append(_mc_r(f"{macro}RPosterior", r_post.get("r") if isinstance(r_post, dict) else r_post))
        lines.append(_mc_r(f"{macro}RDecision", r_dec.get("r") if isinstance(r_dec, dict) else r_dec))
        lines.append(_mc(f"{macro}MeanBelief", bd.get("mean_belief")))
        lines.append(_mc(f"{macro}MeanJoin", bd.get("mean_join")))
    # Second-order beliefs
    sob = beliefs.get("_surv_vs_comm_sob", {})
    lines.append("% Second-order beliefs (surveillance vs comm)")
    lines.append(_mc("SOBCommMean", sob.get("comm_mean")))
    lines.append(_mc("SOBSurvMean", sob.get("surv_mean")))
    lines.append(_mc_pp("SOBDeltaPP", sob.get("delta_pp")))
    lines.append(_mc("SOBPValue", sob.get("p_value"), 4))
    lines.append(_mc("SOBTStat", sob.get("t_stat"), 2))
    # Preference falsification
    pf = beliefs.get("_pref_falsification", {})
    lines.append("% Preference falsification")
    lines.append(_mc("PrefFalsPureMeanBelief", pf.get("pure_mean_belief")))
    lines.append(_mc("PrefFalsSurvMeanBelief", pf.get("surv_mean_belief")))
    lines.append(_mc_pp("PrefFalsBeliefDeltaPP", pf.get("belief_delta_pp")))
    lines.append(_mc_pp("PrefFalsActionDeltaPP", pf.get("action_delta_pp")))
    lines.append("")

    # ── Classifier baselines ──────────────────────────────────────
    cb = stats.get("classifier_baselines", {})
    lines.append("% Classifier baselines")
    bow = cb.get("bow_tfidf", {})
    bow_cv = bow.get("cv_pure", {})
    bow_surv = bow.get("cross_pure_to_surv", {})
    lines.append(_mc("ClassBowAccuracy", bow_cv.get("accuracy_mean")))
    lines.append(_mc("ClassBowAUC", bow_cv.get("auc_mean")))
    lines.append(_mc("ClassBowSurvPredicted", bow_surv.get("predicted_join_rate")))
    lines.append(_mc("ClassBowSurvActual", bow_surv.get("actual_join_rate")))
    gap = None
    if bow_surv.get("predicted_join_rate") is not None and bow_surv.get("actual_join_rate") is not None:
        gap = bow_surv["predicted_join_rate"] - bow_surv["actual_join_rate"]
    lines.append(_mc_pp("ClassBowSurvGapPP", gap))
    # Slider classifier
    slider = cb.get("slider_logistic", {})
    slider_cv = slider.get("cv_pure", {})
    slider_surv = slider.get("cross_pure_to_surv", {})
    lines.append(_mc("ClassSliderAccuracy", slider_cv.get("accuracy_mean")))
    lines.append(_mc("ClassSliderAUC", slider_cv.get("auc_mean")))
    lines.append(_mc("ClassSliderSurvPredicted", slider_surv.get("predicted_join_rate")))
    lines.append(_mc("ClassSliderSurvActual", slider_surv.get("actual_join_rate")))
    # BC comparative statics
    bc_cs = cb.get("bc_comparative_statics", {})
    for cond, macro in [("baseline", "ClassBCBaseline"), ("bc_high_cost", "ClassBCHighCost"), ("bc_low_cost", "ClassBCLowCost")]:
        cd = bc_cs.get(cond, {})
        lines.append(_mc(f"{macro}Predicted", cd.get("classifier_predicted_join")))
        lines.append(_mc(f"{macro}Actual", cd.get("actual_join")))
        lines.append(_mc_pp(f"{macro}GapPP", cd.get("gap_pp")))
    lines.append("")

    # ── B/C sweep (infodesign) ────────────────────────────────────
    lines.append("% B/C sweep")
    baseline_info = info.get("baseline", {})
    bc_high = info.get("bc_high_cost", {})
    bc_low = info.get("bc_low_cost", {})
    lines.append(_mc("BCSweepBaselineMeanJoin", baseline_info.get("mean_join")))
    lines.append(_mc("BCSweepHighCostMeanJoin", bc_high.get("mean_join")))
    lines.append(_mc("BCSweepLowCostMeanJoin", bc_low.get("mean_join")))
    # Logistic cutoffs
    bl_fit = baseline_info.get("logistic_fit", {})
    hc_fit = bc_high.get("logistic_fit", {})
    lc_fit = bc_low.get("logistic_fit", {})
    lines.append(_mc("BCSweepBaselineCutoff", bl_fit.get("cutoff"), 3))
    lines.append(_mc("BCSweepHighCostCutoff", hc_fit.get("cutoff"), 3))
    lines.append(_mc("BCSweepLowCostCutoff", lc_fit.get("cutoff"), 3))
    lines.append("")

    # ── Coordination cues ─────────────────────────────────────────
    lines.append("% Coordination cues")
    for key, macro in [("coord_amplified", "CoordAmplified"), ("coord_suppressed", "CoordSuppressed")]:
        d = info.get(key, {})
        lines.append(_mc(f"{macro}MeanJoin", d.get("mean_join")))
        lines.append(_mc_pp(f"{macro}DeltaPP", d.get("delta_vs_baseline")))
        # Slope from logistic fit
        fit = d.get("logistic_fit", {})
        lines.append(_mc(f"{macro}Slope", fit.get("b1"), 2))
    lines.append("")

    # ── CK framing ────────────────────────────────────────────────
    lines.append("% CK framing designs")
    for key, macro in [
        ("ck_high_coord", "CKHighCoord"), ("ck_low_coord", "CKLowCoord"),
        ("priv_high_coord", "PrivHighCoord"), ("priv_low_coord", "PrivLowCoord"),
    ]:
        d = info.get(key, {})
        lines.append(_mc(f"{macro}MeanJoin", d.get("mean_join")))
        lines.append(_mc_pp(f"{macro}DeltaPP", d.get("delta_vs_baseline")))
    lines.append("")

    # CK interaction macros
    ck = stats.get("ck_interaction", {})
    ck_main = ck.get("ck", {})
    ck_inter = ck.get("interaction", {})
    lines.append("% CK interaction test")
    ck_main_beta = ck_main.get("beta")
    ck_main_p = ck_main.get("p")
    ck_inter_beta = ck_inter.get("beta")
    ck_inter_p = ck_inter.get("p")
    lines.append(r"\providecommand{\CKMainEffectBeta}{" + (_fmt_pp(ck_main_beta, 1) if ck_main_beta is not None else "---") + "}")
    lines.append(r"\providecommand{\CKMainEffectPValue}{" + (_fmt_num(ck_main_p, 4) if ck_main_p is not None else "---") + "}")
    lines.append(r"\providecommand{\CKInteractionBeta}{" + (_fmt_pp(ck_inter_beta, 1) if ck_inter_beta is not None else "---") + "}")
    lines.append(r"\providecommand{\CKInteractionPValue}{" + (_fmt_num(ck_inter_p, 2) if ck_inter_p is not None else "---") + "}")
    lines.append("")

    # ── Temperature robustness ────────────────────────────────────
    temp = stats.get("temperature_robustness", {})
    lines.append("% Temperature robustness")
    for t, macro in [("T=0.3", "TempThree"), ("T=0.7", "TempSeven"), ("T=1.0", "TempOne")]:
        td = temp.get(t, {})
        lines.append(_mc_r(f"{macro}RTheta", (td.get("r_vs_theta") or {}).get("r") if isinstance(td.get("r_vs_theta"), dict) else td.get("r_vs_theta")))
        lines.append(_mc(f"{macro}MeanJoin", td.get("mean_join")))
    lines.append("")

    # ── Uncalibrated robustness ───────────────────────────────────
    uncal = stats.get("uncalibrated", {})
    lines.append("% Uncalibrated robustness")
    for slug in PART1_SLUGS:
        display = DISPLAY_NAMES.get(slug, slug)
        mname = _MACRO_NAMES.get(slug, slug.split("--")[-1].title())
        ud = uncal.get(display, {})
        if not ud:
            continue
        r_val = (ud.get("r_vs_theta") or {}).get("r") if isinstance(ud.get("r_vs_theta"), dict) else ud.get("r_vs_theta")
        lines.append(_mc_r(f"Uncal{mname}R", r_val))
        lines.append(_mc(f"Uncal{mname}MeanJoin", ud.get("mean_join")))
    lines.append("")

    # ── Cross-generator robustness ────────────────────────────────
    cg = stats.get("cross_generator", {})
    lines.append("% Cross-generator robustness")
    for display, entry in cg.items():
        if not isinstance(entry, dict):
            continue
        # Find macro-safe name (LaTeX commands can only contain letters)
        mname = "".join(c for c in display if c.isalpha())[:12]
        for variant in ["baseline", "cable", "journalistic"]:
            vd = entry.get(variant, {})
            if not vd:
                continue
            r_val = (vd.get("r_vs_theta") or {}).get("r") if isinstance(vd.get("r_vs_theta"), dict) else vd.get("r_vs_theta")
            lines.append(_mc_r(f"CrossGen{mname}{variant.title()}R", r_val))
    lines.append("")

    # ── Propaganda saturation ─────────────────────────────────────
    prop_sat = regime.get("_propaganda_saturation_k5_k10", {})
    lines.append("% Propaganda saturation test")
    lines.append(_mc("PropSatKFiveMeanJoin", prop_sat.get("k5_mean_join_real")))
    lines.append(_mc("PropSatKTenMeanJoin", prop_sat.get("k10_mean_join_real")))
    lines.append(_mc_pp("PropSatDeltaPP", prop_sat.get("delta_pp")))
    lines.append(_mc("PropSatPValue", prop_sat.get("p_value"), 4))
    lines.append("")

    # ── Llama propaganda replication ──────────────────────────────
    prop_data = regime.get("propaganda", {})
    llama_k5 = (prop_data.get("k=5") or {}).get("Llama 3.3 70B", {})
    lines.append("% Llama propaganda k=5 replication")
    # delta_real_vs_baseline_pp is already in pp, don't multiply by 100
    llama_delta = llama_k5.get("delta_real_vs_baseline_pp")
    if llama_delta is not None:
        sign = "+" if llama_delta >= 0 else ""
        lines.append(_mc_raw("LlamaPropKFiveDeltaRealPP", f"{sign}{llama_delta:.1f}"))
    lines.append("")

    # ── Mixed-model robustness ─────────────────────────────────
    rob = stats.get("robustness", {})
    lines.append("% Mixed-model robustness")
    mixed_pure = rob.get("mixed-5model-pure", {})
    mixed_comm = rob.get("mixed-5model-comm", {})
    # These experiments report a single entry (Mistral) representing the pooled mixed-model run
    mp_entry = list(mixed_pure.values())[0] if mixed_pure else {}
    mc_entry = list(mixed_comm.values())[0] if mixed_comm else {}
    lines.append(_mc_r("MixedPureRAttack", (mp_entry.get("r_vs_attack") or {}).get("r")))
    lines.append(_mc_r("MixedCommRAttack", (mc_entry.get("r_vs_attack") or {}).get("r")))
    lines.append("")

    # ── Parse errors ──────────────────────────────────────────────
    pe = stats.get("parse_errors", {})
    lines.append("% Parse error rates")
    for slug in PART1_SLUGS:
        display = DISPLAY_NAMES.get(slug, slug)
        mname = _MACRO_NAMES.get(slug, slug.split("--")[-1].title())
        pd_entry = pe.get(display, {})
        if not pd_entry:
            continue
        rate = pd_entry.get("unparseable_rate")
        lines.append(_mc(f"ParseErr{mname}", rate, 3))
    lines.append("")

    return "\n".join(lines) + "\n"


def render_tab_hypotheses(stats: dict) -> str:
    """Render hypothesis summary table (H1-H8) from verified stats."""
    hyp = stats.get("hypothesis_table")
    if not hyp:
        return "% No hypothesis table data available.\n"

    def _fmt_p(p) -> str:
        if p is None:
            return "---"
        try:
            if p != p:  # nan
                return "---"
        except Exception:
            return "---"
        if p < 0.001:
            return "$<$0.001"
        return f"{p:.3f}"

    def _fmt_stat(s) -> str:
        if s is None:
            return "---"
        try:
            if s != s:
                return "---"
        except Exception:
            return "---"
        return f"{s:.3f}"

    rows = []
    for h in hyp:
        hid = h["id"]
        label = h["hypothesis"]
        estimand = h.get("estimand", "---")
        null = h.get("null", "---")
        test = h.get("test", "---")
        stat = _fmt_stat(h.get("stat"))
        p = _fmt_p(h.get("p"))
        supported = h.get("supported", "---")
        rows.append(
            f"{hid} & {label} & {estimand} & {null} & {test} & {stat} & {p} & {supported} \\\\"
        )

    tex = r"""\begin{table*}[t]
\centering
\caption{Pre-specified hypotheses and test results. H1--H4 use pooled Part~I data across all seven models; H5--H8 use the primary model (Mistral Small Creative). ``Supported'' indicates whether the data pattern matches the hypothesis at $\alpha = 0.05$.}
\label{tab:hypotheses}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llllllcl}
\toprule
H & Hypothesis & Estimand & Null & Test & Stat & $p$ & Supported? \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    return tex


def render_tab_cross_generator(stats: dict) -> str:
    """Cross-generator language variant robustness table."""
    cg = stats.get("cross_generator", {})
    if not cg:
        return "% No cross-generator data available.\n"

    models = ["Mistral Small Creative", "Llama 3.3 70B"]
    variants = ["baseline", "cable", "journalistic"]

    rows = []
    for m in models:
        m_data = cg.get(m, {})
        for v in variants:
            d = m_data.get(v, {})
            if not d:
                rows.append(f"{m} & {v.capitalize()} & --- & --- & --- \\\\")
                continue
            n = d.get("n_obs", "---")
            mean_j = _fmt_num(d.get("mean_join"), 3)
            r = (d.get("r_vs_theta") or {}).get("r")
            r_cell = f"${_fmt_r(r, 3)}$" if r is not None else "---"
            fit = d.get("logistic_fit", {})
            cutoff = _fmt_num(fit.get("cutoff"), 3) if fit else "---"
            rows.append(f"{m} & {v.capitalize()} & {n} & {mean_j} & {r_cell} & {cutoff} \\\\")
        rows.append(r"\midrule")
    # Remove trailing midrule
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    tex = r"""\begin{table}[t]
\centering
\caption{Cross-generator robustness. Three text rendering styles (baseline, diplomatic cable, journalistic wire) use identical slider functions and evidence items; only prose formatting differs. The Pearson $r(\theta, J)$ and logistic cutoff are virtually identical across generators.}
\label{tab:cross_generator}
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
Model & Generator & $N$ & Mean join & $r(\theta, J)$ & Cutoff $\hat{\theta}^*$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


def render_tab_placebo_calibration(stats: dict) -> str:
    """Placebo calibration table."""
    pc = stats.get("placebo_calibration", {})
    if not pc:
        return "% No placebo calibration data available.\n"

    models = ["Mistral Small Creative", "Llama 3.3 70B"]

    rows = []
    for m in models:
        m_data = pc.get(m, {})
        # Also get the calibrated baseline r from part1
        part1 = stats.get("part1", {})
        baseline_r = (part1.get(m, {}).get("pure", {}).get("r_vs_theta") or {}).get("r")
        baseline_mean = part1.get(m, {}).get("pure", {}).get("mean_join")

        rows.append(f"{m} & Calibrated & --- & {_fmt_num(baseline_mean, 3)} & ${_fmt_r(baseline_r, 3)}$ \\\\")

        for shift in ["+0.3", "-0.3"]:
            d = m_data.get(shift, {})
            if not d:
                rows.append(f" & $\\Delta c = {shift}$ & --- & --- & --- \\\\")
                continue
            n = d.get("n_obs", "---")
            mean_j = _fmt_num(d.get("mean_join"), 3)
            r = (d.get("r_vs_theta") or {}).get("r")
            r_cell = f"${_fmt_r(r, 3)}$" if r is not None else "---"
            rows.append(f" & $\\Delta c = {shift}$ & {n} & {mean_j} & {r_cell} \\\\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    tex = r"""\begin{table}[t]
\centering
\caption{Placebo calibration. The cutoff center is deliberately shifted by $\pm 0.3$ from its calibrated value. The correlation $r(\theta, J)$ is unchanged; only the mean join rate shifts, confirming that calibration does not create the sigmoid.}
\label{tab:placebo_calibration}
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccc}
\toprule
Model & Condition & $N$ & Mean join & $r(\theta, J)$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


def render_tab_temperature_expanded(stats: dict) -> str:
    """Expanded temperature robustness table (3 models)."""
    # Combine old Mistral data with new Llama + Qwen data
    temp_old = stats.get("temperature_robustness", {})
    temp_new = stats.get("temperature_expanded", {})
    if not temp_old and not temp_new:
        return "% No temperature data available.\n"

    rows = []

    # Mistral (from old)
    if temp_old:
        for key in ["T=0.3", "T=0.7", "T=1.0"]:
            d = temp_old.get(key, {})
            if not d:
                continue
            mean_j = _fmt_num(d["mean_join"], 3)
            r_val = _fmt_r((d.get("r_vs_theta") or {}).get("r"), 3)
            fit = d.get("logistic_fit", {})
            cutoff = _fmt_num(fit.get("cutoff"), 3) if fit else "---"
            rows.append(f"Mistral Small & {key} & {d.get('n_obs','---')} & {mean_j} & ${r_val}$ & {cutoff} \\\\")
        rows.append(r"\midrule")

    # Llama and Qwen (from new)
    for model in ["Llama 3.3 70B", "Qwen3 235B"]:
        m_data = temp_new.get(model, {})
        for temp in ["T=0.3", "T=0.5", "T=0.7", "T=1.0", "T=1.2"]:
            d = m_data.get(temp, {})
            if not d:
                continue
            mean_j = _fmt_num(d["mean_join"], 3)
            r_val = _fmt_r((d.get("r_vs_theta") or {}).get("r"), 3)
            fit = d.get("logistic_fit", {})
            cutoff = _fmt_num(fit.get("cutoff"), 3) if fit else "---"
            short_name = "Llama 70B" if model == "Llama 3.3 70B" else "Qwen 235B"
            rows.append(f"{short_name} & {temp} & {d.get('n_obs','---')} & {mean_j} & ${r_val}$ & {cutoff} \\\\")
        rows.append(r"\midrule")

    if rows and rows[-1] == r"\midrule":
        rows.pop()

    tex = r"""\begin{table}[t]
\centering
\caption{Temperature robustness across three models. The pure global game is run at varying LLM decoding temperatures. The correlation $r(\theta, J)$ is stable across all temperatures and models.}
\label{tab:temperature_expanded}
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
Model & $T$ & $N$ & Mean join & $r(\theta, J)$ & Cutoff $\hat{\theta}^*$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


def render_tab_uncalibrated_expanded(stats: dict) -> str:
    """Expanded uncalibrated table with all available models."""
    uncal = stats.get("uncalibrated_expanded", {})
    if not uncal:
        return "% No expanded uncalibrated data available.\n"

    # Subset: only models with expanded uncalibrated data (excludes Trinity)
    model_order = [
        "Mistral Small Creative", "Llama 3.3 70B", "Qwen3 30B",
        "GPT-OSS 120B", "Qwen3 235B", "MiniMax M2-Her",
    ]

    rows = []
    for m in model_order:
        d = uncal.get(m, {})
        if not d:
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
\caption{Uncalibrated robustness: models run without any calibration adjustment. Even without calibration, six of seven models show strong $r(\theta, J)$, confirming that the sigmoid is not an artifact of the calibration procedure.}
\label{tab:uncalibrated_expanded}
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Model & $N$ & Mean join & $r(\theta, J)$ & $p$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


def render_tab_punishment_risk(stats: dict) -> str:
    """Punishment risk elicitation table."""
    pr = stats.get("punishment_risk", {})
    if not pr:
        return "% No punishment risk data available.\n"

    models = ["Mistral Small Creative", "Llama 3.3 70B"]
    conditions = ["pure", "comm", "surveillance"]

    rows = []
    for m in models:
        m_data = pr.get(m, {})
        for cond in conditions:
            d = m_data.get(cond, {})
            agent = d.get("agent_level", {}) if isinstance(d, dict) else {}
            if not agent:
                continue
            n = agent.get("n_agents", "---")
            mean_pr = _fmt_num(agent.get("mean_pr"), 1)
            pr_join = _fmt_num(agent.get("mean_pr_join"), 1)
            pr_stay = _fmt_num(agent.get("mean_pr_stay"), 1)
            short_m = "Mistral" if "Mistral" in m else "Llama 70B"
            cond_label = cond.capitalize()
            rows.append(f"{short_m} & {cond_label} & {n} & {mean_pr} & {pr_join} & {pr_stay} \\\\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    tex = r"""\begin{table}[t]
\centering
\caption{Elicited punishment risk (0--10 scale). Agents rate expected regime punishment after their JOIN/STAY decision. ``JOIN'' and ``STAY'' columns show the mean rating conditional on the agent's own decision.}
\label{tab:punishment_risk}
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
Model & Condition & $N$ & Mean risk & Risk $|$ JOIN & Risk $|$ STAY \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


def render_tab_bc_classifier(stats: dict) -> str:
    """Render B/C classifier comparative statics table."""
    cb = stats.get("classifier_baselines", {})
    bc = cb.get("bc_comparative_statics", {})
    if not bc:
        return "% No B/C classifier data available.\n"

    cond_labels = {
        "baseline": "Baseline ($\\theta^* = 0.50$)",
        "bc_high_cost": "High cost ($\\theta^* = 0.25$)",
        "bc_low_cost": "Low cost ($\\theta^* = 0.75$)",
    }

    rows = []
    for cond in ["baseline", "bc_high_cost", "bc_low_cost"]:
        d = bc.get(cond)
        if d is None:
            continue
        label = cond_labels.get(cond, cond)
        pred = d["classifier_predicted_join"]
        actual = d["actual_join"]
        gap = d["gap_pp"]
        n = d["n_obs"]
        rows.append(
            f"{label} & {n} & {pred*100:.1f}\\% & {actual*100:.1f}\\% & {gap:+.1f} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{B/C comparative statics: classifier vs.\ actual LLM behavior. A logistic trained on baseline join rates (which captures the same information as slider features) predicts similar join rates across all payoff conditions. Actual LLM behavior shifts by ${\approx}\,50$~pp, demonstrating that agents respond to payoff information not captured by text features.}
\label{tab:bc_classifier}
\small
\begin{tabular}{lcccc}
\toprule
Condition & $N$ & Classifier pred. & Actual & Gap (pp) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_parse_errors(stats: dict) -> str:
    pe = stats.get("parse_errors", {})
    if not pe:
        return "% No parse error data available.\n"

    models = DISPLAY_ORDER
    treatments = ["pure", "comm", "scramble", "flip"]
    treat_labels = {"pure": "Pure", "comm": "Comm", "scramble": "Scramble", "flip": "Flip"}

    rows = []
    for model in models:
        m_data = pe.get(model, {})
        if not m_data:
            continue
        first = True
        for t in treatments:
            t_data = m_data.get(t)
            if t_data is None:
                continue
            api_err = t_data.get("mean_api_error_rate", 0.0)
            unparse = t_data.get("mean_unparseable_rate", 0.0)
            combined = api_err + unparse
            n = t_data.get("n_periods", "---")
            model_col = model if first else ""
            first = False
            rows.append(
                f"{model_col} & {treat_labels[t]} & {n} & "
                f"{api_err*100:.1f}\\% & {unparse*100:.1f}\\% & {combined*100:.1f}\\% \\\\"
            )
        rows.append(r"\addlinespace")

    # Remove trailing \addlinespace
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()

    tex = r"""\begin{table}[t]
\centering
\caption{Parse error and API failure rates by model and treatment. API error = provider-side failure; unparseable = valid response that could not be classified as JOIN or STAY. Combined rates are below 2\% for five of seven models; Trinity Large has elevated API errors (${\approx}\,9$\%) due to provider-side content filtering.}
\label{tab:parse_errors}
\small
\begin{tabular}{llcccc}
\toprule
Model & Treatment & $N$ & API err & Unparseable & Combined \\
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
        "tab_uncalibrated.tex": render_tab_uncalibrated(stats),
        "tab_surv_censor_crossmodel.tex": render_tab_surv_censor_crossmodel(stats),
        "tab_logistic_params.tex": render_tab_logistic_params(stats),
        "tab_surveillance_variants.tex": render_tab_surveillance_variants(stats),
        "tab_bc_statics.tex": render_tab_bc_statics(stats),
        "tab_censor_ck.tex": render_tab_censor_ck(stats),
        "tab_temperature.tex": render_tab_temperature(stats),
        "tab_hypotheses.tex": render_tab_hypotheses(stats),
        "tab_ck_2x2.tex": render_tab_ck_2x2(stats),
        "tab_classifiers.tex": render_tab_classifiers(stats),
        "tab_cross_generator.tex": render_tab_cross_generator(stats),
        "tab_placebo_calibration.tex": render_tab_placebo_calibration(stats),
        "tab_temperature_expanded.tex": render_tab_temperature_expanded(stats),
        "tab_uncalibrated_expanded.tex": render_tab_uncalibrated_expanded(stats),
        "tab_punishment_risk.tex": render_tab_punishment_risk(stats),
        "tab_parse_errors.tex": render_tab_parse_errors(stats),
        "tab_bc_classifier.tex": render_tab_bc_classifier(stats),
        "stats_macros.tex": render_stats_macros(stats),
    }

    for name, content in tables.items():
        _write(OUT_DIR / name, content)

    print(f"Wrote {len(tables)} table(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
