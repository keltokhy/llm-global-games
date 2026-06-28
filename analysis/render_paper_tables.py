"""
Render LaTeX tables for the paper from verified_stats.json.

This avoids manual copy/paste errors: the paper should `\\input{}` these files.

Usage:
    uv run python agent_based_simulation/render_paper_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

from models import DISPLAY_ORDER, DISPLAY_NAMES, PART1_SLUGS, PRIMARY_SLUG


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


def _fmt_p(x: float, nd: int = 3) -> str:
    """Format a displayed p-value; never shows a bare 0.000."""
    if x is None:
        return "---"
    try:
        if x != x:  # nan
            return "---"
    except Exception:
        return "---"
    threshold = 10 ** (-nd)
    if x < threshold:
        return f"$<${threshold:.{nd}f}"
    return f"{x:.{nd}f}"


def _fmt_p_text_plain(val: float | None, nd: int = 3) -> str:
    """Inline p text for math mode: '= 0.123' or '<0.001'."""
    if val is None:
        return "= \\text{---}"
    try:
        if val != val:  # nan
            return "= \\text{---}"
    except Exception:
        return "= \\text{---}"
    threshold = 10 ** (-nd)
    if val < threshold:
        return f"<{threshold:.{nd}f}"
    return f"= {val:.{nd}f}"


def _fmt_int_sep(x) -> str:
    """Integer with LaTeX-safe thousands separators (15000 -> 15{,}000)."""
    if x is None:
        return "---"
    try:
        return f"{int(x):,}".replace(",", "{,}")
    except Exception:
        return "---"


def _round_disp(x: float | None, nd: int = 1) -> float | None:
    """Round to the displayed precision (so deltas can match displayed endpoints)."""
    if x is None:
        return None
    try:
        if x != x:  # nan
            return None
    except Exception:
        return None
    return float(f"{float(x):.{nd}f}")


def _delta_of_displayed(a: float | None, b: float | None, nd: int = 1) -> float | None:
    """b - a computed from the displayed (rounded) endpoints, for internal consistency."""
    ra, rb = _round_disp(a, nd), _round_disp(b, nd)
    if ra is None or rb is None:
        return None
    return round(rb - ra, nd)


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
    slug_by_display = {
        DISPLAY_NAMES[s]: s.replace("--", "/").replace("_", r"\_")
        for s in PART1_SLUGS
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
        slug_cell = f"{{\\scriptsize\\texttt{{{slug_by_display.get(m, '')}}}}}"
        lines.append(
            f"{m} & {slug_cell} & {arch.get(m,'')} & {pure_n} & {comm_n} & {falsif_cell} \\\\"
        )

    # Reconcile collected vs valid communication rows (Trinity API failure)
    comm_totals = (
        ((stats["part1"].get("_pooled_comm_effect") or {}).get("decomposition") or {})
        .get("totals") or {}
    )
    comm_valid = comm_totals.get("comm_rows")
    trinity_note = ""
    if comm_valid is not None and total_comm and comm_valid != total_comm:
        lost = total_comm - int(comm_valid)
        trinity_note = (
            f" The communication total of {total_comm:,} counts rows as collected; "
            f"{lost} Trinity Large row{'s' if lost != 1 else ''} (all 25 agent calls failing at the provider) "
            f"yield{'s' if lost == 1 else ''} no valid join fraction, so "
            f"Table~\\ref{{tab:comm_estimators}} counts {int(comm_valid):,} valid communication rows."
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Model summary. Columns report period-level task rows in the pure, communication, and falsification (scramble+flip) benchmark suites. These core suites use $n=25$ agents per row and $\sigma=0.3$; Appendix~\ref{sec:robustness} varies agent count.}
\label{tab:models}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllccc}
\toprule
Model & Identifier (OpenRouter) & Arch. & Pure & Comm & Falsif. \\
\midrule
"""
    tex += "\n".join(lines) + "\n"
    tex += r"""\midrule
\textbf{Total} & & & \textbf{""" + f"{total_pure}" + r"""} & \textbf{""" + f"{total_comm}" + r"""} & \textbf{""" + f"{total_falsif}" + r"""} \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Seven models spanning six architecture families; the identifier column gives each model's OpenRouter slug. Subsequent tables abbreviate Mistral Small Creative as Mistral, Trinity Large as Trinity, and MiniMax M2-Her as MiniMax M2. Counts are period-level task rows with 25 agents each in the core benchmark suites. Mistral includes appended batches; duplicate-cell accounting is reported in the main text and Table~\ref{tab:comm_estimators}.""" + trinity_note + r""" Models were accessed as hosted snapshots via the OpenRouter API; archived run manifests record a February 2026 collection window, and raw request--response pairs are cached with the run outputs. Core runs use $\sigma = 0.3$ and temperature $= 0.7$ unless otherwise stated.
\end{tablenotes}
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
        return _fmt_pct(d.get("mean_join"), nd=1)

    def n_by_treatment(m: str) -> str:
        parts = []
        for t in ["pure", "comm", "scramble", "flip"]:
            d = part1.get(m, {}).get(t, {})
            n = d.get("n_obs") if isinstance(d, dict) else None
            parts.append(str(n) if n is not None else "---")
        return "/".join(parts)

    rows = []
    for m in models:
        rows.append(
            f"{m} & {r_attack_with_ci(m,'pure')} & {r_attack_with_ci(m,'comm')} & {r_attack_with_ci(m,'scramble')} & {r_attack_with_ci(m,'flip')} & {n_by_treatment(m)} & {mean_join(m)} \\\\"
        )

    pooled = part1.get("_pooled_pure", {}).get("r_vs_attack", {}).get("r")
    pooled_comm = part1.get("_pooled_comm", {}).get("r_vs_attack", {}).get("r")
    pooled_scr = part1.get("_pooled_scramble", {}).get("r_vs_attack", {}).get("r")
    pooled_flip = part1.get("_pooled_flip", {}).get("r_vs_attack", {}).get("r")
    pooled_ns = "/".join(
        str(part1.get(f"_pooled_{t}", {}).get("n_obs") or "---")
        for t in ["pure", "comm", "scramble", "flip"]
    )
    pooled_mean = part1.get("_pooled_pure", {}).get("mean_join")

    pooled_pure_ci = part1.get("_pooled_pure", {}).get("r_vs_attack", {})
    pooled_comm_ci = part1.get("_pooled_comm", {}).get("r_vs_attack", {})
    pooled_scr_ci = part1.get("_pooled_scramble", {}).get("r_vs_attack", {})
    pooled_flip_ci = part1.get("_pooled_flip", {}).get("r_vs_attack", {})

    mean_pure = part1.get("_mean_r_pure_vs_attack")
    mean_comm = mean_r_attack("comm")
    mean_scr = mean_r_attack("scramble")
    mean_flip = mean_r_attack("flip")

    tex = r"""\begin{table}[t]
\centering
\caption{Threshold-policy alignment by model and treatment. Cells report Pearson $r$ between the empirical join fraction and the benchmark attack mass $A(\theta)$ under $B=C=1$ (so $\theta^* = 0.50$); 95\% Fisher-$z$ confidence intervals in brackets.}
\label{tab:main_results}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Main treatments} & \multicolumn{2}{c}{Falsification} & & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Model & Pure & Comm & Scramble & Flip & $N$ (P/C/S/F) & Mean join (\%) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    # Build pooled cells with CIs
    def _r_cell_with_ci(r_val, ci_dict):
        cell = f"${_fmt_r(r_val, 2)}$"
        ci_lo, ci_hi = ci_dict.get("ci_lo"), ci_dict.get("ci_hi")
        if ci_lo is not None and ci_hi is not None:
            cell += f" {{\\scriptsize $[{ci_lo:.2f},{ci_hi:.2f}]$}}"
        return cell

    pooled_pure_cell = _r_cell_with_ci(pooled, pooled_pure_ci)
    pooled_comm_cell = _r_cell_with_ci(pooled_comm, pooled_comm_ci)
    pooled_scr_cell = _r_cell_with_ci(pooled_scr, pooled_scr_ci)
    pooled_flip_cell = _r_cell_with_ci(pooled_flip, pooled_flip_ci)

    tex += r"""\midrule
\textbf{Pooled} & """ + pooled_pure_cell + r""" & """ + pooled_comm_cell + r""" & """ + pooled_scr_cell + r""" & """ + pooled_flip_cell + r""" & """ + pooled_ns + r""" & """ + _fmt_pct(pooled_mean, 1) + r""" \\
\textbf{Mean across models} & """ + r_cell(mean_pure, 2) + r""" & """ + r_cell(mean_comm, 2) + r""" & """ + r_cell(mean_scr, 2) + r""" & """ + r_cell(mean_flip, 2) + r""" & --- & --- \\
\bottomrule
\end{tabular}
}
\begin{tablenotes}
\footnotesize\emph{Notes:} $r = r(J, A(\theta))$: Pearson correlation between empirical join fraction and theoretical attack mass under $B = C = 1$, with 95\% Fisher-$z$ confidence intervals in brackets for every treatment. Join fraction uses valid decisions only (excluding parse errors); mean join is the pure-treatment average in percent. $N$ counts period-level rows (not agents) in the pure (P), communication (C), scramble (S), and flip (F) arms. Pooled: all period-level rows concatenated; Mean: equal-weighted average across models. Duplicate-cell accounting is reported in the main-text footnote and Table~\ref{tab:comm_estimators}.
\end{tablenotes}
\end{table}
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
        r = (d.get("r_vs_attack") or {}).get("r")
        delta = d.get("delta_vs_baseline")
        n = d.get("n_obs")
        delta_cell = "---" if delta is None else _fmt_pp(delta, 1)
        rows.append(
            f"{label} & {_fmt_num(mean,3)} & ${_fmt_r(r,2)}$ & {delta_cell} & {n} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Information design treatment summary (primary model: Mistral Small Creative). $r$ is the Pearson correlation $r(J, A(\theta))$ between join fraction and theoretical attack mass.}
\label{tab:infodesign_summary}
\small
\begin{tabular}{lcccc}
\toprule
Design & Mean & $r$ & $\Delta$ (pp) & $N$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\vspace{0.25em}
\parbox{\columnwidth}{\footnotesize\emph{Notes:} Data from the primary model (pure treatment; $\theta \in [0.20, 0.80]$ on a 9-point grid; $N{=}25$ agents per period). Mean join uses valid decisions; $r$ is Pearson $r(J, A(\theta))$ across rep-level periods.}
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
        lines.append(f"{label} & {_fmt_num(no,3)} & {_fmt_num(yes,3)} & {_fmt_pp(delta,1)} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Surveillance $\times$ censorship interaction in the communication game (primary model: Mistral Small Creative).}
\label{tab:surv_censor}
\small
\begin{tabular}{lccc}
\toprule
Design & No Surv. & Surv. & $\Delta$ (pp) \\
\midrule
"""
    tex += "\n".join(lines) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\par\vspace{0.25em}
\parbox{\columnwidth}{\footnotesize\emph{Notes:} Both columns use the communication information-design grid ($\theta \in [0.20, 0.80]$, 9 points, communication game). The low baseline level (3.0\%) reflects this grid regime, not the Part~I normal-draw regime (${\approx}40\%$); see text for discussion. ``Surv.'' adds surveillance during messaging. All entries are means of \texttt{join\_fraction\_valid}. $\Delta$ is the surveillance increment (Surv.\ $-$ No Surv.) in percentage points.}
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
            return f"${_fmt_r(d['r_vs_attack']['r'], 2)}$"
        return "---"

    rows = []
    for model in order:
        rows.append(
            f"{model} & {cell(model,'baseline','mean')} & {cell(model,'baseline','r')} & "
            f"{cell(model,'scramble','mean')} & {cell(model,'scramble','r')} & "
            f"{cell(model,'flip','mean')} & {cell(model,'flip','r')} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Cross-model replication of key information design conditions. $r$ is the correlation $r(J, A(\theta))$ between join fraction and theoretical attack mass.}
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
\vspace{0.25em}
\parbox{\textwidth}{\footnotesize\emph{Notes:} Information design $\theta$-grid $[0.20, 0.80]$. $r = r(J, A(\theta))$: Pearson correlation between join fraction and theoretical attack mass. $n = 25$ agents per period. Scramble mean join rates for some models reflect earlier experimental batches and are not directly comparable to baseline means; the diagnostic metric is $r \approx 0$.}
\end{table}
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
        r = d["r_vs_attack"]["r"]
        delta = d.get("delta_vs_baseline")
        rows.append(f"{label} & {_fmt_num(mean,3)} & ${_fmt_r(r,2)}$ & {_fmt_pp(delta,1)} \\\\")

    # Sum of single-channel deltas vs full delta
    deltas = [info.get(k, {}).get("delta_vs_baseline") for k in ["stability_clarity", "stability_direction", "stability_dissent"]]
    sum_delta = sum(float(x) for x in deltas if x is not None)
    full_delta = float(info["stability"]["delta_vs_baseline"])

    rows.append(r"\midrule")
    rows.append(f"Sum of channels & --- & --- & {_fmt_pp(sum_delta,1)} \\\\")
    rows.append(f"Full design & --- & --- & {_fmt_pp(full_delta,1)} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Single-channel decomposition of the stability design (primary model: Mistral Small Creative).}
\label{tab:decomposition}
\small
\begin{tabular}{lccc}
\toprule
Channel & Mean & $r$ & $\Delta$ (pp) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\vspace{0.25em}
\parbox{\columnwidth}{\footnotesize\emph{Notes:} Each row is a separate infodesign run for Mistral Small Creative on the same $\theta$ grid as Table~\ref{tab:infodesign_summary}. $\Delta$ reports the mean difference vs.\ the baseline infodesign mean (Table~\ref{tab:infodesign_summary}).}
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
    _short = {
        "Mistral Small Creative": "Mistral",
        "Llama 3.3 70B": "Llama 3.3 70B",
        "GPT-OSS 120B": "GPT-OSS",
        "Qwen3 235B": "Qwen3 235B",
    }

    rows = []
    for m in models:
        d = sxc.get(m, {})
        short = _short.get(m, m)
        if not d:
            rows.append(f"{short} & --- & --- & --- & --- & --- \\\\")
            continue
        bl = d.get("baseline")
        cu = d.get("censor_upper")
        cl = d.get("censor_lower")
        bl_cell = _fmt_num(bl, 3)
        cu_cell = _fmt_num(cu, 3)
        cl_cell = _fmt_num(cl, 3)
        delta_u = (cu - bl) if bl is not None and cu is not None else None
        delta_l = (cl - bl) if bl is not None and cl is not None else None
        du_cell = _fmt_pp(delta_u, 1) if delta_u is not None else "---"
        dl_cell = _fmt_pp(delta_l, 1) if delta_l is not None else "---"
        rows.append(f"{short} & {bl_cell} & {cu_cell} & {cl_cell} & {du_cell} & {dl_cell} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Cross-model surveillance $\times$ censorship interaction. $\Delta$: change vs.\ surveilled baseline (pp).}
\label{tab:surv_censor_crossmodel}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{Mean join (surv.)} & \multicolumn{2}{c}{$\Delta$ (pp)} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-6}
Model & Base & Upper & Lower & $\Delta_U$ & $\Delta_L$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\vspace{0.25em}
\parbox{\columnwidth}{\footnotesize\emph{Notes:} Communication treatment, information design $\theta$-grid, surveillance active during messaging. $\Delta$: change vs.\ surveilled baseline (pp). $n = 25$ agents per period.}
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

    part1 = stats.get("part1", {})

    def _n(m: str, t: str) -> str:
        d = part1.get(m, {}).get(t, {})
        n = d.get("n_obs") if isinstance(d, dict) else None
        return str(n) if n is not None else "---"

    rows = []
    for m in MODEL_ORDER:
        if m not in fits:
            continue
        p = fits[m].get("pure")
        c = fits[m].get("comm")
        rows.append(
            f"{m} & {_cell(p, 'cutoff', 'se_cutoff')} & {_cell(p, 'b1', 'se_b1')} & {_n(m, 'pure')}"
            f" & {_cell(c, 'cutoff', 'se_cutoff')} & {_cell(c, 'b1', 'se_b1')} & {_n(m, 'comm')} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Logistic fit parameters by model and treatment. $\hat{\theta}^*$ is the estimated cutoff ($-b_0/b_1$); $\kappa \equiv b_1$ is the steepness parameter.}
\label{tab:logistic_params}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{Pure} & \multicolumn{3}{c}{Communication} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Model & $\hat{\theta}^*$ (SE) & $\kappa$ (SE) & $N$ & $\hat{\theta}^*$ (SE) & $\kappa$ (SE) & $N$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
}
\begin{tablenotes}
\footnotesize\emph{Notes:} Fitted form: $P(\text{join}\mid\theta) = 1/(1+e^{b_0 + b_1\theta})$. $\hat{\theta}^* = -b_0/b_1$: estimated cutoff. $\kappa \equiv b_1$: logistic steepness (larger = sharper threshold). Sign convention: positive $\kappa$ corresponds to a join curve decreasing in $\theta$; in the agent-level logit (Table~\ref{tab:regressions}) the same comparative static appears as a negative coefficient on $\theta$. $N$ counts period-level rows entering each fit. Standard errors from the covariance matrix of the nonlinear fit; cutoff SE by delta method.
\end{tablenotes}
\end{table}
"""
    return tex


def render_tab_surveillance_variants(stats: dict) -> str:
    sv = stats.get("surveillance_variants", {})
    if not sv:
        return "% No surveillance variant data available.\n"

    by_model = sv.get("by_model", {}) or {}
    model_order = [("Mistral Small Creative", "Mistral"), ("Llama 3.3 70B", "Llama 3.3 70B")]

    rows = []
    baseline_notes = []
    for model, short in model_order:
        m_data = by_model.get(model) or {}
        # Fall back to top-level entries for the primary model
        if not m_data and model == "Mistral Small Creative":
            m_data = {v: sv.get(v, {}) for v in ["placebo", "anonymous"]}
        emitted = False
        base_n = None
        base_mean = None
        for variant in ["placebo", "anonymous"]:
            d = m_data.get(variant) or {}
            if "mean_join" not in d:
                continue
            mean_j = _fmt_pct(d["mean_join"], 1)
            r_val = f"${_fmt_r(d['r_vs_attack']['r'], 2)}$"
            delta = _fmt_pp_raw(d.get("delta_vs_comm_pp"), 1)
            if delta != "---":
                delta = f"${delta}$"
            t_test = d.get("t_test_vs_comm", {}) or {}
            p_val = _fmt_p(t_test.get("p_value"), 3)
            label = "Placebo" if variant == "placebo" else "Anonymous"
            model_cell = short if not emitted else ""
            emitted = True
            base_n = d.get("baseline_n_obs", base_n)
            base_mean = d.get("baseline_mean_join", base_mean)
            rows.append(f"{model_cell} & {label} & {d['n_obs']} & {mean_j} & {r_val} & {delta} & {p_val} \\\\")
        if emitted and base_n is not None and base_mean is not None:
            baseline_notes.append(
                f"{short}: $N = {base_n}$, mean join {_fmt_pct(base_mean, 1)}"
            )
        if emitted:
            rows.append(r"\addlinespace")
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()

    baseline_sentence = ""
    if baseline_notes:
        baseline_sentence = (
            " Communication-baseline arms: " + "; ".join(baseline_notes) + "."
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Surveillance isolation checks. Placebo: monitored for research, no consequences. Anonymous: messages aggregated anonymously. Neither deviates significantly from the communication baseline.}
\label{tab:surveillance_variants}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{llccccc}
\toprule
Model & Variant & $N$ & Mean join & $r(J,A)$ & $\Delta$ (pp) & $p$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Placebo: agents told monitoring is for research purposes only (no consequences). Anonymous: messages aggregated before delivery. $N$ counts country--period rows ($n = 25$ agents per row). $\Delta$: change vs.\ the same model's communication baseline (pp), with $p$ from a two-sample $t$-test against that baseline.""" + baseline_sentence + r"""
\end{tablenotes}
\end{table}
"""
    return tex


def render_tab_prompt_isolation(stats: dict) -> str:
    """Clean baseline-plus-warning surveillance reruns by model."""
    pi = stats.get("prompt_isolation", {}).get("surveillance", {})
    summary = stats.get("prompt_isolation", {}).get("_summary", {})
    if not pi:
        return "% No clean prompt-isolation surveillance data available.\n"

    def _se_cell(delta_pp, t_stat) -> str:
        if delta_pp is None or t_stat in (None, 0):
            return "---"
        try:
            return f"{abs(float(delta_pp) / float(t_stat)):.1f}"
        except Exception:
            return "---"

    rows = []
    for model in DISPLAY_ORDER:
        d = pi.get(model)
        if not d:
            continue
        matched = d.get("matched", {}) or {}
        p_val = _fmt_p(matched.get("p_value"), 3) if isinstance(matched, dict) else "---"
        delta_cell = _fmt_pp_raw(matched.get("delta_pp"), 1)
        if delta_cell != "---":
            delta_cell = f"${delta_cell}$"
        rows.append(
            f"{model} & {matched.get('baseline_unique_cells', '---')}/{matched.get('surveillance_unique_cells', '---')} & "
            f"{matched.get('matched_cells', '---')} & "
            f"{_fmt_pct(matched.get('baseline_mean_join'), 1)} & "
            f"{_fmt_pct(matched.get('mean_join'), 1)} & "
            f"{delta_cell} & "
            f"{_se_cell(matched.get('delta_pp'), matched.get('t_stat'))} & "
            f"${_fmt_r((d.get('r_vs_attack') or {}).get('r'), 2)}$ & {p_val} \\\\"
        )

    rows.append(r"\midrule")
    rows.append(
        "Equal-weight avg & --- & --- & --- & --- & $"
        + _fmt_pp_raw(summary.get("mean_delta_pp"), 1)
        + "$ & --- & --- & --- \\\\"
    )
    rows.append(
        "Pooled matched & --- & "
        + str(summary.get("pooled_matched_n", "---"))
        + " & --- & --- & $"
        + _fmt_pp_raw(summary.get("pooled_matched_delta_pp"), 1)
        + "$ & "
        + _se_cell(summary.get("pooled_matched_delta_pp"), summary.get("pooled_matched_t_stat"))
        + " & --- & "
        + _fmt_p(summary.get("pooled_matched_p_value"), 3)
        + " \\\\"
    )

    # Country-clustered robustness for the primary model (headline effect)
    cc = ((pi.get("Mistral Small Creative") or {}).get("matched") or {}).get("country_clustered") or {}
    cc_note = ""
    if cc.get("t_stat") is not None:
        cc_note = (
            " Aggregating the primary model's matched-cell deltas to country means "
            f"($N = {cc.get('n_countries')}$ countries) gives $t = {cc['t_stat']:.2f}$, "
            f"$p {_fmt_p_text_plain(cc.get('p_value'))}$, so the headline effect survives "
            "country-level clustering."
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Clean prompt-isolation surveillance reruns. The surveillance prompt is the baseline trusted-contact communication prompt plus one appended monitoring warning; the decision prompt is unchanged.}
\label{tab:prompt_isolation}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccc}
\toprule
Model & Full cells (B/S) & Matched cells & Baseline & Surv. & $\Delta$ (pp) & SE & $r(J,A)$ & $p$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Baseline is each model's communication treatment. ``Full cells'' counts the available baseline and surveillance task cells after collapsing duplicate rows by the exact key (country, period, $\theta$, $z$, benefit, $\theta^*$); it is not the support used for the paired effect when common support is smaller. The non-primary baseline communication arms were shorter pilot grids (100--200 cells) while the later surveillance reruns used a larger grid; because those grids were not nested, only 20 exact cells overlap for several models. $N$ for each paired contrast is the number of matched common-support cells; Baseline, Surv., $\Delta$, SE, and $p$ are computed only on those cells, with $p$ from a paired one-sample $t$-test of cell-level (surveillance $-$ baseline) differences and SE the paired standard error of $\Delta$ in pp. $r(J,A)$ is the surveillance-arm threshold-alignment correlation on the full surveillance rerun.""" + cc_note + r"""
\end{tablenotes}
\end{table}
"""
    return tex


def render_tab_bc_statics(stats: dict) -> str:
    """B/C comparative statics: cutoff shifts under strategic-stakes narratives."""
    info = stats.get("infodesign", {})
    designs = ["baseline", "bc_high_cost", "bc_low_cost"]
    labels = {
        "baseline": "Baseline (no header)",
        "bc_high_cost": "High-cost header",
        "bc_low_cost": "Low-cost header",
    }

    baseline_mean = (info.get("baseline") or {}).get("mean_join")
    rows = []
    for d in designs:
        di = info.get(d, {})
        if not di:
            continue
        mean_j = _fmt_pct(di["mean_join"], 1)
        r_val = f"${_fmt_r(di['r_vs_attack']['r'], 2)}$"
        fit = di.get("logistic_fit") or {}
        cutoff = fit.get("cutoff")
        se_cutoff = fit.get("se_cutoff")
        cutoff_cell = "---"
        if cutoff is not None and se_cutoff is not None:
            cutoff_cell = f"${cutoff:+.2f}$ ({se_cutoff:.3f})"
        n = di.get("n_obs", "---")
        if d == "baseline":
            delta = "---"
        else:
            # Difference of displayed percentage endpoints for internal consistency
            dd = _delta_of_displayed(
                (baseline_mean or 0) * 100, di["mean_join"] * 100, 1
            )
            delta = f"${_fmt_pp_raw(dd, 1)}$" if dd is not None else "---"
        rows.append(f"{labels[d]} & {n} & {mean_j} & {r_val} & {cutoff_cell} & {delta} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Strategic-stakes narrative-header comparative statics. High-cost framing emphasizes severe reprisals for failed action; low-cost framing emphasizes minimal consequences. These rows are the two-header narrative contrast, not the seven-cell numeric payoff ladder.}
\label{tab:bc_statics}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lccccc}
\toprule
Design & $N$ & Mean join & $r(J,A)$ & $\hat{\theta}^*$ (SE) & $\Delta$ (pp) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Primary model, information-design $\theta$-grid. $N$ counts country--period rows ($n = 25$ agents per row). Identical briefing bodies across conditions; only the prepended strategic-stakes header varies. The high/low rows are the original header conditions \texttt{bc\_high\_cost} and \texttt{bc\_low\_cost}; they are not the endpoints of the separate narrative ladder reported in Table~\ref{tab:bc_sweep_mapping}. $\Delta$: change in mean join vs.\ baseline (pp, actual $-$ baseline); cutoff SEs are delta-method standard errors from the logistic fit (per-condition mean-join SEs are not stored in the verified statistics). Note the sign convention relative to Table~\ref{tab:bc_classifier}: there, Gap $=$ classifier $-$ actual, and because the classifier's prediction stays near the baseline rate, a header that lowers actual joining appears here as a negative $\Delta$ and there as a positive Gap of approximately the same magnitude.
\end{tablenotes}
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
        r_val = _fmt_r(di["r_vs_attack"]["r"], 2)
        n = di.get("n_obs", "---")
        delta = ""
        if "delta_vs_baseline" in di:
            delta = _fmt_pp(di["delta_vs_baseline"], 1)
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
Design & $N$ & Mean join & $r(J, A(\theta))$ & $\Delta$ (pp) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\vspace{0.25em}
\parbox{\columnwidth}{\footnotesize\emph{Notes:} Primary model, information design $\theta$-grid. ``Known'': agents are told that the regime censors intelligence above a severity threshold. $\Delta$: change vs.\ baseline (pp).}
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
        r_val = _fmt_r(d["r_vs_attack"]["r"], 2)
        n = d.get("n_obs", "---")
        fit = d.get("logistic_fit", {})
        cutoff = _fmt_num(fit.get("cutoff"), 3) if fit.get("cutoff") is not None else "---"
        slope = _fmt_num(fit.get("b1"), 2) if fit.get("b1") is not None else "---"
        rows.append(f"{key} & {n} & {mean_j} & {r_val} & {cutoff} & {slope} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Temperature robustness. The pure global game is run at three LLM decoding temperatures using Mistral Small Creative. The correlation $r(J, A(\theta))$ and logistic parameters are stable across temperatures.}
\label{tab:temperature}
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Temperature & $N$ & Mean join & $r(J, A(\theta))$ & Cutoff $\hat{\theta}^*$ & Slope $\hat{\beta}$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}}
\vspace{0.25em}
\parbox{\columnwidth}{\footnotesize\emph{Notes:} Primary model (Mistral Small Creative), pure treatment, varying LLM decoding temperature.}
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


def _fmt_pp_raw(x: float, nd: int = 2) -> str:
    """Format a value already expressed in percentage points."""
    if x is None:
        return "---"
    try:
        if x != x:  # nan
            return "---"
    except Exception:
        return "---"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{nd}f}"


def _fmt_pp_math(x: float, nd: int = 2) -> str:
    """Math-mode signed percentage-point value (minus signs render correctly)."""
    s = _fmt_pp_raw(x, nd)
    return s if s == "---" else f"${s}$"


def _fmt_arm_counts(pure: int | None, comm: int | None) -> str:
    """Format arm-specific counts compactly when they agree."""
    if pure is None and comm is None:
        return "---"
    if pure == comm:
        return str(pure)
    return f"{pure}/{comm}"


def _fmt_arm_pct(pure: float | None, comm: float | None, nd: int = 1) -> str:
    """Format arm-specific shares as percentages."""
    pure_s = _fmt_pct(pure, nd)
    comm_s = _fmt_pct(comm, nd)
    if pure_s == "---" and comm_s == "---":
        return "---"
    if pure_s == comm_s:
        return pure_s
    return f"{pure_s}/{comm_s}"


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
\caption{Common knowledge $\times$ coordination intensity. Each cell reports mean join rate (270 country--periods). The CK main effect is """ + f"{pp(ck_main.get('beta'), 0) if ck_main.get('beta') is not None else '---'}" + r"""~pp ($p = """ + f"{ck_main.get('p', '---'):.4f}" + r"""$); the interaction is """ + f"{pp(inter_beta, 0) if inter_beta is not None else '---'}" + r"""~pp ($p = """ + f"{inter_p:.2f}" + r"""$). This is a header-only framing test, not the main public-signal bulletin treatment.}
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
\vspace{0.25em}
\parbox{\columnwidth}{\footnotesize\emph{Notes:} Primary model, information design $\theta$-grid. 270 country-periods per cell. CK framing changes only the publicness/source header (``widely shared''); the private briefing body is otherwise unchanged. High-coord: coordination-cue intensity amplified. Unlike the main public-signal treatment, no separate public bulletin is appended.}
\end{table}
"""
    return tex


def render_tab_comm_estimators(stats: dict) -> str:
    """Break down why communication estimators differ across aggregation rules."""
    part1 = stats.get("part1", {})
    decomp = ((part1.get("_pooled_comm_effect") or {}).get("decomposition") or {})
    if not decomp:
        return "% No communication estimator decomposition available.\n"

    row_map = {row.get("model"): row for row in decomp.get("rows_by_model", [])}
    est = decomp.get("estimators_pp") or {}
    totals = decomp.get("totals") or {}
    n_models = max(len(row_map), 1)
    equal_weight_label = f"{100 / n_models:.1f}\\% each"

    rows = []
    for model in DISPLAY_ORDER:
        row = row_map.get(model, {})
        rows.append(
            " & ".join(
                [
                    model,
                    _fmt_arm_counts(row.get("pure_rows"), row.get("comm_rows")),
                    _fmt_arm_pct(row.get("pure_row_share"), row.get("comm_row_share"), 1),
                    _fmt_arm_counts(row.get("pure_unique_cells"), row.get("comm_unique_cells")),
                    str(row.get("matched_cells", "---")),
                    _fmt_pct(row.get("matched_cell_share"), 1),
                    f"{row.get('pure_unmatched_cells', '---')}/{row.get('comm_unmatched_cells', '---')}",
                    _fmt_pp_math(row.get("unpaired_delta_pp"), 2),
                    _fmt_pp_math(row.get("paired_delta_pp"), 2),
                ]
            )
            + r" \\"
        )

    rows.append(r"\midrule")
    rows.append(
        " & ".join(
            [
                "Equal-weight avg",
                "---",
                equal_weight_label,
                "---",
                "---",
                equal_weight_label,
                "---",
                _fmt_pp_math(est.get("equal_weight_unpaired"), 2),
                _fmt_pp_math(est.get("equal_weight_paired"), 2),
            ]
        )
        + r" \\"
    )
    rows.append(
        " & ".join(
            [
                "Pooled",
                _fmt_arm_counts(totals.get("pure_rows"), totals.get("comm_rows")),
                "100.0\\%",
                _fmt_arm_counts(totals.get("pure_unique_cells"), totals.get("comm_unique_cells")),
                str(totals.get("matched_cells", "---")),
                "100.0\\%",
                f"{totals.get('pure_unmatched_cells', '---')}/{totals.get('comm_unmatched_cells', '---')}",
                _fmt_pp_math(est.get("pooled_unpaired"), 2),
                _fmt_pp_math(est.get("pooled_paired"), 2),
            ]
        )
        + r" \\"
    )

    match_key_labels = {
        "model": "model",
        "country": "country",
        "period": "period",
        "theta": r"$\theta$",
        "z": r"$z$",
        "benefit": "benefit",
        "theta_star": r"$\theta^*$",
    }
    match_key = ", ".join(match_key_labels.get(col, col.replace("_", r"\_")) for col in decomp.get("match_key", []))

    # Paired pooled inference (SE from t-stat) for the notes
    paired = (part1.get("_pooled_comm_effect") or {}).get("paired") or {}
    paired_note = ""
    if paired.get("delta_pp") is not None and paired.get("t_stat"):
        se_pp = abs(float(paired["delta_pp"]) / float(paired["t_stat"]))
        paired_note = (
            f" The pooled paired $\\Delta$ of ${paired['delta_pp']:+.2f}$~pp has a paired standard error of "
            f"{se_pp:.2f}~pp ($t = {paired['t_stat']:.2f}$, "
            f"$p {_fmt_p_text_plain(paired.get('p_value'))}$, $N = {_fmt_int_sep(paired.get('n_pairs'))}$ matched cells)."
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Communication estimators: aggregation accounting.}
\label{tab:comm_estimators}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Model & Rows/arm & Row wt. (P/C) & Cells/arm & Matched & Cell wt. & Lost (P/C) & $\Delta$ unpaired (pp) & $\Delta$ paired (pp) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
}
\begin{tablenotes}
\footnotesize\emph{Notes:} Sample: valid country--period rows from the core pure and communication suites, each aggregating $n = 25$ agent decisions. ``Rows/arm'' counts valid country-period rows entering the pooled unpaired estimator in each arm. ``Cells/arm'' counts unique task cells after collapsing duplicates by the exact matching key (""" + match_key + r"""). The paired estimator averages within these cells and keeps only common support; ``Row wt.'' and ``Cell wt.'' are the model shares in the pooled unpaired and pooled paired estimators, respectively.""" + paired_note + r""" Trinity Large has one pure-only cell. Qwen3 235B has 800 communication-only cells because its communication arm is larger than its pure arm; those cells are excluded from the paired common-support estimator.
\end{tablenotes}
\end{table}
"""
    return tex


def render_tab_msg_features(stats: dict) -> str:
    """Top discriminative bigrams between baseline and surveillance messages."""
    mc = stats.get("message_content", {}) or {}
    clf = mc.get("classifier", {}) or {}
    if not clf or clf.get("status") == "error" or not clf.get("top_surv_features"):
        return "% No message-content classifier data available.\n"

    def _esc(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    surv = clf.get("top_surv_features", [])[:10]
    base = clf.get("top_base_features", [])[:10]
    weather = mc.get("weather_token_audit", {}) or {}
    weather_note = ""
    if weather:
        weather_note = (
            f" The weather terms are organic peer-message metaphors, not placebo leakage: "
            f"{_fmt_int_sep(weather.get('base_count'))} baseline messages "
            f"({weather.get('base_pct', '---')}\\%) and "
            f"{_fmt_int_sep(weather.get('surv_count'))} surveillance messages "
            f"({weather.get('surv_pct', '---')}\\%) contain the token \\texttt{{weather}}."
        )
    rows = []
    for i in range(max(len(surv), len(base))):
        s = surv[i] if i < len(surv) else {"feature": "", "coef": None}
        b = base[i] if i < len(base) else {"feature": "", "coef": None}
        s_feat = f"\\texttt{{{_esc(s['feature'])}}}" if s["feature"] else ""
        s_coef = f"${s['coef']:+.2f}$" if s["coef"] is not None else ""
        b_feat = f"\\texttt{{{_esc(b['feature'])}}}" if b["feature"] else ""
        b_coef = f"${b['coef']:+.2f}$" if b["coef"] is not None else ""
        rows.append(f"{s_feat} & {s_coef} & {b_feat} & {b_coef} \\\\")

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\caption{Top 10 most discriminative uni- and bi-grams between baseline and surveillance peer messages (primary model). A logistic classifier on tf-idf features achieves \\MsgClassifierAcc\\% accuracy on a held-out sample of $N = \\MsgClassifierNTest$.}")
    out.append("\\label{tab:msg_features}")
    out.append("\\footnotesize")
    out.append("\\begin{tabular}{lc lc}")
    out.append("\\toprule")
    out.append("\\multicolumn{2}{c}{Surveillance-distinguishing} & \\multicolumn{2}{c}{Baseline-distinguishing} \\\\")
    out.append("\\cmidrule(lr){1-2}\\cmidrule(lr){3-4}")
    out.append("Feature & Coef & Feature & Coef \\\\")
    out.append("\\midrule")
    out.extend(rows)
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\begin{tablenotes}")
    out.append("\\footnotesize\\emph{Notes:} Logistic regression on tf-idf uni- and bi-grams (max 5{,}000 features, min document frequency 10). Train/test split 70/30, stratified ($N_{\\text{train}} = \\MsgClassifierNTrain$, $N_{\\text{test}} = \\MsgClassifierNTest$). Positive coefficients distinguish surveillance messages, negative coefficients distinguish baseline. The sample contains only baseline communication and clean surveillance peer messages; placebo/weather-stakes conditions are excluded." + weather_note + " Direct-reference and coded-metaphor claims in the text come from separate dictionary counts, not from this top-n-gram list alone.")
    out.append("\\end{tablenotes}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


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
            return f"${g:+.1f}$"
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

    # Sample sizes for the notes
    primary_pure_rows = (
        (stats.get("part1", {}).get("Mistral Small Creative") or {}).get("pure") or {}
    ).get("n_obs")
    surv_rows = (
        (stats.get("regime_control", {}).get("surveillance") or {}).get("Mistral Small Creative") or {}
    ).get("n_obs")
    n_note = ""
    if primary_pure_rows and surv_rows:
        n_note = (
            f" Training sample: the primary model's pure arm "
            f"({_fmt_int_sep(primary_pure_rows)} country--period rows, 25 agents per row, "
            f"${_fmt_int_sep(int(primary_pure_rows) * 25)}$ agent decisions); each CV fold trains on 80\\% and tests "
            f"on the held-out 20\\%. The surveillance application scores all "
            f"{_fmt_int_sep(surv_rows)} surveillance-arm rows "
            f"(${_fmt_int_sep(int(surv_rows) * 25)}$ agent decisions)."
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Classifier baselines. Accuracy and AUC are 5-fold CV on pure-treatment data. ``Pred.\ join (surv.)'' is the classifier's predicted join rate when applied to surveillance-treatment briefings; ``Actual'' is the LLM's observed rate. The gap measures the surveillance action shift invisible to briefing-body text classifiers.}
\label{tab:classifiers}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lccccc}
\toprule
Classifier & Acc. & AUC & Pred.\ (surv.) & Actual & Gap \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Primary model (Mistral Small Creative). Accuracy and AUC from 5-fold cross-validation on pure-treatment data.""" + n_note + r""" Gap = classifier-predicted join rate $-$ actual LLM join rate under surveillance (pp).
\end{tablenotes}
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
        """For values stored as fractions (e.g. 0.09 → +9.0)."""
        return f"\\providecommand{{\\{name}}}{{{_fmt_pp(val, nd)}}}"

    def _mc_pp_raw(name, val, nd=1):
        """For values already in percentage points (e.g. -13.5 → -13.5)."""
        if val is None:
            return f"\\providecommand{{\\{name}}}{{---}}"
        try:
            if val != val:  # nan
                return f"\\providecommand{{\\{name}}}{{---}}"
        except Exception:
            return f"\\providecommand{{\\{name}}}{{---}}"
        sign = "+" if val >= 0 else ""
        return f"\\providecommand{{\\{name}}}{{{sign}{val:.{nd}f}}}"

    def _mc_pp_abs_raw(name, val, nd=1):
        """Absolute value for signed percentage-point effects used in prose."""
        if val is None:
            return f"\\providecommand{{\\{name}}}{{---}}"
        try:
            if val != val:  # nan
                return f"\\providecommand{{\\{name}}}{{---}}"
        except Exception:
            return f"\\providecommand{{\\{name}}}{{---}}"
        return f"\\providecommand{{\\{name}}}{{{abs(val):.{nd}f}}}"

    def _mc_pct(name, val, nd=1):
        return f"\\providecommand{{\\{name}}}{{{_fmt_pct(val, nd)}}}"

    def _mc_raw(name, val_str):
        return f"\\providecommand{{\\{name}}}{{{val_str}}}"

    def _fmt_int_text(val):
        if val is None:
            return "---"
        try:
            return f"{int(val):,}"
        except Exception:
            return "---"

    def _fmt_p_text(val, nd=3):
        """Format a p-value for inline prose as either an inequality or equality."""
        if val is None:
            return "---"
        try:
            if val != val:  # nan
                return "---"
        except Exception:
            return "---"
        threshold = 10 ** (-nd)
        if val < threshold:
            return f"<{threshold:.{nd}f}"
        return f"= {val:.{nd}f}"

    def _mc_p_text(name, val, nd=3):
        return f"\\providecommand{{\\{name}}}{{{_fmt_p_text(val, nd)}}}"

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
    pooled_comm_effect = part1.get("_pooled_comm_effect") or {}
    pooled_delta_pp = (pooled_comm_effect.get("unpaired") or {}).get("delta_pp")
    pooled_pval = (pooled_comm_effect.get("unpaired") or {}).get("p_value")
    comm_decomp = pooled_comm_effect.get("decomposition") or {}
    comm_totals = comm_decomp.get("totals") or {}
    comm_est = comm_decomp.get("estimators_pp") or {}

    # Surveillance × censorship: primary model (Mistral) join levels.
    sxc = ((regime.get("surveillance_x_censorship") or {}).get("Mistral Small Creative") or {})

    lines = []
    lines.append("% Auto-generated from analysis/verified_stats.json. Do not edit by hand.")
    lines.append("% Generated by: uv run python analysis/render_paper_tables.py")
    lines.append("")

    # ── Paper design counts ──────────────────────────────────────
    primary_display = DISPLAY_NAMES.get(PRIMARY_SLUG, PRIMARY_SLUG)
    primary_pure_n = ((part1.get(primary_display) or {}).get("pure") or {}).get("n_obs")
    lines.append("% Paper design counts")
    lines.append(_mc_raw("PartOneNModels", str(len(DISPLAY_ORDER))))
    lines.append(_mc_raw("PartOneNFamilies", "6"))
    lines.append(_mc_raw("PrimaryPureNObs", _fmt_int_text(primary_pure_n)))
    lines.append(_mc_raw("PrimaryPureNObsMath", str(primary_pure_n if primary_pure_n is not None else "---")))
    lines.append("")

    # ── Communication summary ─────────────────────────────────────
    lines.append("% Communication summary")
    lines.append(_mc("CommPureMeanModelAvg", mean_pure_models))
    lines.append(_mc("CommCommMeanModelAvg", mean_comm_models))
    lines.append(_mc_pp("CommDeltaPPModelAvg", delta_models))
    lines.append(_mc_pp_raw("CommDeltaPPModelAvgPaired", comm_est.get("equal_weight_paired"), 2))
    lines.append(_mc("CommPureMeanPooled", pooled_pure.get("mean_join")))
    lines.append(_mc("CommCommMeanPooled", pooled_comm.get("mean_join")))
    if pooled_delta_pp is None:
        lines.append(_mc_raw("CommDeltaPPPooled", "---"))
    else:
        sign = "+" if pooled_delta_pp >= 0 else ""
        lines.append(_mc_raw("CommDeltaPPPooled", f"{sign}{pooled_delta_pp:.2f}"))
    lines.append(_mc("CommPValueUnpaired", pooled_pval))
    # Paired test (matched on model/country/period/theta/z/benefit/theta_star)
    paired = (pooled_comm_effect.get("paired") or {})
    paired_delta = paired.get("delta_pp")
    paired_t = paired.get("t_stat")
    paired_p = paired.get("p_value")
    paired_n = paired.get("n_pairs")
    if paired_delta is not None:
        sign = "+" if paired_delta >= 0 else ""
        lines.append(_mc_raw("CommDeltaPPPaired", f"{sign}{paired_delta:.2f}"))
    else:
        lines.append(_mc_raw("CommDeltaPPPaired", "---"))
    lines.append(_mc("CommTStatPaired", paired_t))
    lines.append(_mc("CommPValuePaired", paired_p))
    lines.append(_mc_p_text("CommPValuePairedText", paired_p))
    if paired_n is not None:
        lines.append(_mc_raw("CommNPairs", str(paired_n)))
    if comm_totals.get("pure_unique_cells") is not None:
        lines.append(_mc_raw("CommPureCells", str(comm_totals.get("pure_unique_cells"))))
    if comm_totals.get("comm_unique_cells") is not None:
        lines.append(_mc_raw("CommCommCells", str(comm_totals.get("comm_unique_cells"))))
    if comm_totals.get("matchable_cell_ceiling") is not None:
        lines.append(_mc_raw("CommMatchableCells", str(comm_totals.get("matchable_cell_ceiling"))))
    if comm_totals.get("pure_unmatched_cells") is not None:
        lines.append(_mc_raw("CommPureOnlyCells", str(comm_totals.get("pure_unmatched_cells"))))
    if comm_totals.get("comm_unmatched_cells") is not None:
        lines.append(_mc_raw("CommCommOnlyCells", str(comm_totals.get("comm_unmatched_cells"))))
    lines.append(_mc_pct("CommMatchedShare", comm_totals.get("matched_share_of_ceiling")))
    lines.append(_mc_pct("CommPureSupportShare", comm_totals.get("pure_support_retention")))
    lines.append(_mc_pct("CommCommSupportShare", comm_totals.get("comm_support_retention")))
    lines.append(_mc_pp_raw("CommRowWeightedPairedPP", comm_est.get("row_weighted_paired"), nd=2))
    lines.append(_mc_pp_raw("CommWithinModelReavgPP", comm_est.get("within_model_reavg"), nd=2))
    lines.append(_mc_pp_raw("CommCrossModelReweightPP", comm_est.get("cross_model_reweight"), nd=2))
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
    pure_ps = [
        (((part1.get(model) or {}).get("pure") or {}).get("r_vs_attack") or {}).get("p")
        for model in DISPLAY_ORDER
    ]
    pure_ps = [p for p in pure_ps if p is not None]
    lines.append(_mc_p_text("PureAllModelsAttackPText", max(pure_ps) if pure_ps else None))
    # Mean-of-models comm r_attack
    comm_rs = [
        (v.get("comm", {}).get("r_vs_attack") or {}).get("r")
        for k, v in part1.items()
        if isinstance(k, str) and not k.startswith("_") and isinstance(v, dict) and "comm" in v
    ]
    comm_rs = [x for x in comm_rs if x is not None]
    mean_comm_r = sum(comm_rs) / len(comm_rs) if comm_rs else None
    lines.append(_mc_r("MeanModelRCommAttack", mean_comm_r))
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

    # ── Primary model logistic fit ──────────────────────────────────
    logistic_fits = stats.get("logistic_fits", {})
    primary_fit = (logistic_fits.get(primary_display) or {}).get("pure", {})
    lines.append("% Primary model (Mistral) logistic fit")
    lines.append(_mc("PrimaryPureCutoff", primary_fit.get("cutoff"), 2))
    lines.append(_mc("PrimaryPureSlope", primary_fit.get("b1"), 2))
    lines.append("")

    # ── Pooled OLS ────────────────────────────────────────────────
    ols = stats.get("pooled_ols", {})
    lines.append("% Pooled OLS (join_fraction J regressed on attack mass A(theta))")
    lines.append(_mc("PooledOLSIntercept", ols.get("intercept")))
    lines.append(_mc("PooledOLSSlope", ols.get("slope"), 4))
    lines.append(_mc("PooledOLSRSq", ols.get("r_squared"), 4))
    lines.append(_mc_raw("PooledOLSNObs", _fmt_int_text(ols.get("n_obs"))))
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
    n_country_clusters = (clust.get("clustered_country") or {}).get("n_clusters")
    lines.append(_mc("NCountryClusters", n_country_clusters, 0))
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
        # Comm r-attack per model
        comm = entry.get("comm", {})
        r_comm_attack = (comm.get("r_vs_attack") or {}).get("r")
        lines.append(_mc_r(f"{mname}CommRAttack", r_comm_attack))
    lines.append("")

    # ── Fisher z-tests (pure vs scramble/flip) ────────────────────
    fisher_scr = part1.get("_fisher_pure_vs_scramble_attack", {})
    fisher_flip = part1.get("_fisher_pure_vs_flip_attack", {})
    lines.append("% Fisher z-tests")
    lines.append(_mc("FisherPureVsScrambleZ", fisher_scr.get("z"), 2))
    lines.append(_mc("FisherPureVsScrambleP", fisher_scr.get("p"), 4))
    lines.append(_mc_p_text("FisherPureVsScramblePText", fisher_scr.get("p")))
    lines.append(_mc("FisherPureVsFlipZ", fisher_flip.get("z"), 2))
    lines.append(_mc("FisherPureVsFlipP", fisher_flip.get("p"), 4))
    lines.append(_mc_p_text("FisherPureVsFlipPText", fisher_flip.get("p")))
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
        ("censor_upper_known", "InfodesignCensorUpperKnown"),
        ("hard_scramble", "InfodesignHardScramble"),
        ("stability_clarity", "DecompClarityOnly"),
        ("stability_direction", "DecompDirectionOnly"),
        ("stability_dissent", "DecompDissentOnly"),
    ]:
        mean = ig(key, "mean_join")
        r = ((info.get(key) or {}).get("r_vs_theta") or {}).get("r")
        r_attack = ((info.get(key) or {}).get("r_vs_attack") or {}).get("r")
        delta = ig(key, "delta_vs_baseline")
        lines.append(f"\\providecommand{{\\{macro}Mean}}{{{_fmt_num(mean, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}MeanPct}}{{{_fmt_pct(mean, 1)}}}")
        lines.append(f"\\providecommand{{\\{macro}RTheta}}{{{_fmt_r(r, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}RAttack}}{{{_fmt_r(r_attack, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}DeltaPP}}{{{_fmt_pp(delta, 1)}}}")
        lines.append("")

    # ── Regime fall rates (per design) ─────────────────────────────
    lines.append("% Regime fall rates (fraction of periods where join > theta)")
    for key, macro in [
        ("baseline", "InfodesignBaseline"),
        ("stability", "InfodesignStability"),
        ("instability", "InfodesignInstability"),
        ("public_signal", "InfodesignPublicSignal"),
        ("censor_upper", "InfodesignCensorUpper"),
        ("censor_lower", "InfodesignCensorLower"),
        ("censor_upper_known", "InfodesignCensorUpperKnown"),
        ("hard_scramble", "InfodesignHardScramble"),
    ]:
        fall_rate = ig(key, "regime_fall_rate")
        lines.append(f"\\providecommand{{\\{macro}FallRate}}{{{_fmt_pct(fall_rate, 1)}}}")
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
        r_attack = ((info.get(key) or {}).get("r_vs_attack") or {}).get("r")
        lines.append(f"\\providecommand{{\\{macro}RAttack}}{{{_fmt_r(r_attack, 3)}}}")
        lines.append(f"\\providecommand{{\\{macro}DeltaPP}}{{{_fmt_pp(delta, 1)}}}")
    lines.append("")

    # Direction-transform and evidence-domain ablation robustness
    direction_transforms = stats.get("robustness", {}).get("direction_transforms", {}) or {}
    lines.append("% Generator direction-transform robustness")
    for transform, macro in [("tanh", "GenTanh"), ("linear", "GenLinear"), ("step", "GenStep")]:
        entry = direction_transforms.get(transform, {}) or {}
        lines.append(_mc(f"{macro}N", entry.get("n_obs"), 0))
        lines.append(_mc_pct(f"{macro}MeanJoinPct", entry.get("mean_join")))
        lines.append(_mc_r(f"{macro}RAttack", (entry.get("r_vs_attack") or {}).get("r")))
    for design, macro in [("ablate_coordination", "AblateCoord"), ("ablate_state", "AblateState")]:
        entry = info.get(design, {}) or {}
        delta_frac = entry.get("delta_vs_baseline")
        lines.append(_mc_pp(f"{macro}DeltaPP", delta_frac))
        lines.append(_mc_pp_abs_raw(f"{macro}DeltaAbsPP", delta_frac * 100 if delta_frac is not None else None))
        lines.append(_mc_r(f"{macro}RAttack", (entry.get("r_vs_attack") or {}).get("r")))
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
    surv_r_attacks = []
    for slug in PART1_SLUGS:
        display = DISPLAY_NAMES.get(slug, slug)
        mname = _MACRO_NAMES.get(slug, slug.split("--")[-1].title())
        entry = surv_data.get(display, {})
        lines.append(_mc_pp_raw(f"Surv{mname}DeltaPP", entry.get("delta_vs_baseline_pp")))
        lines.append(_mc(f"Surv{mname}MeanJoin", entry.get("mean_join")))
        # Surveillance r_vs_attack correlation
        surv_r = (entry.get("r_vs_attack") or {}).get("r")
        lines.append(_mc_r(f"Surv{mname}RAttack", surv_r))
        if surv_r is not None:
            surv_r_attacks.append(surv_r)
    # Mean surveillance r across models
    mean_surv_r = sum(surv_r_attacks) / len(surv_r_attacks) if surv_r_attacks else None
    lines.append(_mc_r("SurvMeanRAttack", mean_surv_r))
    lines.append("")

    # ── Surveillance variants (placebo, anonymous) ────────────────
    sv = stats.get("surveillance_variants", {})
    lines.append("% Surveillance variants")
    for variant, macro in [("placebo", "PlaceboSurv"), ("anonymous", "AnonSurv")]:
        vd = sv.get(variant, {})
        lines.append(_mc_pp_raw(f"{macro}DeltaPP", vd.get("delta_vs_comm_pp")))
        tt = vd.get("t_test_vs_comm", {})
        lines.append(_mc(f"{macro}PValue", tt.get("p_value") if isinstance(tt, dict) else None, 4))
        lines.append(_mc(f"{macro}MeanJoin", vd.get("mean_join")))
        lines.append(_mc_pct(f"{macro}MeanJoinPct", vd.get("mean_join")))
    sv_by_model = sv.get("by_model", {}) or {}
    llama_variants = sv_by_model.get("Llama 3.3 70B", {}) or {}
    llama_variant_ref = (llama_variants.get("placebo") or llama_variants.get("anonymous") or {})
    lines.append(_mc("LlamaSurvVariantBaselineN", llama_variant_ref.get("baseline_n_obs"), 0))
    lines.append(_mc("LlamaSurvVariantN", llama_variant_ref.get("n_obs"), 0))
    for variant, macro in [("placebo", "LlamaPlaceboSurv"), ("anonymous", "LlamaAnonSurv")]:
        vd = llama_variants.get(variant, {}) or {}
        lines.append(_mc_pp_raw(f"{macro}DeltaPP", vd.get("delta_vs_comm_pp")))
        lines.append(_mc_pp_abs_raw(f"{macro}DeltaAbsPP", vd.get("delta_vs_comm_pp")))
        lines.append(_mc(f"{macro}CIHalfPP", vd.get("ci_half_95_pp"), 1))
        tt = vd.get("t_test_vs_comm", {}) or {}
        lines.append(_mc(f"{macro}PValue", tt.get("p_value") if isinstance(tt, dict) else None, 4))
    lines.append("")

    # ── Clean prompt-isolation reruns ────────────────────────────
    pi = stats.get("prompt_isolation", {})
    pi_mistral = (pi.get("surveillance") or {}).get("Mistral Small Creative", {})
    pi_llama = (pi.get("surveillance") or {}).get("Llama 3.3 70B", {})
    pi_qwen_l = (pi.get("surveillance") or {}).get("Qwen3 235B", {})
    pi_summary = pi.get("_summary", {})

    def _pi_matched(entry):
        return entry.get("matched", {}) if isinstance(entry, dict) else {}

    def _pi_delta(entry):
        matched = _pi_matched(entry)
        return matched.get("delta_pp", entry.get("delta_vs_baseline_pp"))

    def _pi_mean(entry):
        matched = _pi_matched(entry)
        return matched.get("mean_join", entry.get("mean_join"))

    def _pi_baseline_mean(entry):
        matched = _pi_matched(entry)
        return matched.get("baseline_mean_join", entry.get("baseline_mean_join"))

    def _pi_n(entry):
        matched = _pi_matched(entry)
        return matched.get("n_pairs", entry.get("n_obs"))

    def _pi_p(entry):
        matched = _pi_matched(entry)
        if matched.get("p_value") is not None:
            return matched.get("p_value")
        tt = entry.get("t_test_vs_baseline", {}) if isinstance(entry, dict) else {}
        return tt.get("p_value") if isinstance(tt, dict) else None

    lines.append("% Clean prompt-isolation reruns")
    lines.append(_mc("PromptIsoNModels", pi_summary.get("n_models"), 0))
    lines.append(_mc("PromptIsoNegativeModels", pi_summary.get("n_negative"), 0))
    lines.append(_mc("PromptIsoSigFivePctModels", pi_summary.get("n_p_lt_05"), 0))
    lines.append(_mc("PromptIsoSigOnePctModels", pi_summary.get("n_p_lt_01"), 0))
    lines.append(_mc_pp_raw("PromptIsoMeanDeltaPP", pi_summary.get("mean_delta_pp")))
    lines.append(_mc_pp_abs_raw("PromptIsoMeanDeltaAbsPP", pi_summary.get("mean_delta_pp")))
    lines.append(_mc_pp_raw("PromptIsoMinDeltaPP", pi_summary.get("min_delta_pp")))
    lines.append(_mc_pp_raw("PromptIsoMaxDeltaPP", pi_summary.get("max_delta_pp")))
    lines.append(_mc_r("PromptIsoMeanRAttack", pi_summary.get("mean_r_vs_attack")))
    lines.append(_mc("PromptIsoNonQwenLargeN", pi_summary.get("non_qwen235_n"), 0))
    lines.append(_mc_pp_raw("PromptIsoNonQwenLargeMinDeltaPP", pi_summary.get("non_qwen235_min_delta_pp")))
    lines.append(_mc_pp_raw("PromptIsoNonQwenLargeMaxDeltaPP", pi_summary.get("non_qwen235_max_delta_pp")))
    lines.append(_mc("PromptIsoMistralN", _pi_n(pi_mistral), 0))
    lines.append(_mc("PromptIsoMistralMeanJoin", _pi_mean(pi_mistral)))
    lines.append(_mc_pct("PromptIsoMistralMeanJoinPct", _pi_mean(pi_mistral)))
    lines.append(_mc("PromptIsoMistralBaselineMeanJoin", _pi_baseline_mean(pi_mistral)))
    lines.append(_mc_pct("PromptIsoMistralBaselineMeanJoinPct", _pi_baseline_mean(pi_mistral)))
    lines.append(_mc_pp_raw("PromptIsoMistralDeltaPP", _pi_delta(pi_mistral)))
    lines.append(_mc_pp_abs_raw("PromptIsoMistralDeltaAbsPP", _pi_delta(pi_mistral)))
    lines.append(_mc_r("PromptIsoMistralRAttack", (pi_mistral.get("r_vs_attack") or {}).get("r")))
    lines.append(_mc("PromptIsoMistralPValue", _pi_p(pi_mistral), 4))
    lines.append(_mc_p_text("PromptIsoMistralPValueText", _pi_p(pi_mistral)))
    lines.append(_mc_pp_raw("PromptIsoLlamaDeltaPP", _pi_delta(pi_llama)))
    lines.append(_mc_pp_abs_raw("PromptIsoLlamaDeltaAbsPP", _pi_delta(pi_llama)))
    lines.append(_mc("PromptIsoQwenLN", _pi_n(pi_qwen_l), 0))
    lines.append(_mc_pp_raw("PromptIsoQwenLDeltaPP", _pi_delta(pi_qwen_l)))
    lines.append(_mc_pp_abs_raw("PromptIsoQwenLDeltaAbsPP", _pi_delta(pi_qwen_l)))
    lines.append(_mc("PromptIsoQwenLPValue", _pi_p(pi_qwen_l), 3))
    # Pooled matched-cell estimand across models
    lines.append(_mc_pp_raw("PromptIsoPooledMatchedDeltaPP", pi_summary.get("pooled_matched_delta_pp")))
    lines.append(_mc("PromptIsoPooledMatchedN", pi_summary.get("pooled_matched_n"), 0))
    lines.append(_mc_p_text("PromptIsoPooledMatchedPText", pi_summary.get("pooled_matched_p_value")))
    # Country-clustered version of the primary-model paired contrast
    pi_mistral_cc = (_pi_matched(pi_mistral) or {}).get("country_clustered", {}) or {}
    lines.append(_mc("PromptIsoMistralCountryClustT", pi_mistral_cc.get("t_stat"), 2))
    lines.append(_mc_p_text("PromptIsoMistralCountryClustPText", pi_mistral_cc.get("p_value")))
    lines.append(_mc("PromptIsoMistralCountryClustN", pi_mistral_cc.get("n_countries"), 0))
    lines.append("")

    # ── Message controls ─────────────────────────────────────────
    message_controls = stats.get("message_controls", {}) or {}
    degraded = message_controls.get("degraded_messages", {}) or {}
    no_msg = message_controls.get("no_messages", {})
    lines.append("% Message controls")
    lines.append(_mc("DegradedN", degraded.get("n_obs"), 0))
    lines.append(_mc("DegradedBaselineN", degraded.get("baseline_n_obs"), 0))
    lines.append(_mc("DegradedMeanJoin", degraded.get("mean_join")))
    lines.append(_mc_pct("DegradedMeanJoinPct", degraded.get("mean_join")))
    lines.append(_mc("DegradedBaselineMeanJoin", degraded.get("baseline_mean_join")))
    lines.append(_mc_pct("DegradedBaselineMeanJoinPct", degraded.get("baseline_mean_join")))
    lines.append(_mc_pp_raw("DegradedDeltaPP", degraded.get("delta_vs_baseline_pp")))
    lines.append(_mc_pp_abs_raw("DegradedDeltaAbsPP", degraded.get("delta_vs_baseline_pp")))
    degraded_t = degraded.get("t_test_vs_baseline", {}) or {}
    lines.append(_mc("DegradedPValue", degraded_t.get("p_value") if isinstance(degraded_t, dict) else None, 4))
    lines.append(_mc_p_text("DegradedPValueText", degraded_t.get("p_value") if isinstance(degraded_t, dict) else None))
    lines.append(_mc_pct("DegradedPureMeanJoinPct", degraded.get("pure_mean_join")))
    lines.append(_mc("NoMsgN", no_msg.get("n_obs"), 0))
    lines.append(_mc("NoMsgMeanJoin", no_msg.get("mean_join")))
    lines.append(_mc_pct("NoMsgMeanJoinPct", no_msg.get("mean_join")))
    lines.append(_mc("NoMsgBaselineMeanJoin", no_msg.get("baseline_mean_join")))
    lines.append(_mc_pct("NoMsgBaselineMeanJoinPct", no_msg.get("baseline_mean_join")))
    lines.append(_mc_pp_raw("NoMsgDeltaPP", no_msg.get("delta_vs_baseline_pp")))
    no_msg_t = no_msg.get("t_test_vs_baseline", {})
    lines.append(_mc("NoMsgPValue", no_msg_t.get("p_value") if isinstance(no_msg_t, dict) else None, 4))
    # Surveillance vs no-message comparison (primary model)
    lines.append(_mc("PromptIsoSurvMeanJoin", no_msg.get("surv_mean_join")))
    lines.append(_mc_pct("PromptIsoSurvMeanJoinPct", no_msg.get("surv_mean_join")))
    lines.append(_mc_pp_raw("PromptIsoVsNoMsgDeltaPP", no_msg.get("delta_surv_vs_nomsg_pp")))
    lines.append(_mc_pp_abs_raw("PromptIsoVsNoMsgDeltaAbsPP", no_msg.get("delta_surv_vs_nomsg_pp")))
    no_msg_t_sv = no_msg.get("t_test_surv_vs_nomsg", {})
    lines.append(_mc("PromptIsoVsNoMsgPValue", no_msg_t_sv.get("p_value") if isinstance(no_msg_t_sv, dict) else None, 4))
    no_msg_by_model = message_controls.get("no_messages_by_model", {}) or {}
    for model_name, prefix in [
        ("Llama 3.3 70B", "LlamaNoMsg"),
        ("Qwen3 30B", "QwenNoMsg"),
    ]:
        model_msg = no_msg_by_model.get(model_name, {}) or {}
        lines.append(_mc(prefix + "N", model_msg.get("n_obs"), 0))
        lines.append(_mc_pct(prefix + "MeanJoinPct", model_msg.get("mean_join")))
        lines.append(_mc_pp_raw(prefix + "DeltaPP", model_msg.get("delta_vs_baseline_pp")))
        lines.append(_mc_pp_raw(prefix + "SurvVsNoMsgDeltaPP", model_msg.get("delta_surv_vs_nomsg_pp")))
        lines.append(_mc_pp_abs_raw(prefix + "SurvVsNoMsgDeltaAbsPP", model_msg.get("delta_surv_vs_nomsg_pp")))
        model_msg_t = model_msg.get("t_test_surv_vs_nomsg", {}) or {}
        lines.append(_mc(prefix + "SurvVsNoMsgPValue", model_msg_t.get("p_value") if isinstance(model_msg_t, dict) else None, 4))
    no_msg_deltas = [no_msg.get("delta_vs_baseline_pp")]
    no_msg_deltas.extend(
        model_msg.get("delta_vs_baseline_pp")
        for model_msg in no_msg_by_model.values()
        if isinstance(model_msg, dict)
    )
    no_msg_abs = [abs(float(x)) for x in no_msg_deltas if x is not None]
    if no_msg_abs:
        lines.append(_mc_raw("NoMsgDeltaAbsMinPP", f"{min(no_msg_abs):.0f}"))
        lines.append(_mc_raw("NoMsgDeltaAbsMaxPP", f"{max(no_msg_abs):.0f}"))
    lines.append("")

    # ── Message content analysis (classifier + themes) ────────────
    msg_content = stats.get("message_content", {}) or {}
    lines.append("% Message content classifier and themes (primary model)")
    msg_clf = msg_content.get("classifier", {}) or {}
    lines.append(_mc("MsgClassifierAcc", msg_clf.get("accuracy_pct"), 1))
    lines.append(_mc_raw("MsgClassifierNTrain", _fmt_int_sep(msg_clf.get("n_train"))))
    lines.append(_mc_raw("MsgClassifierNTest", _fmt_int_sep(msg_clf.get("n_test"))))
    msg_by_model = msg_content.get("classifier_by_model", {}) or {}
    llama_msg = ((msg_by_model.get("Llama 3.3 70B") or {}).get("balanced") or {})
    qwen_msg = ((msg_by_model.get("Qwen3 30B") or {}).get("balanced") or {})
    lines.append(_mc("MsgClassifierLlamaAcc", llama_msg.get("accuracy_pct"), 1))
    lines.append(_mc_raw("MsgClassifierLlamaNTest", _fmt_int_sep(llama_msg.get("n_test"))))
    lines.append(_mc("MsgClassifierQwenSAcc", qwen_msg.get("accuracy_pct"), 1))
    lines.append(_mc_raw("MsgClassifierQwenSNTest", _fmt_int_sep(qwen_msg.get("n_test"))))
    direct_coded = msg_content.get("direct_to_coded_summary", {}) or {}
    lines.append(_mc("MsgDirectCodedNModels", direct_coded.get("n_models"), 0))
    lines.append(_mc("MsgDirectCodedRepModels", direct_coded.get("n_direct_down_coded_up"), 0))
    msg_themes = msg_content.get("themes", {}) or {}
    for theme_key, macro_prefix in [
        ("weakness", "MsgWeakness"),
        ("strength", "MsgStrength"),
        ("direct_regime", "MsgDirectRegime"),
        ("coded_metaphor", "MsgCodedMetaphor"),
    ]:
        t = msg_themes.get(theme_key, {}) or {}
        lines.append(_mc(macro_prefix + "Base", t.get("base_pct"), 1))
        lines.append(_mc(macro_prefix + "Surv", t.get("surv_pct"), 1))
        # Delta as the difference of the displayed (rounded) endpoints, so the
        # printed base/surv/delta triple is internally consistent.
        lines.append(_mc_pp_raw(
            macro_prefix + "DeltaPP",
            _delta_of_displayed(t.get("base_pct"), t.get("surv_pct"), 1),
        ))
        lines.append(_mc(macro_prefix + "Z", t.get("z"), 2))
        lines.append(_mc(macro_prefix + "P", t.get("p"), 4))
    lines.append("")

    # ── Cross-task discriminant placebo (Llama 3.3 70B) ───────────
    ctp = stats.get("cross_task_placebo", {}) or {}
    lines.append("% Cross-task discriminant placebo (Llama 3.3 70B)")
    lines.append(_mc("CrossTaskN", ctp.get("n_surv"), 0))
    lines.append(_mc("CrossTaskBaselineMeanJoin", ctp.get("base_mean_join")))
    lines.append(_mc_pct("CrossTaskBaselineMeanJoinPct", ctp.get("base_mean_join")))
    lines.append(_mc("CrossTaskSurvMeanJoin", ctp.get("surv_mean_join")))
    lines.append(_mc_pct("CrossTaskSurvMeanJoinPct", ctp.get("surv_mean_join")))
    lines.append(_mc_pp_raw("CrossTaskDeltaPP", ctp.get("delta_pp")))
    lines.append(_mc_pp_abs_raw("CrossTaskDeltaAbsPP", ctp.get("delta_pp")))
    llama_delta = _pi_delta(pi_llama)
    coord_task_delta_abs = abs(llama_delta) if llama_delta is not None else None
    cross_task_delta_abs = abs(ctp.get("delta_pp")) if ctp.get("delta_pp") is not None else None
    coord_specific_pp = None
    if coord_task_delta_abs is not None and cross_task_delta_abs is not None:
        coord_specific_pp = coord_task_delta_abs - cross_task_delta_abs
    lines.append(_mc("CrossTaskCoordTaskDeltaAbsPP", coord_task_delta_abs, 1))
    lines.append(_mc("CrossTaskCoordSpecificPP", coord_specific_pp, 1))
    ctp_t = ctp.get("t_test", {}) or {}
    lines.append(_mc("CrossTaskPValue", ctp_t.get("p_value") if isinstance(ctp_t, dict) else None, 4))
    lines.append("")

    # ── Cross-model writer-reader rotation ──────────────────────────
    xrot = stats.get("cross_model_message_rotation", {}) or {}
    lines.append("% Cross-model writer-reader rotation")
    for key, prefix in [
        ("llama_writes_qwen_reads", "XModelLlamaQwen"),
        ("qwen_writes_llama_reads", "XModelQwenLlama"),
        ("within_llama_same_cells", "XModelWithinLlama"),
    ]:
        rot = xrot.get(key, {}) or {}
        lines.append(_mc(prefix + "NPairs", rot.get("n_pairs"), 0))
        lines.append(_mc_pp_raw(prefix + "DeltaPP", rot.get("paired_delta_pp")))
        lines.append(_mc_pp_abs_raw(prefix + "DeltaAbsPP", rot.get("paired_delta_pp")))
        lines.append(_mc(prefix + "TStat", rot.get("paired_t"), 2))
        lines.append(_mc(prefix + "PValue", rot.get("paired_p"), 4))
        lines.append(_mc_p_text(prefix + "PValueText", rot.get("paired_p")))
    lines.append("")

    # ── Fixed messages test ───────────────────────────────────────
    fm = stats.get("fixed_messages_test", {})
    lines.append("% Fixed messages surveillance test")
    lines.append(_mc_pp_raw("FixedMsgDeltaPP", fm.get("delta_pp")))
    lines.append(_mc("FixedMsgTStat", fm.get("ttest_t"), 2))
    lines.append(_mc("FixedMsgPValue", fm.get("ttest_p"), 4))
    lines.append(_mc_p_text("FixedMsgPValueText", fm.get("ttest_p")))
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
        r_part = bd.get("r_partial", {})
        lines.append(_mc_r(f"{macro}RPartial", r_part.get("r") if isinstance(r_part, dict) else r_part))
        lines.append(_mc(f"{macro}MeanBelief", bd.get("mean_belief")))
        lines.append(_mc(f"{macro}MeanJoin", bd.get("mean_join")))
    belief_comm = beliefs.get("comm", {}) or {}
    belief_surv = beliefs.get("surveillance", {}) or {}
    if belief_comm and belief_surv:
        lines.append(_mc_pp(
            "BeliefCommSurvActionDeltaPP",
            (belief_surv.get("mean_join") or 0) - (belief_comm.get("mean_join") or 0),
        ))
        lines.append(_mc_pp_abs_raw(
            "BeliefCommSurvActionDeltaAbsPP",
            100 * ((belief_surv.get("mean_join") or 0) - (belief_comm.get("mean_join") or 0)),
        ))
        lines.append(_mc_pp(
            "BeliefCommSurvFirstOrderDeltaPP",
            (belief_surv.get("mean_belief") or 0) - (belief_comm.get("mean_belief") or 0),
        ))
    # Second-order beliefs
    sob = beliefs.get("_surv_vs_comm_sob", {})
    lines.append("% Second-order beliefs (surveillance vs comm)")
    lines.append(_mc("SOBCommMean", sob.get("comm_mean")))
    lines.append(_mc("SOBSurvMean", sob.get("surv_mean")))
    lines.append(_mc_pp_raw("SOBDeltaPP", sob.get("delta_pp")))
    lines.append(_mc("SOBPValue", sob.get("p_value"), 4))
    lines.append(_mc("SOBTStat", sob.get("t_stat"), 2))
    # 95% CI for the SOB contrast (messages-excluded, comm vs surveillance),
    # computed from the agent-level second-order summary statistics.
    sob_comm = (beliefs.get("comm", {}) or {}).get("second_order", {}) or {}
    sob_surv = (beliefs.get("surveillance", {}) or {}).get("second_order", {}) or {}
    sob_ci_lo = sob_ci_hi = None
    if all(sob_comm.get(k) is not None for k in ("mean", "std", "n")) and all(
        sob_surv.get(k) is not None for k in ("mean", "std", "n")
    ):
        import math as _math
        sob_delta = (float(sob_surv["mean"]) - float(sob_comm["mean"])) * 100
        sob_se = 100 * _math.sqrt(
            float(sob_comm["std"]) ** 2 / float(sob_comm["n"])
            + float(sob_surv["std"]) ** 2 / float(sob_surv["n"])
        )
        sob_ci_lo = sob_delta - 1.96 * sob_se
        sob_ci_hi = sob_delta + 1.96 * sob_se
    lines.append(_mc_pp_raw("SOBDeltaCILoPP", sob_ci_lo))
    lines.append(_mc_pp_raw("SOBDeltaCIHiPP", sob_ci_hi))
    # Preference falsification
    pf = beliefs.get("_pref_falsification", {})
    lines.append("% Preference falsification")
    lines.append(_mc("PrefFalsPureMeanBelief", pf.get("pure_mean_belief")))
    lines.append(_mc("PrefFalsSurvMeanBelief", pf.get("surv_mean_belief")))
    lines.append(_mc_pp_raw("PrefFalsBeliefDeltaPP", pf.get("belief_delta_pp")))
    lines.append(_mc_pp_raw("PrefFalsActionDeltaPP", pf.get("action_delta_pp")))
    lines.append("")

    # ── Classifier baselines ──────────────────────────────────────
    cb = stats.get("classifier_baselines", {})
    lines.append("% Classifier baselines")
    bow = cb.get("bow_tfidf", {})
    bow_cv = bow.get("cv_pure", {})
    bow_surv = bow.get("cross_pure_to_surv", {})
    lines.append(_mc("ClassBowAccuracy", bow_cv.get("accuracy_mean")))
    lines.append(_mc("ClassBowAUC", bow_cv.get("auc_mean")))
    lines.append(_mc_pct("ClassBowAccuracyPct", bow_cv.get("accuracy_mean")))
    lines.append(_mc("ClassBowSurvPredicted", bow_surv.get("predicted_join_rate")))
    lines.append(_mc("ClassBowSurvActual", bow_surv.get("actual_join_rate")))
    lines.append(_mc_pct("ClassBowSurvPredictedPct", bow_surv.get("predicted_join_rate")))
    lines.append(_mc_pct("ClassBowSurvActualPct", bow_surv.get("actual_join_rate")))
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
    lines.append(_mc_pct("ClassSliderAccuracyPct", slider_cv.get("accuracy_mean")))
    lines.append(_mc("ClassSliderSurvPredicted", slider_surv.get("predicted_join_rate")))
    lines.append(_mc("ClassSliderSurvActual", slider_surv.get("actual_join_rate")))
    lines.append(_mc_pct("ClassSliderSurvPredictedPct", slider_surv.get("predicted_join_rate")))
    # BC comparative statics
    bc_cs = cb.get("bc_comparative_statics", {})
    for cond, macro in [("baseline", "ClassBCBaseline"), ("bc_high_cost", "ClassBCHighCost"), ("bc_low_cost", "ClassBCLowCost")]:
        cd = bc_cs.get(cond, {})
        lines.append(_mc(f"{macro}Predicted", cd.get("classifier_predicted_join")))
        lines.append(_mc(f"{macro}Actual", cd.get("actual_join")))
        lines.append(_mc_pp_raw(f"{macro}GapPP", cd.get("gap_pp")))
    lines.append("")

    # ── Regression macros (from regression_results.json) ──────────
    reg_path = Path(__file__).resolve().parent / "regression_results.json"
    if reg_path.exists():
        with open(reg_path) as f:
            import json as _json
            reg = _json.load(f)
        lines.append("% Agent-level regression results")
        # Belief-action equation pseudo R²
        action_eq = reg.get("belief_regressions", {}).get("action_equation", {})
        lines.append(_mc("ActionEqPseudoRSq", action_eq.get("pseudo_r2"), 3))
        lines.append(_mc_raw("ActionEqNObs", f"{action_eq.get('n_obs', '---'):,}" if action_eq.get('n_obs') else "---"))
        # Main logit
        main_logit = reg.get("agent_logit", {}).get("main_logit", {})
        lines.append(_mc("MainLogitPseudoRSq", main_logit.get("pseudo_r2"), 4))
        lines.append(_mc_raw("MainLogitNObs", f"{main_logit.get('n_obs', '---'):,}" if main_logit.get('n_obs') else "---"))
        # Marginal effects at the mean
        mem = main_logit.get("marginal_effects_at_mean", {})
        if mem:
            lines.append(_mc_raw("MEMThetaPP", f"{abs(mem.get('theta', 0)) * 100:.0f}"))
            lines.append(_mc_raw("MEMSurvPP", f"{abs(mem.get('treat_surveillance', 0)) * 100:.0f}"))
        # Discrete (0->1) average marginal effect of surveillance
        ame_d = main_logit.get("ame_discrete", {}) or {}
        if ame_d.get("treat_surveillance") is not None:
            lines.append(_mc_raw(
                "MEMSurvDiscretePP",
                f"{abs(ame_d['treat_surveillance']) * 100:.1f}",
            ))
        lines.append("")

    # ── Message informativeness ─────────────────────────────────────
    mi_path = Path(__file__).resolve().parent / "_archive" / "message_informativeness_results.json"
    if mi_path.exists():
        import json as _json2
        with open(mi_path) as f:
            mi = _json2.load(f)
        lines.append("% Message informativeness")
        comm_r2 = mi.get("comm", {}).get("R2_text_to_theta")
        surv_r2 = mi.get("surveillance", {}).get("R2_text_to_theta")
        lines.append(_mc("MsgRSqComm", comm_r2, 2))
        lines.append(_mc("MsgRSqSurv", surv_r2, 2))
        if comm_r2 and surv_r2 and comm_r2 > 0:
            pct_drop = round((1.0 - surv_r2 / comm_r2) * 100)
            lines.append(_mc_raw("MsgRSqDropPct", f"{pct_drop}\\%"))
        lines.append("")

    # ── Level-k benchmark ─────────────────────────────────────────
    level_k = stats.get("level_k", {})
    if level_k:
        lines.append("% Level-k benchmark (BNE vs L1 vs L2)")
        for model_key, macro_prefix in [("bne", "LevelKBNE"), ("l1", "LevelKLOne"), ("l2", "LevelKLTwo")]:
            mk = level_k.get(model_key, {})
            lines.append(_mc(f"{macro_prefix}RMSE", mk.get("rmse"), 3))
            lines.append(_mc_r(f"{macro_prefix}R", mk.get("r")))
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

    # ── B/C sweep cutoff tracking ────────────────────────────────
    bc_track = info.get("_bc_sweep_cutoff_tracking", {})
    lines.append("% B/C sweep cutoff tracking")
    lines.append(_mc("BCSweepCutoffTrackingR", bc_track.get("r"), 3))
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
        r_attack = (td.get("r_vs_attack") or {}).get("r") if isinstance(td.get("r_vs_attack"), dict) else None
        lines.append(_mc_r(f"{macro}RAttack", r_attack))
        lines.append(_mc(f"{macro}MeanJoin", td.get("mean_join")))
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
            r_theta = (vd.get("r_vs_theta") or {}).get("r") if isinstance(vd.get("r_vs_theta"), dict) else vd.get("r_vs_theta")
            r_attack = (vd.get("r_vs_attack") or {}).get("r") if isinstance(vd.get("r_vs_attack"), dict) else None
            lines.append(_mc_r(f"CrossGen{mname}{variant.title()}R", r_theta))
            lines.append(_mc_r(f"CrossGen{mname}{variant.title()}RAttack", r_attack))
    lines.append("")

    # ── Per-model hard scramble ──────────────────────────────────
    cross_model = info.get("_cross_model", {})
    lines.append("% Per-model hard scramble")
    for model_name, designs in cross_model.items():
        if not isinstance(designs, dict):
            continue
        hs = designs.get("hard_scramble", {})
        if not hs:
            continue
        mname = "".join(c for c in model_name if c.isalpha())[:12]
        r_a = (hs.get("r_vs_attack") or {}).get("r") if isinstance(hs.get("r_vs_attack"), dict) else None
        r_t = (hs.get("r_vs_theta") or {}).get("r") if isinstance(hs.get("r_vs_theta"), dict) else None
        p_val = (hs.get("r_vs_attack") or {}).get("p") if isinstance(hs.get("r_vs_attack"), dict) else None
        n_obs = hs.get("n_obs")
        lines.append(_mc_r(f"HardScramble{mname}RAttack", r_a))
        lines.append(_mc_r(f"HardScramble{mname}RTheta", r_t))
        lines.append(_mc(f"HardScramble{mname}PValue", p_val, 2))
        lines.append(_mc(f"HardScramble{mname}NObs", n_obs, 0))
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

    # ── Pooled scramble/flip aggregates ──────────────────────────
    pooled_scr = part1.get("_pooled_scramble", {})
    lines.append("% Pooled scramble/flip aggregates")
    lines.append(_mc_r("PooledScrambleRAttack", (pooled_scr.get("r_vs_attack") or {}).get("r")))
    lines.append(_mc_r("PooledScrambleRTheta", (pooled_scr.get("r_vs_theta") or {}).get("r")))
    lines.append(_mc("PooledScrambleMeanJoin", pooled_scr.get("mean_join")))
    lines.append(_mc_r("PooledFlipRTheta", (pooled_flip.get("r_vs_theta") or {}).get("r")))
    lines.append(_mc("PooledFlipMeanJoin", pooled_flip.get("mean_join")))
    lines.append("")

    # ── Mean-of-models scramble r ─────────────────────────────────
    scr_rs = [
        (v.get("scramble", {}).get("r_vs_attack") or {}).get("r")
        for k, v in part1.items()
        if isinstance(k, str) and not k.startswith("_") and isinstance(v, dict) and "scramble" in v
    ]
    scr_rs = [x for x in scr_rs if x is not None]
    mean_scr_r = sum(scr_rs) / len(scr_rs) if scr_rs else None
    lines.append("% Mean-of-models scramble r")
    lines.append(_mc_r("MeanModelRScrambleAttack", mean_scr_r))
    lines.append("")

    # ── Mean join across all models (Part 1 pure) ────────────────
    lines.append("% Mean join rate across all models (pure)")
    lines.append(_mc("MeanModelPureMeanJoin", pooled_pure.get("mean_join")))
    lines.append("")

    # ── Per-model pure mean join range ────────────────────────────
    pure_means_sorted = sorted(pure_means) if pure_means else []
    lines.append("% Per-model pure mean join range")
    if pure_means_sorted:
        lines.append(_mc("MinModelPureMeanJoin", pure_means_sorted[0]))
        lines.append(_mc("MaxModelPureMeanJoin", pure_means_sorted[-1]))
    lines.append("")

    # ── Per-model pure r_attack range ─────────────────────────────
    per_model_r_attacks = []
    for k, v in part1.items():
        if isinstance(k, str) and not k.startswith("_") and isinstance(v, dict):
            r_a = (v.get("pure", {}).get("r_vs_attack") or {}).get("r")
            if r_a is not None:
                per_model_r_attacks.append(r_a)
    per_model_r_attacks_sorted = sorted(per_model_r_attacks)
    lines.append("% Per-model pure r_attack range")
    if per_model_r_attacks_sorted:
        lines.append(_mc_r("MinModelPureRAttack", per_model_r_attacks_sorted[0]))
        lines.append(_mc_r("MaxModelPureRAttack", per_model_r_attacks_sorted[-1]))
    lines.append("")

    # ── B/C sweep percentage forms ────────────────────────────────
    lines.append("% B/C sweep percentage forms")
    lines.append(_mc_pct("BCSweepBaselineMeanPct", baseline_info.get("mean_join")))
    lines.append(_mc_pct("BCSweepHighCostMeanPct", bc_high.get("mean_join")))
    lines.append(_mc_pct("BCSweepLowCostMeanPct", bc_low.get("mean_join")))
    stakes = stats.get("infodesign", {}) or {}
    stakes_map = {
        "StakesVeryHighCost": "bc_very_high_cost",
        "StakesModerateHigh": "bc_moderate_high_cost",
        "StakesNeutral": "bc_neutral",
        "StakesModerateLow": "bc_moderate_low_cost",
        "StakesVeryLowCost": "bc_very_low_cost",
        "StakesPlacebo": "bc_placebo",
        "StakesHighCostVTwo": "bc_high_cost_v2",
        "StakesLowCostVTwo": "bc_low_cost_v2",
    }
    for macro_prefix, stat_key in stakes_map.items():
        lines.append(_mc_pct(macro_prefix + "MeanPct", (stakes.get(stat_key) or {}).get("mean_join")))
    very_high = (stakes.get("bc_very_high_cost") or {}).get("mean_join")
    very_low = (stakes.get("bc_very_low_cost") or {}).get("mean_join")
    if very_high is not None and very_low is not None:
        lines.append(_mc_raw("StakesRangePP", f"{(float(very_low) - float(very_high)) * 100:.1f}"))
    lines.append("")

    # ── B/C sweep from classifier_baselines (actual_join for prose) ─
    bc_cs = cb.get("bc_comparative_statics", {})
    lines.append("% B/C comparative statics (actual join rates)")
    for cond, macro in [("baseline", "BCStatBaseline"), ("bc_high_cost", "BCStatHighCost"), ("bc_low_cost", "BCStatLowCost")]:
        cd = bc_cs.get(cond, {})
        lines.append(_mc_pct(f"{macro}MeanPct", cd.get("actual_join")))
    lines.append("")

    # ── Beliefs second-order details ──────────────────────────────
    lines.append("% Beliefs second-order details")
    for treatment, macro in [("pure", "SOBPure"), ("comm", "SOBComm"), ("surveillance", "SOBSurv")]:
        sob_data = beliefs.get(treatment, {}).get("second_order", {})
        # SOBCommMean and SOBSurvMean already emitted above from _surv_vs_comm_sob;
        # only emit SOBPureMean here to avoid duplicate \providecommand.
        if macro == "SOBPure":
            lines.append(_mc("{}Mean".format(macro), sob_data.get("mean")))
        lines.append(_mc_pct("{}MeanPct".format(macro), sob_data.get("mean")))
        r_theta = (sob_data.get("r_vs_theta") or {}).get("r") if isinstance(sob_data.get("r_vs_theta"), dict) else None
        lines.append(_mc_r("{}RTheta".format(macro), r_theta))
    lines.append("")

    # ── Beliefs additional detail macros ──────────────────────────
    lines.append("% Beliefs additional details")
    for treatment, macro in [("pure", "BeliefPure"), ("comm", "BeliefComm"), ("surveillance", "BeliefSurv")]:
        bd = beliefs.get(treatment, {})
        lines.append(_mc_pct(f"{macro}MeanBeliefPct", bd.get("mean_belief")))
        lines.append(_mc_pct(f"{macro}MeanJoinPct", bd.get("mean_join")))
    lines.append("")

    # ── Surveillance cross-model average delta ────────────────────
    surv_deltas = [
        entry.get("delta_vs_baseline_pp")
        for entry in surv_data.values()
        if isinstance(entry, dict) and entry.get("delta_vs_baseline_pp") is not None
    ]
    mean_surv_delta = sum(surv_deltas) / len(surv_deltas) if surv_deltas else None
    lines.append("% Surveillance cross-model average")
    lines.append(_mc_pp_raw("SurvMeanDeltaPP", mean_surv_delta))
    # Also per-model baseline mean join
    for slug in PART1_SLUGS:
        display = DISPLAY_NAMES.get(slug, slug)
        mname = _MACRO_NAMES.get(slug, slug.split("--")[-1].title())
        entry = surv_data.get(display, {})
        if entry:
            lines.append(_mc_pct(f"Surv{mname}BaselineMeanPct", entry.get("baseline_mean_join")))
            lines.append(_mc_pct(f"Surv{mname}MeanJoinPct", entry.get("mean_join")))
    lines.append("")

    # ── Preference falsification additional ──────────────────────
    lines.append("% Preference falsification additional")
    lines.append(_mc("PrefFalsBeliefPValue", pf.get("belief_p_value"), 2))
    lines.append(_mc("PrefFalsActionPValue", pf.get("action_p_value"), 4))
    lines.append("")

    # ── Fixed messages percentage forms ───────────────────────────
    lines.append("% Fixed messages percentage forms")
    lines.append(_mc_pct("FixedMsgBaselineMeanPct", fm.get("baseline_mean_join")))
    lines.append(_mc_pct("FixedMsgSurvMeanPct", fm.get("surv_mean_join")))
    lines.append("")

    # ── CK cell means percentage forms ────────────────────────────
    ck_cells = ck.get("cell_means", {})
    lines.append("% CK cell means pct")
    for cell, macro in [
        ("ck_high_coord", "CKHighCoordCellPct"),
        ("ck_low_coord", "CKLowCoordCellPct"),
        ("priv_high_coord", "PrivHighCoordCellPct"),
        ("priv_low_coord", "PrivLowCoordCellPct"),
    ]:
        lines.append(_mc_pct(macro, ck_cells.get(cell)))
    lines.append("")

    # ── Infodesign flip mean join pct (for prose) ─────────────────
    lines.append("% Infodesign flip pct for prose")
    flip_mean = ig("flip", "mean_join")
    lines.append(_mc_pct("InfodesignFlipMeanJoinPct", flip_mean))
    lines.append("")

    # ── Parse errors ──────────────────────────────────────────────
    pe = stats.get("parse_errors", {})
    lines.append("% Parse error rates")
    parse_treatments = ["pure", "comm", "scramble", "flip"]
    parse_models_below_two = 0
    for slug in PART1_SLUGS:
        display = DISPLAY_NAMES.get(slug, slug)
        mname = _MACRO_NAMES.get(slug, slug.split("--")[-1].title())
        pd_entry = pe.get(display, {})
        if not pd_entry:
            continue
        treatment_rows = [
            pd_entry.get(t)
            for t in parse_treatments
            if pd_entry.get(t) is not None
        ]
        if treatment_rows and all(
            (
                (row.get("mean_api_error_rate", 0.0) or 0.0)
                + (row.get("mean_unparseable_rate", 0.0) or 0.0)
            ) < 0.02
            for row in treatment_rows
        ):
            parse_models_below_two += 1
        pure_row = pd_entry.get("pure", {}) or {}
        lines.append(_mc(f"ParseErr{mname}", pure_row.get("mean_unparseable_rate"), 3))
    lines.append(_mc("ParseErrModelsBelowTwo", parse_models_below_two, 0))
    lines.append(_mc("ParseErrNModels", len(PART1_SLUGS), 0))
    lines.append("")

    # ── Misc paper stats (cutoff range, temperature, robustness, etc.) ──
    misc = stats.get("misc", {})
    if misc:
        lines.append("% Misc inline paper stats")
        # Cutoff range
        lines.append(_mc_r("CutoffMin", misc.get("cutoff_min")))
        lines.append(_mc_r("CutoffMax", misc.get("cutoff_max")))
        # Temperature robustness (primary model)
        lines.append(_mc_r("TempRMinPrimary", misc.get("temp_r_min_primary")))
        lines.append(_mc_r("TempRMaxPrimary", misc.get("temp_r_max_primary")))
        # Temperature robustness (all combos)
        lines.append(_mc_r("TempRMinAll", misc.get("temp_r_min_all")))
        lines.append(_mc_r("TempRMaxAll", misc.get("temp_r_max_all")))
        lines.append(_mc_raw("TempNCombos", str(misc.get("temp_n_combos", ""))))
        # Agent count robustness
        lines.append(_mc_r("AgentCountRMin", misc.get("agent_count_r_min")))
        lines.append(_mc_r("AgentCountRMax", misc.get("agent_count_r_max")))
        # Network density
        lines.append(_mc_r("NetworkKEightR", misc.get("network_k8_r")))
        lines.append(_mc_r("NetworkKFourR", misc.get("network_k4_r")))
        # Flip r
        lines.append(_mc_r("FlipRMax", misc.get("flip_r_max")))
        # Cross-generator
        lines.append(_mc("CrossgenMaxDiff", misc.get("crossgen_max_diff"), 2))
        # Infodesign comm join rates
        lines.append(_mc("IDCommBaselinePct", misc.get("idcomm_baseline_pct"), 1))
        lines.append(_mc("IDCommCensorLowerPct", misc.get("idcomm_censor_lower_pct"), 1))
        lines.append(_mc("IDCommCensorUpperPct", misc.get("idcomm_censor_upper_pct"), 1))
        # Punishment risk
        lines.append(_mc("PunishRiskMean", misc.get("punishment_risk_mean"), 1))
        lines.append(_mc("PunishRiskMaxDiff", misc.get("punishment_risk_max_diff"), 1))
        # Agent-level regression N
        n_agents = misc.get("agent_level_n")
        if n_agents:
            n_str = f"{n_agents:,}".replace(",", "{,}")
            lines.append(_mc_raw("AgentLevelN", n_str))
        # Finite-N benchmark
        lines.append(_mc("FiniteNMinR", misc.get("finite_n_min_r"), 2))
        lines.append(_mc("FiniteNPrimaryR", misc.get("finite_n_primary_r"), 4))
        lines.append(_mc("FiniteNPooledR", misc.get("finite_n_pooled_r"), 4))
        # Regime survival
        lines.append(_mc_raw("BaselineRegimeSurvPct",
                             f"{misc.get('baseline_regime_survival_pct', 50)}\\%"))
        # Text baseline r
        lines.append(_mc("TextBaselineR", misc.get("text_baseline_r"), 2))
        # Group-size awareness
        lines.append(_mc("GSPureJoin", misc.get("gs_pure_join"), 3))
        lines.append(_mc("GSBaselinePureJoin", misc.get("gs_baseline_pure_join"), 3))
        lines.append(_mc_raw("GSCommPremiumPP",
                             f"{misc.get('gs_comm_premium_pp', 0):+.1f}"))
        # Word frequencies — surveillance
        lines.append("% Word frequency stats (surveillance)")
        lines.append(_mc("WFActComm", misc.get("wf_act_comm"), 1))
        lines.append(_mc("WFActSurv", misc.get("wf_act_surv"), 1))
        lines.append(_mc("WFCollapseComm", misc.get("wf_collapse_comm"), 1))
        lines.append(_mc("WFCollapseSurv", misc.get("wf_collapse_surv"), 1))
        lines.append(_mc("WFActionJoinComm", misc.get("wf_action_join_comm"), 1))
        lines.append(_mc("WFActionJoinSurv", misc.get("wf_action_join_surv"), 1))
        # Deduplication robustness
        lines.append("% Deduplication robustness (footnote)")
        lines.append(_mc("DedupRPre", misc.get("dedup_r_pre"), 3))
        lines.append(_mc("DedupRPost", misc.get("dedup_r_post"), 3))
        dn = misc.get("dedup_n_unique")
        if dn is not None:
            lines.append(_mc_raw("DedupNUnique", f"{dn:,}".replace(",", "{,}")))
        # Infodesign scramble p-value
        lines.append(_mc("InfodesignScramblePValue", misc.get("infodesign_scramble_p"), 2))
        # Llama infodesign scramble r
        lines.append(_mc_r("LlamaInfodesignScrambleR", misc.get("llama_infodesign_scramble_r")))
        # Trinity parse error rate
        tp = misc.get("trinity_api_error_pct")
        if tp is not None:
            lines.append(_mc_raw("TrinityAPIErrorPct", str(tp)))
        lines.append("")

    return "\n".join(lines) + "\n"


def render_tab_beliefs(stats: dict) -> str:
    """Render beliefs table with partial correlations."""
    beliefs = stats.get("beliefs_v2", {})
    if not beliefs:
        return "% No beliefs_v2 data available.\n"

    def _r_ci_cell(d: dict | None) -> str:
        d = d or {}
        r = d.get("r")
        if r is None:
            return r"$\text{---}$"
        cell = f"${_fmt_r(r)}$"
        ci_lo, ci_hi = d.get("ci_lo"), d.get("ci_hi")
        if ci_lo is not None and ci_hi is not None:
            cell += f" {{\\scriptsize $[{ci_lo:.2f},{ci_hi:.2f}]$}}"
        return cell

    def _ci_only(d: dict | None) -> str:
        d = d or {}
        ci_lo, ci_hi = d.get("ci_lo"), d.get("ci_hi")
        if ci_lo is None or ci_hi is None:
            return ""
        return f"{{\\scriptsize $[{ci_lo:.2f},{ci_hi:.2f}]$}}"

    rows = []
    for treatment, label in [("pure", "Pure"), ("comm", "Comm."), ("surveillance", "Surv.")]:
        bd = beliefs.get(treatment, {})
        if bd.get("status") == "missing" or not bd:
            continue
        n = _fmt_int_sep(bd.get("n"))
        r_post = bd.get("r_posterior_belief") or {}
        r_dec = bd.get("r_belief_decision") or {}
        r_part = (bd.get("r_partial") or {}).get("r")
        mean_b = bd.get("mean_belief")
        mean_b_cell = f"{mean_b * 100:.1f}" if mean_b is not None else "---"
        rows.append(
            f"{label} & {n} & ${_fmt_r(r_post.get('r'))}$ & "
            f"${_fmt_r(r_dec.get('r'))}$ & ${_fmt_r(r_part)}$ & {mean_b_cell} \\\\"
        )
        ci_row = f" & & {_ci_only(r_post)} & {_ci_only(r_dec)} & & \\\\"
        if ci_row.strip(" &\\"):
            rows.append(ci_row)

    tex = r"""\begin{table}[t]
\centering
\caption{Agent-level belief correlations (primary model: Mistral Small Creative).}
\label{tab:beliefs}
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{lccccc}
\toprule
& $N$ & $r_{\text{belief,posterior}}$ & $r_{\text{belief,decision}}$ & $r_{\text{partial}}$ & Mean belief (\%) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} $N$ counts agent-level observations (one elicited belief per agent decision). Beliefs are elicited on a 0--100 probability scale; mean belief is reported on that percent scale. $r_{\text{belief,posterior}}$: stated belief vs.\ the theoretical posterior $P(\theta < \theta^* \mid x_i)$ implied by the agent's signal. $r_{\text{belief,decision}}$: stated belief vs.\ the JOIN/STAY decision. $r_{\text{partial}}$: belief--decision partial correlation controlling for the signal z-score. 95\% Fisher-$z$ confidence intervals in brackets.
\end{tablenotes}
\end{table}
"""
    return tex


def render_tab_hypotheses(stats: dict) -> str:
    """Render hypothesis summary table (H1-H5) from verified stats."""
    hyp = stats.get("hypothesis_table")
    if not hyp:
        return "% No hypothesis table data available.\n"

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
        if stat != "---":
            stat = f"${stat}$"
        p = _fmt_p(h.get("p"))
        n = _fmt_int_sep(h.get("n"))
        es = _fmt_stat(h.get("effect_size"))
        if es != "---":
            es = f"${es}$"
        supported = h.get("supported", "---")
        rows.append(
            f"{hid} & {label} & {test} & {stat} & {n} & {p} & {es} & {supported} \\\\"
        )

    bonf_alpha = 0.05 / len(hyp) if hyp else 0.05
    bonf_survivors = ", ".join(
        h["id"]
        for h in hyp
        if h.get("p") is not None and h.get("p") == h.get("p") and h.get("p") <= bonf_alpha
    )
    if not bonf_survivors:
        bonf_survivors = "none"

    bonf_alpha_text = f"{bonf_alpha:.5f}".rstrip("0").rstrip(".")

    tex = r"""\begin{table}[t]
\centering
\caption{Hypothesis family and test results. H1--H4 use pooled benchmark-phase data across all seven models; H5 uses the primary model (Mistral Small Creative). Effect size: $r$ for correlations (H1--H3), Cohen's $d$ or $d_z$ for mean comparisons (H4--H5). Outcome labels use $\alpha = 0.05$; H2 is reported as a falsification check, not as evidence for a zero effect.}
\label{tab:hypotheses}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llllcccl}
\toprule
H & Hypothesis & Test & Statistic & $N$ & $p$ & Effect & Outcome \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} The Test column gives the statistic type for each row: Pearson $r$ for H1--H3 (the Statistic column reports $r$ itself) and paired $t$-tests for H4--H5 (the Statistic column reports $t$). $N$ counts period-level rows for H1--H3 (H2 uses the pooled scramble arm) and matched task cells for the paired tests H4--H5. H1--H4: pooled benchmark-phase data across all seven models (H4 uses a paired $t$-test matched on model/country/period/$\theta$ task cells). H5: primary model (Mistral Small Creative), paired on matched prompt-isolation cells. Bonferroni survivors at $\alpha = """
    tex += bonf_alpha_text
    tex += r"""$: """
    tex += bonf_survivors
    tex += r""".
\end{tablenotes}
\end{table}
"""
    return tex


def render_tab_cross_generator(stats: dict) -> str:
    """Cross-generator language variant robustness table."""
    cg = stats.get("cross_generator", {})
    if not cg:
        return "% No cross-generator data available.\n"

    models = ["Mistral Small Creative", "Llama 3.3 70B"]
    variants = ["baseline", "cable", "journalistic"]

    _short = {"Mistral Small Creative": "Mistral", "Llama 3.3 70B": "Llama 3.3 70B"}
    rows = []
    for m in models:
        m_data = cg.get(m, {})
        short = _short.get(m, m)
        for v in variants:
            d = m_data.get(v, {})
            if not d:
                rows.append(f"{short} & {v.capitalize()} & --- & --- & --- & --- \\\\")
                continue
            n = d.get("n_obs", "---")
            mean_j = _fmt_pct(d.get("mean_join"), 1)
            ra = d["r_vs_attack"]
            r_cell = f"${_fmt_r(ra['r'], 2)}$"
            if ra.get("ci_lo") is not None and ra.get("ci_hi") is not None:
                r_cell += f" {{\\scriptsize $[{ra['ci_lo']:.2f},{ra['ci_hi']:.2f}]$}}"
            fit = d.get("logistic_fit", {})
            cutoff = f"${fit['cutoff']:+.2f}$" if fit.get("cutoff") is not None else "---"
            rows.append(f"{short} & {v.capitalize()} & {n} & {mean_j} & {r_cell} & {cutoff} \\\\")
        rows.append(r"\midrule")
    # Remove trailing midrule
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    tex = r"""\begin{table}[t]
\centering
\caption{Cross-generator robustness. Three text rendering styles (baseline, diplomatic cable, journalistic wire) use identical slider functions and evidence items; only prose formatting differs. The Pearson $r(J, A(\theta))$ and logistic cutoff are virtually identical across generators.}
\label{tab:cross_generator}
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{llcccc}
\toprule
Model & Generator & $N$ & Mean join & $r(J,A)$ [95\% CI] & $\hat{\theta}^*$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} $N$ counts country--period rows ($n = 25$ agents per row); mean join in percent. 95\% Fisher-$z$ confidence intervals for $r$ in brackets. Cutoff $\hat{\theta}^*$ from the same logistic specification as Table~\ref{tab:logistic_params}, reported to two decimals. Runs use the core settings ($\sigma = 0.3$, temperature 0.7).
\end{tablenotes}
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

    def _cutoff_cell(fit: dict) -> str:
        c = (fit or {}).get("cutoff")
        return f"${c:+.2f}$" if c is not None else "---"

    # Mistral (from old)
    if temp_old:
        for key in ["T=0.3", "T=0.7", "T=1.0"]:
            d = temp_old.get(key, {})
            if not d:
                continue
            mean_j = _fmt_pct(d["mean_join"], 1)
            r_val = _fmt_r(d["r_vs_attack"]["r"], 2)
            cutoff = _cutoff_cell(d.get("logistic_fit", {}))
            rows.append(f"Mistral & {key} & {d.get('n_obs','---')} & {mean_j} & ${r_val}$ & {cutoff} \\\\")
        rows.append(r"\midrule")

    # Llama and Qwen (from new)
    for model in ["Llama 3.3 70B", "Qwen3 235B"]:
        m_data = temp_new.get(model, {})
        for temp in ["T=0.3", "T=0.5", "T=0.7", "T=1.0", "T=1.2"]:
            d = m_data.get(temp, {})
            if not d:
                continue
            mean_j = _fmt_pct(d["mean_join"], 1)
            r_val = _fmt_r(d["r_vs_attack"]["r"], 2)
            cutoff = _cutoff_cell(d.get("logistic_fit", {}))
            rows.append(f"{model} & {temp} & {d.get('n_obs','---')} & {mean_j} & ${r_val}$ & {cutoff} \\\\")
        rows.append(r"\midrule")

    if rows and rows[-1] == r"\midrule":
        rows.pop()

    tex = r"""\begin{table}[t]
\centering
\caption{Temperature robustness across three models. The pure global game is run at varying LLM decoding temperatures. The correlation $r(J, A(\theta))$ is stable across all temperatures and models.}
\label{tab:temperature_expanded}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
Model & $T$ & $N$ & Mean join & $r(J, A(\theta))$ & Cutoff $\hat{\theta}^*$ \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Pure treatment, varying LLM decoding temperature. $N$ counts country--period rows ($n = 25$ agents per row); mean join in percent. Mistral was run at only three temperatures (0.3, 0.7, 1.0) because the hosted endpoint was retired before the 0.5 and 1.2 cells could be collected; Llama 3.3 70B and Qwen3 235B cover the full five-point ladder.
\end{tablenotes}
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
    have_sd = False
    for m in models:
        m_data = pr.get(m, {})
        emitted = False
        for cond in conditions:
            d = m_data.get(cond, {})
            agent = d.get("agent_level", {}) if isinstance(d, dict) else {}
            if not agent:
                continue
            n = _fmt_int_sep(agent.get("n_agents"))
            mean_pr = _fmt_num(agent.get("mean_pr"), 1)
            sd_pr = agent.get("std_pr")
            if sd_pr is not None:
                have_sd = True
                mean_pr = f"{mean_pr} ({sd_pr:.1f})"
            pr_join = _fmt_num(agent.get("mean_pr_join"), 1)
            pr_stay = _fmt_num(agent.get("mean_pr_stay"), 1)
            short_m = "Mistral" if "Mistral" in m else "Llama 3.3 70B"
            model_cell = short_m if not emitted else ""
            emitted = True
            cond_label = {"pure": "Pure", "comm": "Comm.", "surveillance": "Surv."}.get(cond, cond.capitalize())
            rows.append(f"{model_cell} & {cond_label} & {n} & {mean_pr} & {pr_join} & {pr_stay} \\\\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    sd_note = (
        " Standard deviations of the overall rating in parentheses; conditional means are reported without dispersion measures, which are not stored in the verified statistics."
        if have_sd
        else " Dispersion measures are not stored in the verified statistics and are therefore omitted."
    )

    tex = r"""\begin{table}[t]
\centering
\caption{Elicited punishment risk (0--10 scale). Agents rate expected regime punishment after their JOIN/STAY decision. ``JOIN'' and ``STAY'' columns show the mean rating conditional on the agent's own decision.}
\label{tab:punishment_risk}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{llcccc}
\toprule
Model & Cond. & $N$ & Mean (SD) & $\mid$ JOIN & $\mid$ STAY \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Agent-level elicitation on a 0--10 scale; $N$ counts agent-level ratings (one per agent decision).""" + sd_note + r""" The punishment-risk module was fielded only on Mistral and Llama 3.3 70B; the other five models were not run with this elicitation.
\end{tablenotes}
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
        "baseline": "Baseline (no header)",
        "bc_high_cost": "High-cost header",
        "bc_low_cost": "Low-cost header",
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
            f"{label} & {n} & {pred*100:.1f}\\% & {actual*100:.1f}\\% & ${gap:+.1f}$ \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Narrative-header conditions: classifier vs.\ actual LLM behavior. Actual join rates shift sharply across strategic-stakes framing conditions; the classifier, trained on briefing-body text features alone, cannot predict this header-induced shift.}
\label{tab:bc_classifier}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
Condition & $N$ & Classif.\ pred. & Actual & Gap (pp) \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Primary model (Mistral Small Creative). $N$ counts country--period rows on the information-design $\theta$-grid ($n = 25$ agents per row). Logistic classifier trained on baseline slider features from the briefing body. Gap = classifier-predicted join rate $-$ actual LLM join rate (pp). The sign convention is the reverse of Table~\ref{tab:bc_statics}, which reports $\Delta = $ actual $-$ baseline: because the classifier's prediction stays near the baseline rate, a header that lowers actual joining shows up there as a negative $\Delta$ and here as a positive Gap of approximately the same magnitude. High/low rows are the two narrative-header conditions on the information-design grid, not numeric payoff-ladder endpoints.
\end{tablenotes}
\end{table}
"""
    return tex


def render_tab_parse_errors(stats: dict) -> str:
    pe = stats.get("parse_errors", {})
    if not pe:
        return "% No parse error data available.\n"

    models = DISPLAY_ORDER
    treatments = ["pure", "comm", "scramble", "flip"]
    treat_labels = {"pure": "Pure", "comm": "Comm", "scramble": "Scr.", "flip": "Flip"}
    _pe_short = {
        "Mistral Small Creative": "Mistral",
        "Llama 3.3 70B": "Llama 3.3 70B",
        "Qwen3 30B": "Qwen3 30B",
        "GPT-OSS 120B": "GPT-OSS 120B",
        "Qwen3 235B": "Qwen3 235B",
        "Trinity Large": "Trinity",
        "MiniMax M2-Her": "MiniMax M2",
    }

    rows = []
    for model in models:
        m_data = pe.get(model, {})
        if not m_data:
            continue
        first = True
        short_model = _pe_short.get(model, model)
        for t in treatments:
            t_data = m_data.get(t)
            if t_data is None:
                continue
            api_err = t_data.get("mean_api_error_rate", 0.0)
            unparse = t_data.get("mean_unparseable_rate", 0.0)
            combined = api_err + unparse
            n = t_data.get("n_periods", "---")
            model_col = short_model if first else ""
            first = False
            rows.append(
                f"{model_col} & {treat_labels[t]} & {n} & "
                f"{api_err*100:.1f}\\% & {unparse*100:.1f}\\% & {combined*100:.1f}\\% \\\\"
            )
        rows.append(r"\addlinespace")

    # Remove trailing \addlinespace
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()

    models_below_two = 0
    for model in models:
        treatment_rows = [
            pe.get(model, {}).get(t)
            for t in treatments
            if pe.get(model, {}).get(t) is not None
        ]
        if treatment_rows and all(
            (
                (row.get("mean_api_error_rate", 0.0) or 0.0)
                + (row.get("mean_unparseable_rate", 0.0) or 0.0)
            ) < 0.02
            for row in treatment_rows
        ):
            models_below_two += 1

    trinity_balance = ""
    trinity_rows = pe.get("Trinity Large", {})
    trinity_corrs = [
        row.get("api_error_theta_r")
        for row in trinity_rows.values()
        if isinstance(row, dict) and row.get("api_error_theta_r") is not None
    ]
    trinity_ps = [
        row.get("api_error_theta_p")
        for row in trinity_rows.values()
        if isinstance(row, dict) and row.get("api_error_theta_p") is not None
    ]
    if trinity_corrs and trinity_ps:
        trinity_balance = (
            " A signal-balance diagnostic for Trinity's per-period API-error rate gives "
            f"max $|r(\\theta, \\text{{error}})| = {max(abs(x) for x in trinity_corrs):.2f}$ "
            f"over the listed treatments (minimum $p {_fmt_p_text_plain(min(trinity_ps))}$)."
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Parse error and API failure rates by model and treatment.}
\label{tab:parse_errors}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
Model & Treat. & $N$ & API err & Unparse. & Combined \\
\midrule
"""
    _num_words = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    }
    below_word = _num_words.get(models_below_two, str(models_below_two))
    total_word = _num_words.get(len(models), str(len(models)))

    # Name the exceptions explicitly with their observed ranges.
    gpt = pe.get("GPT-OSS 120B", {})
    gpt_unp = [
        (gpt.get(t) or {}).get("mean_unparseable_rate")
        for t in ["scramble", "flip"]
    ]
    gpt_unp = [x for x in gpt_unp if x is not None]
    trin = pe.get("Trinity Large", {})
    trin_api = [
        (trin.get(t) or {}).get("mean_api_error_rate")
        for t in treatments
    ]
    trin_api = [x for x in trin_api if x is not None]
    exceptions_note = ""
    if gpt_unp and trin_api:
        exceptions_note = (
            f" The two exceptions are GPT-OSS 120B, with {min(gpt_unp)*100:.1f}--{max(gpt_unp)*100:.1f}\\% "
            f"unparseable responses in the scramble and flip arms, and Trinity, with "
            f"{min(trin_api)*100:.0f}--{max(trin_api)*100:.0f}\\% API errors from provider content filtering."
        )

    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize\emph{Notes:} Per-period averages. API error = provider-side failure (timeout, rate limit, content filter). Unparseable = valid response not classified as JOIN/STAY. Combined $< 2$\% in every listed treatment for """ + below_word + r""" of """ + total_word + r""" models.""" + exceptions_note + r""" Trinity's elevated API errors (${\approx}\,\TrinityAPIErrorPct\%$) mean its missingness is reported rather than resolved.""" + trinity_balance + r"""
\end{tablenotes}
\end{table}
"""
    return tex


def render_stats_belief_factorial(stats: dict) -> str:
    """Render belief-factorial macros from raw experiment logs."""
    bf = stats.get("belief_factorial", {}) or {}
    cells = bf.get("cells", {}) or {}
    effects = bf.get("effects", {}) or {}

    def _cell(cell: str, field: str):
        return (cells.get(cell) or {}).get(field)

    def _effect(name: str, field: str):
        return (effects.get(name) or {}).get(field)

    def _p_eq(name: str) -> str:
        p = _effect(name, "p_value")
        return "---" if p is None else _fmt_num(p, 2)

    def _p_lt(name: str) -> str:
        """A valid strict upper bound for p, for use as '$p < \\X$'.

        paper.tex uses these macros inside '$p < \\cdot$', so the stored value
        must always exceed the true p-value (it is a threshold, not a point
        estimate).
        """
        p = _effect(name, "p_value")
        if p is None:
            return "---"
        for threshold in (0.001, 0.01, 0.05):
            if p < threshold:
                return f"{threshold:.3f}".rstrip("0")  # 0.001 stays; 0.010 -> 0.01
        # Fallback: smallest 2dp value strictly above p
        import math as _math
        bound = _math.ceil(p * 100) / 100
        if bound <= p:
            bound += 0.01
        return f"{bound:.2f}"

    def _p_text(name: str) -> str:
        """Self-contained p text ('= 0.34' or '<0.001') for new-style usage."""
        p = _effect(name, "p_value")
        return _fmt_p_text_plain(p) if p is not None else "---"

    def _cell_delta(cell_a: str, cell_b: str, field: str, scale: float = 1.0):
        """Displayed-endpoint-consistent delta: round endpoints first (item:
        deltas must equal the difference of the rounded cells)."""
        a = _cell(cell_a, field)
        b = _cell(cell_b, field)
        if a is None or b is None:
            return None
        return _delta_of_displayed(float(a) * scale, float(b) * scale, 1)

    lines = [
        "% Belief factorial macros (2x2: surveillance x messages-in-beliefs)",
        "% Regenerated from revision-beliefs-post-* experiment logs (post-decision elicitation,",
        "% n = 12,500 agent-observations per cell = 500 country-periods x 25 agents).",
        "",
        "% Cell means: first-order beliefs",
        f"\\providecommand{{\\BFCommNoMsgBelief}}{{{_fmt_num(_cell('comm_nomsg', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFCommMsgBelief}}{{{_fmt_num(_cell('comm_msg', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvNoMsgBelief}}{{{_fmt_num(_cell('surv_nomsg', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvMsgBelief}}{{{_fmt_num(_cell('surv_msg', 'belief_mean'), 1)}}}",
        "",
        "% Cell means: second-order beliefs",
        f"\\providecommand{{\\BFCommNoMsgSOB}}{{{_fmt_num(_cell('comm_nomsg', 'second_order_mean'), 1)}}}",
        f"\\providecommand{{\\BFCommMsgSOB}}{{{_fmt_num(_cell('comm_msg', 'second_order_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvNoMsgSOB}}{{{_fmt_num(_cell('surv_nomsg', 'second_order_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvMsgSOB}}{{{_fmt_num(_cell('surv_msg', 'second_order_mean'), 1)}}}",
        "",
        "% Cell means: join rates",
        f"\\providecommand{{\\BFCommNoMsgJoin}}{{{_fmt_pct(_cell('comm_nomsg', 'join_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvNoMsgJoin}}{{{_fmt_pct(_cell('surv_nomsg', 'join_mean'), 1)}}}",
        f"\\providecommand{{\\BFCommMsgJoin}}{{{_fmt_pct(_cell('comm_msg', 'join_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvMsgJoin}}{{{_fmt_pct(_cell('surv_msg', 'join_mean'), 1)}}}",
        f"\\providecommand{{\\BFCellN}}{{{int(_cell('comm_msg', 'n') or 0):,}}}",
        "",
        "% Surveillance effect (comm -> surv), messages excluded",
        "% Deltas are computed from the displayed (rounded) cell means so the",
        "% printed cells and deltas are internally consistent.",
        f"\\providecommand{{\\BFSurvDeltaBelNoMsg}}{{{_fmt_num(_cell_delta('comm_nomsg', 'surv_nomsg', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvDeltaBelNoMsgP}}{{{_p_eq('surv_delta_belief_nomsg')}}}",
        f"\\providecommand{{\\BFSurvDeltaBelNoMsgPText}}{{{_p_text('surv_delta_belief_nomsg')}}}",
        "",
        "% Surveillance effect (comm -> surv), messages included",
        f"\\providecommand{{\\BFSurvDeltaBelMsg}}{{{_fmt_num(_cell_delta('comm_msg', 'surv_msg', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvDeltaBelMsgP}}{{{_p_lt('surv_delta_belief_msg')}}}",
        f"\\providecommand{{\\BFSurvDeltaBelMsgPText}}{{{_p_text('surv_delta_belief_msg')}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsg}}{{{_fmt_num(_cell_delta('comm_msg', 'surv_msg', 'second_order_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsgP}}{{{_p_lt('surv_delta_sob_msg')}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsgPText}}{{{_p_text('surv_delta_sob_msg')}}}",
        "",
        "% Surveillance effect on SOB, messages excluded",
        f"\\providecommand{{\\BFSurvDeltaSOBNoMsg}}{{{_fmt_num(_cell_delta('comm_nomsg', 'surv_nomsg', 'second_order_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBNoMsgP}}{{{_p_eq('surv_delta_sob_nomsg')}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBNoMsgPText}}{{{_p_text('surv_delta_sob_nomsg')}}}",
        "",
        "% Message effect on beliefs (with - without messages)",
        f"\\providecommand{{\\BFMsgEffectComm}}{{{_fmt_r(_cell_delta('comm_nomsg', 'comm_msg', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFMsgEffectSurv}}{{{_fmt_r(_cell_delta('surv_nomsg', 'surv_msg', 'belief_mean'), 1)}}}",
        "",
        "% Action shifts (join rates are displayed in percent, so deltas use",
        "% the rounded percent endpoints)",
        f"\\providecommand{{\\BFActionDeltaPP}}{{{_fmt_num(_cell_delta('comm_msg', 'surv_msg', 'join_mean', 100), 1)}}}",
        f"\\providecommand{{\\BFActionDeltaNoMsgPP}}{{{_fmt_num(_cell_delta('comm_nomsg', 'surv_nomsg', 'join_mean', 100), 1)}}}",
        f"\\providecommand{{\\BFActionDeltaMsgPP}}{{{_fmt_num(_cell_delta('comm_msg', 'surv_msg', 'join_mean', 100), 1)}}}",
        f"\\providecommand{{\\BFActionDeltaP}}{{{_p_lt('surv_delta_join_msg')}}}",
        f"\\providecommand{{\\BFActionDeltaPText}}{{{_p_text('surv_delta_join_msg')}}}",
        f"\\providecommand{{\\BFActionDeltaNoMsgPText}}{{{_p_text('surv_delta_join_nomsg')}}}",
        "",
        "% Pre-decision elicitation (messages-included only; the pre+nomsg cell",
        "% was not collected before mistral-small-creative was retired).",
        f"\\providecommand{{\\BFCommMsgSOBPre}}{{{_fmt_num(_cell('comm_msg_pre', 'second_order_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvMsgSOBPre}}{{{_fmt_num(_cell('surv_msg_pre', 'second_order_mean'), 1)}}}",
        f"\\providecommand{{\\BFCommMsgBeliefPre}}{{{_fmt_num(_cell('comm_msg_pre', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvMsgBeliefPre}}{{{_fmt_num(_cell('surv_msg_pre', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFCommMsgJoinPre}}{{{_fmt_pct(_cell('comm_msg_pre', 'join_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvMsgJoinPre}}{{{_fmt_pct(_cell('surv_msg_pre', 'join_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsgPre}}{{{_fmt_num(_cell_delta('comm_msg_pre', 'surv_msg_pre', 'second_order_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsgPreP}}{{{_p_lt('surv_delta_sob_msg_pre')}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsgPrePText}}{{{_p_text('surv_delta_sob_msg_pre')}}}",
        f"\\providecommand{{\\BFSurvDeltaBelMsgPre}}{{{_fmt_num(_cell_delta('comm_msg_pre', 'surv_msg_pre', 'belief_mean'), 1)}}}",
        f"\\providecommand{{\\BFSurvDeltaJoinMsgPrePP}}{{{_fmt_num(_cell_delta('comm_msg_pre', 'surv_msg_pre', 'join_mean', 100), 1)}}}",
        "",
    ]

    # ── Country-clustered inference (Issue: agent-level Welch tests treat the
    #    25 agents in a country-period as independent). Each contrast is
    #    collapsed to the 500 matched country-period cells, paired across arms,
    #    and clustered on the 10 countries: cluster-robust t (G-1 = 9 dof) plus
    #    an exact restricted wild-cluster bootstrap (full 2^10 enumeration).
    def _cl(name: str, field: str):
        return ((effects.get(name) or {}).get("cluster") or {}).get(field)

    def _wild_text(name: str) -> str:
        """Wild-cluster bootstrap p for '$p \\BF...WildP$' usage."""
        p = _cl(name, "p_wild_exact")
        if p is None:
            return "---"
        if p >= 0.01:
            return f"= {p:.2f}"
        return f"= {p:.3f}"  # exact floor with 10 clusters is 2/1024 = 0.002

    def _clnum(name: str, field: str, nd: int) -> str:
        return _fmt_num(_cl(name, field), nd)

    n_cl = _cl("surv_delta_join_msg", "n_clusters")
    n_ce = _cl("surv_delta_join_msg", "n_cells")
    lines += [
        "% Country-clustered inference on the matched country-period cells.",
        "% Cluster-robust t has G-1 dof; wild-cluster bootstrap is exact (2^G).",
        f"\\providecommand{{\\BFNClusters}}{{{int(n_cl) if n_cl else 10}}}",
        f"\\providecommand{{\\BFNCells}}{{{int(n_ce) if n_ce else 500}}}",
        f"\\providecommand{{\\BFClusterDoF}}{{{int(_cl('surv_delta_join_msg', 'dof') or 9)}}}",
        "",
        "% Wild-cluster bootstrap p-values (self-contained, for '$p \\X$' usage)",
        f"\\providecommand{{\\BFSurvDeltaBelNoMsgWildP}}{{{_wild_text('surv_delta_belief_nomsg')}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBNoMsgWildP}}{{{_wild_text('surv_delta_sob_nomsg')}}}",
        f"\\providecommand{{\\BFActionNoMsgWildP}}{{{_wild_text('surv_delta_join_nomsg')}}}",
        f"\\providecommand{{\\BFSurvDeltaBelMsgWildP}}{{{_wild_text('surv_delta_belief_msg')}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsgWildP}}{{{_wild_text('surv_delta_sob_msg')}}}",
        f"\\providecommand{{\\BFActionMsgWildP}}{{{_wild_text('surv_delta_join_msg')}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsgPreWildP}}{{{_wild_text('surv_delta_sob_msg_pre')}}}",
        "",
        "% Cluster-robust t-statistics (G-1 dof) for the key SOB / action wedge",
        f"\\providecommand{{\\BFSurvDeltaSOBNoMsgClT}}{{{_clnum('surv_delta_sob_nomsg', 't_stat', 1)}}}",
        f"\\providecommand{{\\BFActionNoMsgClT}}{{{_clnum('surv_delta_join_nomsg', 't_stat', 1)}}}",
        f"\\providecommand{{\\BFSurvDeltaSOBMsgClT}}{{{_clnum('surv_delta_sob_msg', 't_stat', 1)}}}",
        "",
    ]
    return "\n".join(lines)


def render_tab_llama_ledger(stats: dict) -> str:
    r"""Ledger of every Llama 3.3 70B surveillance estimate.

    Static table that references existing macros (defined in stats_macros.tex
    and stats_nested.tex, both \input before the appendix), so the grid, the
    matched support, and the effect size for each Llama estimate stay
    consistent with the rest of the paper and cannot drift. Added so that the
    -8.0 pp nested estimate and the noisy -15.0 pp 20-cell estimate are never
    conflated as "the full Llama" number again.
    """
    return r"""% Auto-generated by analysis/render_paper_tables.py. Do not edit by hand.
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\caption{Every Llama 3.3 70B surveillance estimate: effect size, matched support, grid, and purpose. All deltas are surveillance (or the stated arm) minus its own baseline on the matched cells. The full-scale nested estimate ($\NestedSurvDeltaPP$~pp, $N=\NestedSurvN$) is the headline cross-model confirmation; the original prompt-isolation estimate ($\PromptIsoLlamaDeltaPP$~pp) rests on only 20 overlapping cells and is a noisy sign check, not a full replication.}
\label{tab:llama_ledger}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcll}
\toprule
Arm & $\Delta$ (pp) & Matched $N$ & Grid & Purpose \\
\midrule
\multicolumn{5}{l}{\textit{Surveillance (monitoring warning), coordination task:}} \\
\quad Nested clean surveillance & $\NestedSurvDeltaPP$ & \NestedSurvN & nested $10\times50$, common seed & full-scale confirmation (\S\ref{sec:surveillance}) \\
\quad Prompt-isolation (original) & $\PromptIsoLlamaDeltaPP$ & 20 & 20-cell pilot/rerun overlap & original sign check (Table~\ref{tab:prompt_isolation}) \\
\quad Mild routine-monitoring note & $\NestedMildDeltaPP$ & \NestedMildN & nested $10\times50$ & dose response (Appendix~\ref{sec:dose_response}) \\
\quad Severe-consequence note & $\NestedSevereDeltaPP$ & \NestedSevereN & nested $10\times50$ & dose response (Appendix~\ref{sec:dose_response}) \\
\midrule
\multicolumn{5}{l}{\textit{Replay / decoding of surveilled messages, coordination task:}} \\
\quad Raw-message replay & $\NestedRawReplayDeltaPP$ & \NestedRawReplayN & nested $10\times50$ & validates replay design ($\approx$ live) \\
\quad Decoded-message replay & $\NestedDecodedDeltaPP$ & \NestedDecodedN & nested $10\times50$ & decoding restores participation \\
\midrule
\multicolumn{5}{l}{\textit{Cross-model writer--reader rotation:}} \\
\quad Within-Llama (matched baseline) & $\XModelWithinLlamaDeltaPP$ & \XModelWithinLlamaNPairs & rotation cells & reflexivity baseline (\S\ref{sec:surveillance}) \\
\quad Llama writes, Qwen3 30B reads & $\XModelLlamaQwenDeltaPP$ & \XModelLlamaQwenNPairs & rotation cells & cross-family transmission \\
\quad Qwen3 30B writes, Llama reads & $\XModelQwenLlamaDeltaPP$ & \XModelQwenLlamaNPairs & rotation cells & cross-family transmission \\
\midrule
\multicolumn{5}{l}{\textit{Other decision task:}} \\
\quad Private bet on regime fall & $\CrossTaskDeltaPP$ & \CrossTaskN & cross-task grid & task contrast (\S\ref{sec:surveillance}) \\
\midrule
\multicolumn{5}{l}{\textit{Reference control (no monitoring framing):}} \\
\quad Codedness-induction control & $\NestedStyleDeltaPP$ & \NestedStyleN & nested $10\times50$ & isolates coded form alone \\
\bottomrule
\end{tabular}
}
\begin{tablenotes}
\footnotesize\emph{Notes:} ``Matched $N$'' is the number of common-support cells entering each paired contrast. ``Nested'' arms share the identical 500-cell grid ($10$ countries $\times 50$ periods, $n=25$, common seed) and are matched by construction. The original prompt-isolation estimate shares only 20 cells with its pilot baseline because those grids were not nested. Rotation contrasts are matched on (country, period, $\theta$) cells across writer and reader models. The private-bet and nested-coordination samples are not cell-matched to each other.
\end{tablenotes}
\end{table}
"""


def main() -> None:
    stats = _load()

    tables = {
        "tab_llama_ledger.tex": render_tab_llama_ledger(stats),
        "tab_models.tex": render_tab_models(stats),
        "tab_main_results.tex": render_tab_main_results(stats),
        "tab_comm_estimators.tex": render_tab_comm_estimators(stats),
        "tab_logistic_params.tex": render_tab_logistic_params(stats),
        "tab_surveillance_variants.tex": render_tab_surveillance_variants(stats),
        "tab_prompt_isolation.tex": render_tab_prompt_isolation(stats),
        "tab_bc_statics.tex": render_tab_bc_statics(stats),
        "tab_beliefs.tex": render_tab_beliefs(stats),
        "tab_hypotheses.tex": render_tab_hypotheses(stats),
        "tab_classifiers.tex": render_tab_classifiers(stats),
        "tab_msg_features.tex": render_tab_msg_features(stats),
        "tab_cross_generator.tex": render_tab_cross_generator(stats),
        "tab_temperature_expanded.tex": render_tab_temperature_expanded(stats),
        "tab_punishment_risk.tex": render_tab_punishment_risk(stats),
        "tab_parse_errors.tex": render_tab_parse_errors(stats),
        "tab_bc_classifier.tex": render_tab_bc_classifier(stats),
        "stats_macros.tex": render_stats_macros(stats),
        "stats_belief_factorial.tex": render_stats_belief_factorial(stats),
    }

    for name, content in tables.items():
        _write(OUT_DIR / name, content)

    print(f"Wrote {len(tables)} table(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
