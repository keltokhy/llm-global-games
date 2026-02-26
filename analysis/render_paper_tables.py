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


def _fmt_pct(x: float | None, nd: int = 1) -> str:
    """Format a fraction as a percentage (e.g., 0.237 -> '23.7')."""
    if x is None:
        return "---"
    try:
        if x != x:  # nan
            return "---"
    except Exception:
        return "---"
    return f"{x * 100:.{nd}f}"


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

    def fall_pct(m: str) -> str:
        d = part1.get(m, {}).get("pure", {})
        return _fmt_pct(d.get("regime_fall_rate"))

    def n_pure(m: str) -> str:
        d = part1.get(m, {}).get("pure", {})
        return str(d.get("n_obs") or "---")

    rows = []
    for m in models:
        rows.append(
            f"{m} & {r_attack(m,'pure')} & {r_attack(m,'comm')} & {r_attack(m,'scramble')} & {r_attack(m,'flip')} & {n_pure(m)} & {mean_join(m)} & {fall_pct(m)} \\\\"
        )

    pooled = part1.get("_pooled_pure", {}).get("r_vs_attack", {}).get("r")
    pooled_comm = part1.get("_pooled_comm", {}).get("r_vs_attack", {}).get("r")
    pooled_scr = part1.get("_pooled_scramble", {}).get("r_vs_attack", {}).get("r")
    pooled_flip = part1.get("_pooled_flip", {}).get("r_vs_attack", {}).get("r")
    pooled_n = part1.get("_pooled_pure", {}).get("n_obs")
    pooled_mean = part1.get("_pooled_pure", {}).get("mean_join")
    pooled_fall = part1.get("_pooled_pure", {}).get("regime_fall_rate")

    mean_pure = part1.get("_mean_r_pure_vs_attack")
    mean_comm = mean_r_attack("comm")
    mean_scr = mean_r_attack("scramble")
    mean_flip = mean_r_attack("flip")

    tex = r"""\begin{table*}[t]
\centering
\caption{Equilibrium alignment by model and treatment. Cells report Pearson $r$ between the empirical join fraction and the theoretical attack mass $A(\theta)$. Fall~\% is the fraction of periods in which the regime falls ($\text{join fraction} > \theta$) under the pure treatment.}
\label{tab:main_results}
\small
\begin{tabular}{lccccccc}
\toprule
& \multicolumn{2}{c}{Main treatments} & \multicolumn{2}{c}{Falsification} & & & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Model & Pure & Comm & Scramble & Flip & $n_{\text{pure}}$ & Mean join & Fall~\% \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\midrule
\textbf{Pooled} & $""" + _fmt_r(pooled, 2) + r"""$ & $""" + _fmt_r(pooled_comm, 2) + r"""$ & $""" + _fmt_r(pooled_scr, 2) + r"""$ & $""" + _fmt_r(pooled_flip, 2) + r"""$ & """ + f"{pooled_n}" + r""" & """ + _fmt_mean(pooled_mean, 2) + r""" & """ + _fmt_pct(pooled_fall) + r""" \\
\textbf{Mean across models} & """ + r_cell(mean_pure, 2) + r""" & """ + r_cell(mean_comm, 2) + r""" & """ + r_cell(mean_scr, 2) + r""" & """ + r_cell(mean_flip, 2) + r""" & --- & --- & --- \\
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
        fall = d.get("regime_fall_rate")
        r = (d.get("r_vs_theta") or {}).get("r")
        delta = d.get("delta_vs_baseline")
        n = d.get("n_obs")
        delta_cell = "---" if delta is None else _fmt_r(delta, nd=3).replace("+", "+")
        rows.append(
            f"{label} & {_fmt_num(mean,3)} & {_fmt_pct(fall)} & ${_fmt_r(r,3)}$ & {delta_cell} & {n} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Information design treatment summary (primary model: Mistral Small Creative). $r$ is the Pearson correlation between $\theta$ and join fraction. Fall~\% is the fraction of periods in which the regime falls.}
\label{tab:infodesign_summary}
\small
\begin{tabular}{lccccc}
\toprule
Design & Mean & Fall~\% & $r$ & $\Delta$ & $N$ \\
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
    base_fall = base.get("regime_fall_rate")

    def row_prop(k: int) -> tuple[str, dict]:
        d = regime["propaganda"][f"k={k}"]["Mistral Small Creative"]
        return f"Prop $k={k}$", d

    prop_rows = [row_prop(2), row_prop(5), row_prop(10)]
    surv = regime["surveillance"]["Mistral Small Creative"]
    ps = regime["propaganda_surveillance"]["Mistral Small Creative"]

    lines = []
    lines.append(f"Comm (baseline) & {_fmt_num(base_mean,3).lstrip('0')} & {_fmt_num(base_mean,3).lstrip('0')} & ${_fmt_r(base_r,3)}$ & --- & {_fmt_pct(base_fall)} \\\\")
    lines.append(r"\midrule")

    for label, d in prop_rows:
        mean_all = d["mean_join_all"]
        mean_real = d.get("mean_join_real")
        r = d["r_vs_theta_all"]["r"]
        delta_real = d.get("delta_real_vs_baseline_pp")
        delta_cell = "---" if delta_real is None else f"{_fmt_r(delta_real/100,3)}"
        fall = d.get("regime_fall_rate")
        lines.append(
            f"{label} & {_fmt_num(mean_all,3).lstrip('0')} & {_fmt_num(mean_real,3).lstrip('0')} & ${_fmt_r(r,3)}$ & {delta_cell} & {_fmt_pct(fall)} \\\\"
        )

    lines.append(r"\midrule")
    surv_fall = surv.get("regime_fall_rate")
    lines.append(
        f"Surveillance & {_fmt_num(surv['mean_join'],3).lstrip('0')} & {_fmt_num(surv['mean_join'],3).lstrip('0')} & ${_fmt_r(surv['r_vs_theta']['r'],3)}$ & {_fmt_r(surv['delta_vs_baseline_pp']/100,3)} & {_fmt_pct(surv_fall)} \\\\"
    )
    ps_fall = ps.get("regime_fall_rate")
    lines.append(
        f"Prop+Surv & {_fmt_num(ps['mean_join_all'],3).lstrip('0')} & --- & ${_fmt_r(ps['r_vs_theta_all']['r'],3)}$ & --- & {_fmt_pct(ps_fall)} \\\\"
    )

    tex = r"""\begin{table}[t]
\centering
\caption{Propaganda and surveillance effects (primary model: Mistral Small Creative). ``All'' includes propaganda agents; ``Real'' excludes them (computed from logs). $\Delta$ is the change in real-agent mean join vs.\ baseline communication. Fall~\% is the fraction of periods in which the regime falls.}
\label{tab:surveillance_propaganda}
\small
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{2}{c}{Mean join} & & & \\
\cmidrule(lr){2-3}
Treatment & All & Real & $r$ & $\Delta$ & Fall~\% \\
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

    # Fall rates: no-surveillance from infodesign, surveillance from sxc
    nosurv_fall = sxc.get("nosurv_fall_rate", {})
    surv_fall = sxc.get("fall_rate", {})

    lines = []
    for label, key in [("Baseline", "baseline"), ("Upper cens.", "censor_upper"), ("Lower cens.", "censor_lower")]:
        no = {"baseline": baseline, "censor_upper": up, "censor_lower": lo}[key]
        yes = float(sxc[key])
        delta = yes - no
        no_fall = nosurv_fall.get(key)
        yes_fall = surv_fall.get(key)
        lines.append(
            f"{label} & {_fmt_num(no,3)} & {_fmt_num(yes,3)} & {_fmt_r(delta,nd=3)} & {_fmt_pct(no_fall)} & {_fmt_pct(yes_fall)} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Surveillance $\times$ censorship interaction (primary model: Mistral Small Creative). Fall~\% columns show the fraction of periods in which the regime falls.}
\label{tab:surv_censor}
\small
\begin{tabular}{lccccc}
\toprule
& \multicolumn{2}{c}{Mean join} & & \multicolumn{2}{c}{Fall~\%} \\
\cmidrule(lr){2-3} \cmidrule(lr){5-6}
Design & No Surv. & Surv. & $\Delta$ & No Surv. & Surv. \\
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


def render_tab_beliefs(stats: dict) -> str:
    """Belief analysis table (A7): correlation, partial r, join rate by bin."""
    beliefs = stats.get("beliefs", {})
    treatments = ["pure", "comm", "surveillance", "propaganda_k5"]
    labels = {"pure": "Pure", "comm": "Communication",
              "surveillance": "Surveillance", "propaganda_k5": "Propaganda $k{=}5$"}

    rows = []
    for t in treatments:
        b = beliefs.get(t, {})
        if not isinstance(b, dict) or "n" not in b:
            continue
        r_post = (b.get("r_posterior_belief") or {}).get("r")
        r_bel_dec = (b.get("r_belief_decision") or {}).get("r")
        r_partial = (b.get("r_partial_belief_decision_given_signal") or {}).get("r")
        n = b.get("n")
        mean_bel = b.get("mean_belief")
        rows.append(
            f"{labels.get(t, t)} & {n} & ${_fmt_r(r_post)}$ & ${_fmt_r(r_bel_dec)}$ "
            f"& ${_fmt_r(r_partial)}$ & {_fmt_num(mean_bel, 3)} \\\\"
        )

    # Cross-treatment
    cross = beliefs.get("_cross_pure_vs_surv", {})
    cross_row = ""
    if cross:
        cross_row = (
            r"\midrule" "\n"
            r"\multicolumn{6}{l}{\textit{Pure $\to$ Surveillance shift: "
            f"$\\Delta$belief $= {_fmt_r(cross.get('belief_shift'), 3)}$, "
            f"$\\Delta$action $= {_fmt_r(cross.get('action_shift'), 3)}$"
            r"}}" " \\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Belief elicitation analysis (primary model: Mistral Small Creative). $r_{\text{post}}$: correlation between Bayesian posterior and stated belief. $r_{\text{b,d}}$: belief--decision correlation. $r_{\text{partial}}$: partial correlation of belief and decision controlling for signal.}
\label{tab:beliefs}
\small
\begin{tabular}{lccccc}
\toprule
Treatment & $N$ & $r_{\text{post}}$ & $r_{\text{b,d}}$ & $r_{\text{partial}}$ & Mean belief \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    if cross_row:
        tex += cross_row + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def render_tab_message_content(stats: dict) -> str:
    """Message content table (A7/B3): keyword frequencies and action-signaling."""
    msg = stats.get("message_content", {})
    treatments = ["comm", "surveillance", "propaganda_k5"]
    labels = {"comm": "Comm", "surveillance": "Surveillance",
              "propaganda_k5": "Prop $k{=}5$"}

    # Determine which keywords to include (all in PAPER_KEYWORDS)
    keywords = ["act", "fight", "ready", "moment", "patience", "loyal",
                "stable", "strong", "cautious", "risk"]

    # Header
    kw_header = " & ".join([f"\\texttt{{{kw}}}" for kw in keywords])

    rows = []
    for t in treatments:
        entry = msg.get(t, {})
        if not isinstance(entry, dict) or "keyword_freq_pct" not in entry:
            continue
        kw_freqs = entry["keyword_freq_pct"]
        cells = []
        for kw in keywords:
            val = kw_freqs.get(kw)
            cells.append(f"{val:.1f}" if val is not None else "---")
        action_sig = entry.get("action_signaling_rate")
        msg_len = entry.get("mean_msg_length")
        n = entry.get("n_obs")
        rows.append(
            f"{labels.get(t, t)} & {n} & " + " & ".join(cells)
            + f" & {_fmt_num(action_sig, 2) if action_sig is not None else '---'}"
            + f" & {int(msg_len) if msg_len is not None else '---'}"
            + r" \\"
        )

    tex = r"""\begin{table*}[t]
\centering
\caption{Message content by treatment (primary model: Mistral Small Creative). Keyword columns show frequency (\%\ of words). Action-signal: fraction of messages with more action than caution words. Length: mean characters.}
\label{tab:message_content}
\tiny
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lc""" + "c" * len(keywords) + r"""cc}
\toprule
Treatment & $N$ & """ + kw_header + r""" & Act-sig & Len \\
\midrule
"""
    tex += "\n".join(rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    return tex


def render_tab_calibration_robustness(stats: dict) -> str:
    """Calibration robustness table (B2): r_vs_theta, calibrated centers."""
    part1 = stats.get("part1", {})
    text_baseline = stats.get("text_baseline", {})
    models = [
        "Mistral Small Creative", "Llama 3.3 70B", "OLMo 3 7B",
        "Ministral 3B", "Qwen3 30B", "GPT-OSS 120B",
        "Qwen3 235B", "Trinity Large", "MiniMax M2-Her",
    ]

    rows = []
    for m in models:
        entry = part1.get(m, {})
        pure = entry.get("pure", {})
        if not isinstance(pure, dict) or "r_vs_theta" not in pure:
            continue
        r_theta = (pure.get("r_vs_theta") or {}).get("r")
        r_attack = (pure.get("r_vs_attack") or {}).get("r")
        rmse = pure.get("rmse_vs_attack")
        tb = text_baseline.get(m, {})
        text_slope = tb.get("logistic_slope")

        rows.append(
            f"{m} & ${_fmt_r(r_theta)}$ & ${_fmt_r(r_attack)}$ "
            f"& {_fmt_num(rmse, 3) if rmse is not None else '---'} "
            f"& {_fmt_num(text_slope, 1) if text_slope is not None else '---'} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Calibration robustness. $r_\theta$: raw correlation with regime strength. $r_A$: correlation with theoretical attack mass. RMSE: root mean squared error vs.\ $A(\theta)$. Text slope: logistic slope of na\"ive $1 - \text{direction}$ predictor.}
\label{tab:calibration_robustness}
\small
\begin{tabular}{lcccc}
\toprule
Model & $r_\theta$ & $r_A$ & RMSE & Text slope \\
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
        "tab_beliefs.tex": render_tab_beliefs(stats),
        "tab_message_content.tex": render_tab_message_content(stats),
        "tab_calibration_robustness.tex": render_tab_calibration_robustness(stats),
    }

    for name, content in tables.items():
        _write(OUT_DIR / name, content)

    print(f"Wrote {len(tables)} table(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
