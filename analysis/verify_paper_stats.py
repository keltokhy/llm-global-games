"""
Verify and regenerate ALL statistics reported in the paper.

Single source of truth: reads raw CSVs, computes every number,
outputs verified_stats.json for paper table generation.

Usage: uv run python verify_paper_stats.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "output"
OUT = Path(__file__).resolve().parent / "verified_stats.json"

# ── Models with full Part I data ──────────────────────────────────────

PART1_MODELS = [
    "mistralai--mistral-small-creative",
    "meta-llama--llama-3.3-70b-instruct",
    "allenai--olmo-3-7b-instruct",
    "mistralai--ministral-3b-2512",
    "qwen--qwen3-30b-a3b-instruct-2507",
    "openai--gpt-oss-120b",
    "qwen--qwen3-235b-a22b-2507",
    "arcee-ai--trinity-large-preview_free",
    "minimax--minimax-m2-her",
]

SHORT = {
    "mistralai--mistral-small-creative": "Mistral Small Creative",
    "meta-llama--llama-3.3-70b-instruct": "Llama 3.3 70B",
    "allenai--olmo-3-7b-instruct": "OLMo 3 7B",
    "mistralai--ministral-3b-2512": "Ministral 3B",
    "qwen--qwen3-30b-a3b-instruct-2507": "Qwen3 30B",
    "openai--gpt-oss-120b": "GPT-OSS 120B",
    "qwen--qwen3-235b-a22b-2507": "Qwen3 235B",
    "arcee-ai--trinity-large-preview_free": "Trinity Large",
    "minimax--minimax-m2-her": "MiniMax M2-Her",
}


def load(model: str, treatment: str) -> pd.DataFrame:
    """Load a summary CSV, return empty DataFrame if missing."""
    p = ROOT / model / f"experiment_{treatment}_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_infodesign(model: str, design: str = "all") -> pd.DataFrame:
    """Load infodesign summary CSV."""
    p = ROOT / model / f"experiment_infodesign_{design}_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def pearson_with_ci(x, y, alpha=0.05):
    """Pearson r with Fisher-z confidence interval and p-value."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {"r": float("nan"), "p": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": n}
    r, p = stats.pearsonr(x, y)
    # Fisher z-transform for CI
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    return {"r": round(r, 4), "p": round(p, 6), "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4), "n": int(n)}


def fisher_z_test(r1, n1, r2, n2):
    """Fisher z-test for difference between two independent correlations."""
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z_stat = (z1 - z2) / se
    p = 2 * stats.norm.sf(abs(z_stat))
    return {"z": round(z_stat, 4), "p": round(p, 6)}

def _join_col(df: pd.DataFrame) -> str:
    if "join_fraction_valid" in df.columns and df["join_fraction_valid"].notna().any():
        return "join_fraction_valid"
    return "join_fraction"


def _safe_mean(x) -> float:
    x = pd.Series(x).dropna()
    return float(x.mean()) if len(x) else float("nan")


def _safe_std(x) -> float:
    x = pd.Series(x).dropna()
    return float(x.std()) if len(x) else float("nan")


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _find_summaries(base_dir: Path, pattern: str = "experiment_*_summary.csv") -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(base_dir.glob(f"**/{pattern}"))


def _model_slug_from_summary_path(path: Path, base_dir: Path) -> str | None:
    """Infer model slug as the parent folder under base_dir."""
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    return parts[0]


def _load_experiment_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _real_join_from_comm_log(log_rows: list[dict]) -> pd.Series:
    """Compute per-period join fraction among non-propaganda agents from comm logs."""
    vals = []
    for row in log_rows:
        agents = row.get("agents") or []
        real = [a for a in agents if not a.get("is_propaganda", False)]
        if not real:
            vals.append(float("nan"))
            continue
        n_join = sum(1 for a in real if a.get("decision") == "JOIN")
        vals.append(n_join / len(real))
    return pd.Series(vals, dtype=float)


# ═══════════════════════════════════════════════════════════════════
# PART I: Main treatments (pure, comm, scramble, flip)
# ═══════════════════════════════════════════════════════════════════

def compute_part1():
    """Compute all Part I statistics by model and pooled."""
    results = {}
    all_pure = []
    all_comm = []
    all_scramble = []
    all_flip = []

    def _infer_n_agents(df: pd.DataFrame) -> int | None:
        if not {"n_join", "join_fraction"}.issubset(df.columns):
            return None
        d = df.dropna(subset=["n_join", "join_fraction"]).copy()
        d = d[d["join_fraction"] > 0]
        if len(d) == 0:
            return None
        est = (d["n_join"] / d["join_fraction"]).round().astype(int)
        mode = est.mode()
        return int(mode.iloc[0]) if len(mode) else None

    for model in PART1_MODELS:
        m = {}
        name = SHORT[model]
        n_agents = None

        for treatment in ["pure", "comm", "scramble", "flip"]:
            df = load(model, treatment)
            if len(df) == 0:
                m[treatment] = {"status": "missing"}
                continue

            jcol = _join_col(df)
            jf = df[jcol].astype(float).values
            theta = df["theta"].astype(float).values
            n_agents = n_agents or _infer_n_agents(df)

            # Correlation with theta (what the code actually computes)
            r_theta = pearson_with_ci(theta, jf)

            # Correlation with theoretical attack mass (what the paper claims)
            if "theoretical_attack" in df.columns:
                attack = df["theoretical_attack"].astype(float).values
                r_attack = pearson_with_ci(attack, jf)
                mask = np.isfinite(attack) & np.isfinite(jf)
                rmse = float(np.sqrt(np.mean((jf[mask] - attack[mask]) ** 2))) if mask.any() else float("nan")
                mae = float(np.mean(np.abs(jf[mask] - attack[mask]))) if mask.any() else float("nan")
            else:
                r_attack = {"r": float("nan"), "note": "theoretical_attack column missing"}
                rmse = float("nan")
                mae = float("nan")

            m[treatment] = {
                "r_vs_theta": r_theta,
                "r_vs_attack": r_attack,
                "n_obs": len(df),
                "mean_join": round(_safe_mean(jf), 4),
                "std_join": round(_safe_std(jf), 4),
                "mean_theta": round(_safe_mean(theta), 4),
                "rmse_vs_attack": round(rmse, 4) if np.isfinite(rmse) else None,
                "mae_vs_attack": round(mae, 4) if np.isfinite(mae) else None,
            }

            if treatment == "pure":
                all_pure.append(df)
            elif treatment == "comm":
                all_comm.append(df)
            elif treatment == "scramble":
                all_scramble.append(df)
            elif treatment == "flip":
                all_flip.append(df)

        if n_agents is not None:
            m["_n_agents"] = int(n_agents)

        # Communication effect (unpaired + paired-on-task-key, if both exist)
        pure_df = load(model, "pure")
        comm_df = load(model, "comm")
        jcol = _join_col(pure_df) if len(pure_df) else "join_fraction"
        if len(pure_df) > 0 and len(comm_df) > 0:
            pure_y = pure_df[jcol].astype(float)
            comm_y = comm_df[jcol].astype(float)
            delta_unpaired = _safe_mean(comm_y) - _safe_mean(pure_y)
            t_stat_u, t_p_u = stats.ttest_ind(comm_y.dropna(), pure_y.dropna())

            # Pair by task key (country,period,theta,z,benefit,theta_star), averaging duplicates.
            key_cols = [c for c in ["country", "period", "theta", "z", "benefit", "theta_star"] if c in pure_df.columns and c in comm_df.columns]
            paired = {}
            if key_cols:
                pure_g = pure_df.groupby(key_cols, as_index=False)[jcol].mean().rename(columns={jcol: "pure"})
                comm_g = comm_df.groupby(key_cols, as_index=False)[jcol].mean().rename(columns={jcol: "comm"})
                merged = pure_g.merge(comm_g, on=key_cols, how="inner").dropna(subset=["pure", "comm"])
                if len(merged):
                    diff = merged["comm"] - merged["pure"]
                    t_stat_p, t_p_p = stats.ttest_1samp(diff, 0.0)
                    paired = {
                        "n_pairs": int(len(diff)),
                        "delta_pp": round(float(diff.mean()) * 100, 2),
                        "t_stat": round(float(t_stat_p), 4),
                        "p_value": round(float(t_p_p), 6),
                    }

            m["comm_effect"] = {
                "unpaired": {
                    "delta_pp": round(delta_unpaired * 100, 2),
                    "t_stat": round(float(t_stat_u), 4),
                    "p_value": round(float(t_p_u), 6),
                },
                "paired": paired if paired else {"status": "missing"},
            }

        results[name] = m

    # Pooled across all models
    def _pooled_entry(dfs: list[pd.DataFrame]) -> dict:
        if not dfs:
            return {}
        pooled = pd.concat(dfs, ignore_index=True)
        jcol = _join_col(pooled)
        n_agents = _infer_n_agents(pooled)
        out = {
            "r_vs_theta": pearson_with_ci(pooled["theta"], pooled[jcol]),
            "n_obs": int(len(pooled)),
            "mean_join": round(_safe_mean(pooled[jcol]), 4),
        }
        if "theoretical_attack" in pooled.columns:
            out["r_vs_attack"] = pearson_with_ci(pooled["theoretical_attack"], pooled[jcol])
            attack = pooled["theoretical_attack"].astype(float).values
            jf = pooled[jcol].astype(float).values
            mask = np.isfinite(attack) & np.isfinite(jf)
            if mask.any():
                out["rmse_vs_attack"] = round(float(np.sqrt(np.mean((jf[mask] - attack[mask]) ** 2))), 4)
                out["mae_vs_attack"] = round(float(np.mean(np.abs(jf[mask] - attack[mask]))), 4)
        if n_agents is not None:
            out["n_agents"] = int(n_agents)
        return out

    results["_pooled_pure"] = _pooled_entry(all_pure) if all_pure else {}
    results["_pooled_comm"] = _pooled_entry(all_comm) if all_comm else {}
    results["_pooled_scramble"] = _pooled_entry(all_scramble) if all_scramble else {}
    results["_pooled_flip"] = _pooled_entry(all_flip) if all_flip else {}

    # Pooled communication effect (unpaired + paired-on-task-key with model included)
    if all_pure and all_comm:
        pp = pd.concat(all_pure, ignore_index=True)
        pc = pd.concat(all_comm, ignore_index=True)
        jcol = _join_col(pp)
        delta_unpaired = _safe_mean(pc[jcol]) - _safe_mean(pp[jcol])
        t_stat_u, t_p_u = stats.ttest_ind(pc[jcol].dropna(), pp[jcol].dropna())

        key_cols = [c for c in ["model", "country", "period", "theta", "z", "benefit", "theta_star"] if c in pp.columns and c in pc.columns]
        paired = {}
        if key_cols:
            pp_g = pp.groupby(key_cols, as_index=False)[jcol].mean().rename(columns={jcol: "pure"})
            pc_g = pc.groupby(key_cols, as_index=False)[jcol].mean().rename(columns={jcol: "comm"})
            merged = pp_g.merge(pc_g, on=key_cols, how="inner").dropna(subset=["pure", "comm"])
            if len(merged):
                diff = merged["comm"] - merged["pure"]
                t_stat_p, t_p_p = stats.ttest_1samp(diff, 0.0)
                paired = {
                    "n_pairs": int(len(diff)),
                    "delta_pp": round(float(diff.mean()) * 100, 2),
                    "t_stat": round(float(t_stat_p), 4),
                    "p_value": round(float(t_p_p), 6),
                }

        results["_pooled_comm_effect"] = {
            "unpaired": {
                "delta_pp": round(delta_unpaired * 100, 2),
                "t_stat": round(float(t_stat_u), 4),
                "p_value": round(float(t_p_u), 6),
            },
            "paired": paired if paired else {"status": "missing"},
        }

        # Fisher z tests on pooled r(A, join) (treated as independent correlations).
        try:
            r_pure = results["_pooled_pure"]["r_vs_attack"]["r"]
            n_pure = results["_pooled_pure"]["r_vs_attack"]["n"]
            r_scr = results["_pooled_scramble"]["r_vs_attack"]["r"]
            n_scr = results["_pooled_scramble"]["r_vs_attack"]["n"]
            r_flip = results["_pooled_flip"]["r_vs_attack"]["r"]
            n_flip = results["_pooled_flip"]["r_vs_attack"]["n"]
            results["_fisher_pure_vs_scramble_attack"] = fisher_z_test(r_pure, n_pure, r_scr, n_scr)
            results["_fisher_pure_vs_flip_attack"] = fisher_z_test(r_pure, n_pure, r_flip, n_flip)
        except Exception:
            pass

    # Convenience: falsification sample sizes per model (scramble+flip)
    falsif_n = {}
    for model in PART1_MODELS:
        name = SHORT[model]
        m = results.get(name, {})
        n_scr = (m.get("scramble") or {}).get("n_obs") if isinstance(m.get("scramble"), dict) else None
        n_flip = (m.get("flip") or {}).get("n_obs") if isinstance(m.get("flip"), dict) else None
        if isinstance(n_scr, int) and isinstance(n_flip, int):
            falsif_n[name] = int(n_scr + n_flip)
    results["_falsification_n_by_model"] = falsif_n

    # Mean of per-model r values
    model_rs_theta = []
    model_rs_attack = []
    for model in PART1_MODELS:
        name = SHORT[model]
        if name in results and "pure" in results[name]:
            entry = results[name]["pure"]
            if isinstance(entry, dict) and "r_vs_theta" in entry:
                r_t = entry["r_vs_theta"].get("r")
                r_a = entry["r_vs_attack"].get("r") if isinstance(entry.get("r_vs_attack"), dict) else None
                if r_t is not None and not np.isnan(r_t):
                    model_rs_theta.append(r_t)
                if r_a is not None and not np.isnan(r_a):
                    model_rs_attack.append(r_a)

    results["_mean_r_pure_vs_theta"] = round(np.mean(model_rs_theta), 4) if model_rs_theta else None
    results["_mean_r_pure_vs_attack"] = round(np.mean(model_rs_attack), 4) if model_rs_attack else None
    results["_mean_abs_r_pure_vs_theta"] = round(np.mean(np.abs(model_rs_theta)), 4) if model_rs_theta else None

    return results


# ═══════════════════════════════════════════════════════════════════
# PART II: Information design
# ═══════════════════════════════════════════════════════════════════

def compute_infodesign():
    """Compute statistics for information design experiments."""
    results = {}

    # Primary model: Mistral
    model = "mistralai--mistral-small-creative"
    df_all = load_infodesign(model, "all")
    if len(df_all) == 0:
        print("  WARNING: no infodesign data for primary model")
        return results

    jcol = "join_fraction_valid" if "join_fraction_valid" in df_all.columns else "join_fraction"

    designs = df_all["design"].unique() if "design" in df_all.columns else []
    for design in designs:
        sub = df_all[df_all["design"] == design]
        r_theta = pearson_with_ci(sub["theta"], sub[jcol])
        r_attack = pearson_with_ci(sub["theoretical_attack"], sub[jcol]) \
            if "theoretical_attack" in sub.columns else {}
        results[design] = {
            "mean_join": round(sub[jcol].mean(), 4),
            "r_vs_theta": r_theta,
            "r_vs_attack": r_attack,
            "n_obs": len(sub),
        }

    # Treatment effects relative to baseline
    if "baseline" in results:
        baseline_mean = results["baseline"]["mean_join"]
        for design in results:
            if design != "baseline":
                delta = results[design]["mean_join"] - baseline_mean
                results[design]["delta_vs_baseline"] = round(delta, 4)

    # Cross-model infodesign replication
    cross = {}
    for m in PART1_MODELS:
        df = load_infodesign(m, "all")
        if len(df) == 0:
            continue
        name = SHORT[m]
        cross[name] = {}
        for design in df["design"].unique():
            sub = df[df["design"] == design]
            r_t = pearson_with_ci(sub["theta"], sub[jcol])
            cross[name][design] = {
                "mean_join": round(sub[jcol].mean(), 4),
                "r_vs_theta": r_t,
                "n_obs": len(sub),
            }
    results["_cross_model"] = cross

    return results


# ═══════════════════════════════════════════════════════════════════
# PART III: Surveillance, propaganda, interactions
# ═══════════════════════════════════════════════════════════════════

def compute_regime_control():
    """Statistics for surveillance, propaganda, and interactions."""
    results = {}

    # Surveillance (comm baseline vs surveilled comm), per model
    surv_base = ROOT / "surveillance"
    for f in _find_summaries(surv_base, "experiment_comm_summary.csv"):
        model_slug = _model_slug_from_summary_path(f, surv_base)
        if model_slug is None:
            continue
        df = _load_summary(f)
        jcol = _join_col(df)
        out = {
            "mean_join": round(_safe_mean(df[jcol]), 4),
            "r_vs_theta": pearson_with_ci(df["theta"], df[jcol]),
            "n_obs": int(len(df)),
        }
        # Delta vs baseline comm (main output dir)
        base_comm = _load_summary(ROOT / model_slug / "experiment_comm_summary.csv")
        if len(base_comm):
            bj = _join_col(base_comm)
            out["delta_vs_baseline_pp"] = round((out["mean_join"] - _safe_mean(base_comm[bj])) * 100, 2)
            out["baseline_mean_join"] = round(_safe_mean(base_comm[bj]), 4)
        results.setdefault("surveillance", {})[SHORT.get(model_slug, model_slug)] = out

    # Propaganda k=2,5,10
    for k in [2, 5, 10]:
        prop_base = ROOT / f"propaganda-k{k}"
        for f in _find_summaries(prop_base, "experiment_comm_summary.csv"):
            model_slug = _model_slug_from_summary_path(f, prop_base)
            if model_slug is None:
                continue
            df = _load_summary(f)
            jcol = _join_col(df)
            out = {
                "mean_join_all": round(_safe_mean(df[jcol]), 4),
                "r_vs_theta_all": pearson_with_ci(df["theta"], df[jcol]),
                "n_obs": int(len(df)),
            }
            # Real-citizen join fraction from comm log
            log_path = f.parent / "experiment_comm_log.json"
            logs = _load_experiment_log(log_path)
            if logs:
                real = _real_join_from_comm_log(logs)
                out["mean_join_real"] = round(_safe_mean(real), 4)
                out["std_join_real"] = round(_safe_std(real), 4)
                out["real_delta_vs_all_pp"] = round((out["mean_join_real"] - out["mean_join_all"]) * 100, 2)

            # Delta vs baseline comm (main output dir), using real if available, else all
            base_comm = _load_summary(ROOT / model_slug / "experiment_comm_summary.csv")
            if len(base_comm):
                bj = _join_col(base_comm)
                baseline_mean = _safe_mean(base_comm[bj])
                out["baseline_mean_join"] = round(baseline_mean, 4)
                out["delta_vs_baseline_pp"] = round((out["mean_join_all"] - baseline_mean) * 100, 2)
                if "mean_join_real" in out:
                    out["delta_real_vs_baseline_pp"] = round((out["mean_join_real"] - baseline_mean) * 100, 2)

            results.setdefault("propaganda", {}).setdefault(f"k={k}", {})[SHORT.get(model_slug, model_slug)] = out

    # Propaganda + surveillance
    ps_base = ROOT / "propaganda-surveillance"
    for f in _find_summaries(ps_base, "experiment_comm_summary.csv"):
        model_slug = _model_slug_from_summary_path(f, ps_base)
        if model_slug is None:
            continue
        df = _load_summary(f)
        jcol = _join_col(df)
        out = {
            "mean_join_all": round(_safe_mean(df[jcol]), 4),
            "r_vs_theta_all": pearson_with_ci(df["theta"], df[jcol]),
            "n_obs": int(len(df)),
        }
        log_path = f.parent / "experiment_comm_log.json"
        logs = _load_experiment_log(log_path)
        if logs:
            real = _real_join_from_comm_log(logs)
            out["mean_join_real"] = round(_safe_mean(real), 4)
        results.setdefault("propaganda_surveillance", {})[SHORT.get(model_slug, model_slug)] = out

    # Surveillance x censorship
    sxc_base = ROOT / "surveillance-x-censorship"
    for f in _find_summaries(sxc_base, "experiment_infodesign_all_summary.csv"):
        model_slug = _model_slug_from_summary_path(f, sxc_base)
        if model_slug is None:
            continue
        df = _load_summary(f)
        jcol = _join_col(df)
        by_design = df.groupby("design")[jcol].mean().to_dict()
        out = {k: round(float(v), 4) for k, v in by_design.items()}
        # deltas vs no-surveillance infodesign baseline
        base_info = _load_summary(ROOT / model_slug / "experiment_infodesign_all_summary.csv")
        if len(base_info):
            bj = _join_col(base_info)
            base_means = base_info.groupby("design")[bj].mean().to_dict()
            out["delta_vs_nosurv_pp"] = {
                k: round((float(v) - float(base_means.get(k, np.nan))) * 100, 2)
                for k, v in by_design.items()
                if k in base_means
            }
        results.setdefault("surveillance_x_censorship", {})[SHORT.get(model_slug, model_slug)] = out

    return results


# ═══════════════════════════════════════════════════════════════════
# ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════

def compute_robustness():
    """Statistics for robustness checks."""
    results = {}

    # Agent count variations
    for n in [5, 10, 50, 100]:
        d = ROOT / f"mistralai--mistral-small-creative-n{n}"
        for f in _find_summaries(d, "experiment_pure_summary.csv"):
            model_slug = _model_slug_from_summary_path(f, d)
            if model_slug is None:
                continue
            df = _load_summary(f)
            jcol = _join_col(df)
            out = {
                "r_vs_theta": pearson_with_ci(df["theta"], df[jcol]),
                "mean_join": round(_safe_mean(df[jcol]), 4),
                "n_obs": int(len(df)),
            }
            if "theoretical_attack" in df.columns:
                out["r_vs_attack"] = pearson_with_ci(df["theoretical_attack"], df[jcol])
            results.setdefault("agent_count", {}).setdefault(f"n={n}", {})[SHORT.get(model_slug, model_slug)] = out

    # Network k=8
    k8_base = ROOT / "network-k8"
    for f in _find_summaries(k8_base, "experiment_comm_summary.csv"):
        model_slug = _model_slug_from_summary_path(f, k8_base)
        if model_slug is None:
            continue
        df = _load_summary(f)
        jcol = _join_col(df)
        out = {
            "mean_join": round(_safe_mean(df[jcol]), 4),
            "r_vs_theta": pearson_with_ci(df["theta"], df[jcol]),
        }
        if "theoretical_attack" in df.columns:
            out["r_vs_attack"] = pearson_with_ci(df["theoretical_attack"], df[jcol])
        results.setdefault("network_k8", {})[SHORT.get(model_slug, model_slug)] = out

    # Mixed-model
    for dname, treatment in [("mixed-5model-pure", "experiment_pure_summary.csv"), ("mixed-5model-comm", "experiment_comm_summary.csv")]:
        dp = ROOT / dname
        for f in _find_summaries(dp, treatment):
            model_slug = _model_slug_from_summary_path(f, dp)
            if model_slug is None:
                continue
            df = _load_summary(f)
            jcol = _join_col(df)
            out = {
                "mean_join": round(_safe_mean(df[jcol]), 4),
                "r_vs_theta": pearson_with_ci(df["theta"], df[jcol]),
                "n_obs": int(len(df)),
            }
            if "theoretical_attack" in df.columns:
                out["r_vs_attack"] = pearson_with_ci(df["theoretical_attack"], df[jcol])
            results.setdefault(dname, {})[SHORT.get(model_slug, model_slug)] = out

    # Bandwidth robustness (infodesign), primary model
    for bw_dir in ["bandwidth-005", "bandwidth-030"]:
        dp = ROOT / bw_dir
        for f in _find_summaries(dp, "experiment_infodesign_all_summary.csv"):
            model_slug = _model_slug_from_summary_path(f, dp)
            if model_slug is None:
                continue
            df = _load_summary(f)
            jcol = _join_col(df)
            by_design = df.groupby("design")[jcol].mean().to_dict()
            results.setdefault("bandwidth", {}).setdefault(bw_dir, {})[SHORT.get(model_slug, model_slug)] = {
                k: round(float(v), 4) for k, v in by_design.items()
            }

    return results


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════

def compute_calibration():
    """Compute calibration statistics for all models.

    Loads calibrated_params JSON and autocalibrate_history CSV per model.
    Reports cutoff_center, rounds to convergence, and final loss.
    """
    results = {}
    all_centers = []

    for model in PART1_MODELS:
        name = SHORT[model]
        entry = {}

        # Load calibrated params
        params_path = ROOT / model / f"calibrated_params_{model}.json"
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)
            cc = params.get("cutoff_center", float("nan"))
            entry["cutoff_center"] = round(cc, 4)
            entry["abs_cutoff_center"] = round(abs(cc), 4)
            all_centers.append(cc)
        else:
            entry["cutoff_center"] = None

        # Load autocalibrate history
        hist_path = ROOT / model / "autocalibrate_history.csv"
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            n_rounds = int(hist["round"].max())
            entry["n_rounds"] = n_rounds

            # Initial (round 1) fitted_center = uncalibrated bias
            entry["uncalibrated_center"] = round(float(hist.iloc[0]["fitted_center"]), 4)

            # Final round fitted_center = residual after calibration
            entry["final_fitted_center"] = round(float(hist.iloc[-1]["fitted_center"]), 4)

            # Convergence: did |fitted_center| < 0.15 at termination?
            final_fc = abs(float(hist.iloc[-1]["fitted_center"]))
            entry["converged"] = bool(final_fc < 0.15)

            # Final loss
            loss_col = "calibration_loss"
            if loss_col in hist.columns:
                entry["final_loss"] = round(float(hist.iloc[-1][loss_col]), 4)

            # Final fitted slope (emergent)
            entry["final_fitted_slope"] = round(float(hist.iloc[-1]["fitted_slope"]), 4)

        results[name] = entry

    # Aggregate statistics
    if all_centers:
        abs_centers = [abs(c) for c in all_centers]
        results["_summary"] = {
            "n_models": len(all_centers),
            "mean_abs_cutoff_center": round(float(np.mean(abs_centers)), 4),
            "median_abs_cutoff_center": round(float(np.median(abs_centers)), 4),
            "max_abs_cutoff_center": round(float(np.max(abs_centers)), 4),
            "min_abs_cutoff_center": round(float(np.min(abs_centers)), 4),
            "range_cutoff_center": [
                round(float(np.min(all_centers)), 4),
                round(float(np.max(all_centers)), 4),
            ],
            "largest_adjustment_model": SHORT[PART1_MODELS[
                int(np.argmax(abs_centers))
            ]],
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# DISCREPANCY ANALYSIS (optional; kept minimal to avoid drift)
# ═══════════════════════════════════════════════════════════════════

def discrepancy_report(all_stats):
    """Light sanity check: flag missing files or NaNs in key pooled stats."""
    report = []
    part1 = all_stats.get("part1", {})
    pooled_pure = part1.get("_pooled_pure", {})
    pooled_comm = part1.get("_pooled_comm", {})
    for label, entry in [("pooled_pure", pooled_pure), ("pooled_comm", pooled_comm)]:
        r = (entry.get("r_vs_attack") or {}).get("r")
        if r is None or (isinstance(r, float) and np.isnan(r)):
            report.append({"issue": f"{label} missing r_vs_attack"})
    return report


# ═══════════════════════════════════════════════════════════════════
# OLS regression (pooled)
# ═══════════════════════════════════════════════════════════════════

def pooled_ols(all_stats):
    """Run OLS: join_fraction = b0 + b1 * A(theta) + epsilon, pooled."""
    all_pure = []
    for model in PART1_MODELS:
        df = load(model, "pure")
        if len(df) > 0:
            all_pure.append(df)
    if not all_pure:
        return {}
    pooled = pd.concat(all_pure, ignore_index=True)
    jcol = "join_fraction_valid" if "join_fraction_valid" in pooled.columns else "join_fraction"
    y = pooled[jcol].values
    x = pooled["theoretical_attack"].values if "theoretical_attack" in pooled.columns else None
    if x is None:
        return {}
    # OLS with intercept
    X = np.column_stack([np.ones_like(x), x])
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot
    return {
        "intercept": round(beta[0], 4),
        "slope": round(beta[1], 4),
        "r_squared": round(r_squared, 4),
        "n_obs": len(y),
    }


def main():
    print("Computing Part I statistics...")
    part1 = compute_part1()

    print("Computing information design statistics...")
    infodesign = compute_infodesign()

    print("Computing regime control statistics...")
    regime = compute_regime_control()

    print("Computing robustness statistics...")
    robust = compute_robustness()

    print("Computing calibration statistics...")
    calibration = compute_calibration()

    print("Running pooled OLS...")
    ols = pooled_ols(None)

    all_stats = {
        "part1": part1,
        "infodesign": infodesign,
        "regime_control": regime,
        "robustness": robust,
        "calibration": calibration,
        "pooled_ols": ols,
    }

    print("\nRunning discrepancy analysis...")
    discrepancies = discrepancy_report(all_stats)
    all_stats["discrepancy_report"] = discrepancies

    # Print summary
    print("\n" + "="*70)
    print("KEY SUMMARY")
    print("="*70)

    mean_r_theta = part1.get("_mean_r_pure_vs_theta")
    mean_r_attack = part1.get("_mean_r_pure_vs_attack")
    mean_abs_r = part1.get("_mean_abs_r_pure_vs_theta")
    print(f"  Mean r(θ, join) across models:      {mean_r_theta}")
    print(f"  Mean |r(θ, join)| across models:     {mean_abs_r}")
    print(f"  Mean r(A(θ), join) across models:    {mean_r_attack}")

    pooled_pure = part1.get("_pooled_pure", {})
    if pooled_pure:
        print(f"  Pooled r(θ, join):                   {pooled_pure.get('r_vs_theta', {}).get('r')}")
        print(f"  Pooled r(A(θ), join):                {pooled_pure.get('r_vs_attack', {}).get('r')}")
        print(f"  Pooled N:                            {pooled_pure.get('n_obs')}")
        print(f"  Pooled mean join:                    {pooled_pure.get('mean_join')}")

    if ols:
        print(f"  OLS: join = {ols['intercept']} + {ols['slope']} * A(θ), R² = {ols['r_squared']}")

    # Save
    with open(OUT, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nFull results saved to {OUT}")


if __name__ == "__main__":
    main()
