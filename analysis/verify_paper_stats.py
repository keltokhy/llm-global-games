"""
Verify and regenerate ALL statistics reported in the paper.

Single source of truth: reads raw CSVs, computes every number,
outputs verified_stats.json for paper table generation.

Usage: uv run python verify_paper_stats.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

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
# A1: BELIEF ANALYSIS
# ═══════════════════════════════════════════════════════════════════

# Belief data sources
_MISTRAL_DIR = ROOT / "mistralai--mistral-small-creative"
_BELIEF_SOURCES = {
    "pure": _MISTRAL_DIR / "_overwrite_200period_backup" / "experiment_pure_beliefs_log.json",
    "surveillance": _MISTRAL_DIR / "_overwrite_200period_backup" / "experiment_surveillance_beliefs_log.json",
    "comm": _MISTRAL_DIR / "_beliefs_comm" / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
    "propaganda_k5": _MISTRAL_DIR / "_beliefs_propaganda_k5" / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
}


def _load_belief_agents(log_path: Path) -> list[dict]:
    """Extract flat agent-level records with beliefs from a log file."""
    if not log_path.exists():
        return []
    with open(log_path) as f:
        periods = json.load(f)
    sigma = 0.3
    rows = []
    for p in periods:
        theta = p["theta"]
        theta_star = p["theta_star"]
        x_star = theta_star + sigma * stats.norm.ppf(max(min(theta_star, 1 - 1e-6), 1e-6))
        for a in p["agents"]:
            if a.get("belief") is None or a.get("api_error"):
                continue
            if a.get("is_propaganda"):
                continue
            signal = a["signal"]
            belief = a["belief"] / 100.0
            decision = 1 if a["decision"] == "JOIN" else 0
            posterior = stats.norm.cdf((theta_star - signal) / sigma)
            rows.append({
                "theta": theta, "theta_star": theta_star, "signal": signal,
                "z_score": a["z_score"], "belief": belief, "decision": decision,
                "posterior": posterior,
            })
    return rows


def compute_beliefs() -> dict:
    """Compute belief analysis statistics (A1)."""
    results = {}
    all_treatments = {}

    for treatment, path in _BELIEF_SOURCES.items():
        rows = _load_belief_agents(path)
        if not rows:
            results[treatment] = {"status": "missing", "path": str(path)}
            continue
        all_treatments[treatment] = rows
        n = len(rows)
        beliefs = np.array([r["belief"] for r in rows])
        decisions = np.array([r["decision"] for r in rows])
        signals = np.array([r["signal"] for r in rows])
        posteriors = np.array([r["posterior"] for r in rows])

        # Core correlations
        r_post = pearson_with_ci(posteriors, beliefs)
        r_bel_dec = pearson_with_ci(beliefs, decisions)

        # Partial r(belief, decision | signal): residualize both on signal
        slope_ds, int_ds = np.polyfit(signals, decisions, 1)
        resid_d = decisions - (int_ds + slope_ds * signals)
        slope_bs, int_bs = np.polyfit(signals, beliefs, 1)
        resid_b = beliefs - (int_bs + slope_bs * signals)
        r_partial = pearson_with_ci(resid_b, resid_d)

        # Join rate by belief bin (5 bins)
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        bin_stats = []
        for lo, hi in bins:
            mask = (beliefs >= lo) & (beliefs < hi)
            if mask.sum() > 0:
                bin_stats.append({
                    "bin": f"{lo:.0%}-{hi:.0%}",
                    "join_rate": round(float(decisions[mask].mean()), 4),
                    "mean_belief": round(float(beliefs[mask].mean()), 4),
                    "n": int(mask.sum()),
                })

        join_beliefs = beliefs[decisions == 1]
        stay_beliefs = beliefs[decisions == 0]

        entry = {
            "n": n,
            "mean_belief": round(float(beliefs.mean()), 4),
            "std_belief": round(float(beliefs.std()), 4),
            "mean_join_rate": round(float(decisions.mean()), 4),
            "r_posterior_belief": r_post,
            "r_belief_decision": r_bel_dec,
            "r_partial_belief_decision_given_signal": r_partial,
            "mean_belief_join": round(float(join_beliefs.mean()), 4) if len(join_beliefs) else None,
            "mean_belief_stay": round(float(stay_beliefs.mean()), 4) if len(stay_beliefs) else None,
            "join_rate_by_bin": bin_stats,
        }
        results[treatment] = entry

    # Cross-treatment comparisons (pure vs surveillance)
    if "pure" in all_treatments and "surveillance" in all_treatments:
        pure_b = np.array([r["belief"] for r in all_treatments["pure"]])
        surv_b = np.array([r["belief"] for r in all_treatments["surveillance"]])
        pure_d = np.array([r["decision"] for r in all_treatments["pure"]])
        surv_d = np.array([r["decision"] for r in all_treatments["surveillance"]])
        t_bel, p_bel = stats.ttest_ind(pure_b, surv_b)
        t_dec, p_dec = stats.ttest_ind(pure_d, surv_d)
        results["_cross_pure_vs_surv"] = {
            "belief_shift": round(float(surv_b.mean() - pure_b.mean()), 4),
            "action_shift": round(float(surv_d.mean() - pure_d.mean()), 4),
            "t_belief": round(float(t_bel), 4),
            "p_belief": round(float(p_bel), 6),
            "t_action": round(float(t_dec), 4),
            "p_action": round(float(p_dec), 6),
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# A2-A3, A5: MESSAGE CONTENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════

ACTION_WORDS = {
    "act", "action", "rise", "rising", "revolt", "rebel", "join", "fight",
    "resist", "overthrow", "protest", "strike", "march", "mobilize", "move",
    "now", "cracking", "crumbling", "collapse", "collapsing", "falling",
    "weak", "weakening", "fracture", "fracturing", "breaking", "fragile",
    "opportunity", "moment", "window", "momentum", "ready", "time",
    "together", "unite", "unified", "solidarity", "everyone",
}

CAUTION_WORDS = {
    "wait", "careful", "caution", "cautious", "patience", "patient", "risk",
    "risky", "dangerous", "danger", "trap", "stable", "strong", "strength",
    "grip", "control", "powerful", "secure", "security", "surveillance",
    "monitor", "watching", "uncertain", "unclear", "premature",
    "hold", "hesitate", "steady", "firm", "loyal", "intact",
    "suppress", "crackdown", "retaliate", "punish",
}

# Specific keywords cited in the paper
PAPER_KEYWORDS = [
    "act", "fight", "ready", "moment", "patience", "loyal", "stable",
    "strong", "cautious", "risk", "unite", "together", "cracking",
]

_MSG_LOG_PATHS = {
    "comm": ROOT / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
    "surveillance": ROOT / "surveillance" / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
    "propaganda_k5": ROOT / "propaganda-k5" / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
    "propaganda_k10": ROOT / "propaganda-k10" / "mistralai--mistral-small-creative" / "experiment_comm_log.json",
}


def _extract_msg_features(message: str) -> dict | None:
    """Extract text features from a single message."""
    if not message:
        return None
    words = re.findall(r"[a-z]+", message.lower())
    n_words = len(words)
    if n_words == 0:
        return None
    n_action = sum(1 for w in words if w in ACTION_WORDS)
    n_caution = sum(1 for w in words if w in CAUTION_WORDS)
    # Per-keyword frequencies
    kw_freq = {}
    for kw in PAPER_KEYWORDS:
        kw_freq[f"kw_{kw}"] = sum(1 for w in words if w == kw) / n_words
    return {
        "msg_length": len(message),
        "word_count": n_words,
        "n_action": n_action,
        "n_caution": n_caution,
        "action_score": (n_action - n_caution) / n_words,
        **kw_freq,
    }


def _load_msg_agents(path: Path) -> pd.DataFrame:
    """Load comm log into agent-level DataFrame with message features."""
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        data = json.load(f)
    rows = []
    for period in data:
        theta = period["theta"]
        for agent in period["agents"]:
            if agent.get("is_propaganda"):
                continue
            msg = agent.get("message_sent", "")
            if not msg:
                continue
            feats = _extract_msg_features(msg)
            if feats is None:
                continue
            decision = 1 if agent.get("decision", "").upper() == "JOIN" else 0
            rows.append({
                "theta": theta,
                "z_score": agent["z_score"],
                "direction": agent.get("direction", np.nan),
                "decision": decision,
                **feats,
            })
    return pd.DataFrame(rows)


def compute_message_content() -> dict:
    """Compute message content statistics (A2-A3, A5)."""
    results = {}

    for treatment, path in _MSG_LOG_PATHS.items():
        df = _load_msg_agents(path)
        if len(df) == 0:
            results[treatment] = {"status": "missing"}
            continue

        entry = {"n_obs": len(df)}

        # Per-keyword frequencies
        kw_cols = [c for c in df.columns if c.startswith("kw_")]
        kw_freqs = {}
        for col in kw_cols:
            kw = col[3:]  # strip "kw_"
            kw_freqs[kw] = round(float(df[col].mean() * 100), 2)  # percent
        entry["keyword_freq_pct"] = kw_freqs

        # Mean message length
        entry["mean_msg_length"] = round(float(df["msg_length"].mean()), 1)
        entry["mean_word_count"] = round(float(df["word_count"].mean()), 1)

        # Action-signaling classification (A3):
        # A message is "action-signaling" if action_score > 0
        # (more action words than caution words)
        join_mask = df["decision"] == 1
        stay_mask = df["decision"] == 0

        # Among JOIN agents: fraction with action_score > 0
        if join_mask.sum() > 0:
            entry["action_signaling_join"] = round(
                float((df.loc[join_mask, "action_score"] > 0).mean()), 4
            )
        # Among STAY agents: fraction with caution > action (action_score < 0)
        if stay_mask.sum() > 0:
            entry["caution_signaling_stay"] = round(
                float((df.loc[stay_mask, "action_score"] < 0).mean()), 4
            )

        # Overall action-signaling rate
        entry["action_signaling_rate"] = round(
            float((df["action_score"] > 0).mean()), 4
        )

        # R² of text features → θ
        text_cols = ["word_count", "action_score", "msg_length"]
        text_cols = [c for c in text_cols if c in df.columns]
        if text_cols and len(df) > 10:
            X = df[text_cols].values
            y = df["theta"].values
            X_full = np.column_stack([np.ones(len(X)), X])
            try:
                beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
                y_hat = X_full @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                entry["R2_text_to_theta"] = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
            except Exception:
                entry["R2_text_to_theta"] = None

        results[treatment] = entry

    # Cross-treatment comparisons and t-tests
    if "comm" in results and "surveillance" in results:
        comm_df = _load_msg_agents(_MSG_LOG_PATHS["comm"])
        surv_df = _load_msg_agents(_MSG_LOG_PATHS["surveillance"])
        if len(comm_df) > 0 and len(surv_df) > 0:
            # T-test on action_score
            t_as, p_as = stats.ttest_ind(comm_df["action_score"], surv_df["action_score"])
            # Fisher z-test on per-keyword differences
            kw_tests = {}
            for kw in PAPER_KEYWORDS:
                col = f"kw_{kw}"
                if col in comm_df.columns and col in surv_df.columns:
                    t_kw, p_kw = stats.ttest_ind(comm_df[col], surv_df[col])
                    kw_tests[kw] = {
                        "comm_pct": round(float(comm_df[col].mean() * 100), 2),
                        "surv_pct": round(float(surv_df[col].mean() * 100), 2),
                        "t_stat": round(float(t_kw), 4),
                        "p_value": round(float(p_kw), 6),
                    }
            results["_comm_vs_surv"] = {
                "action_score_t": round(float(t_as), 4),
                "action_score_p": round(float(p_as), 6),
                "keyword_tests": kw_tests,
            }

    # Propaganda keyword shifts (A5)
    if "comm" in results and "propaganda_k5" in results:
        comm_df = _load_msg_agents(_MSG_LOG_PATHS["comm"])
        prop_df = _load_msg_agents(_MSG_LOG_PATHS["propaganda_k5"])
        if len(comm_df) > 0 and len(prop_df) > 0:
            kw_tests = {}
            for kw in PAPER_KEYWORDS:
                col = f"kw_{kw}"
                if col in comm_df.columns and col in prop_df.columns:
                    t_kw, p_kw = stats.ttest_ind(comm_df[col], prop_df[col])
                    kw_tests[kw] = {
                        "comm_pct": round(float(comm_df[col].mean() * 100), 2),
                        "prop_k5_pct": round(float(prop_df[col].mean() * 100), 2),
                        "t_stat": round(float(t_kw), 4),
                        "p_value": round(float(p_kw), 6),
                    }
            # Caution-coded messages among STAY agents
            comm_stay = comm_df[comm_df["decision"] == 0]
            prop_stay = prop_df[prop_df["decision"] == 0]
            comm_caution_rate = float((comm_stay["action_score"] < 0).mean()) if len(comm_stay) > 0 else None
            prop_caution_rate = float((prop_stay["action_score"] < 0).mean()) if len(prop_stay) > 0 else None
            results["_comm_vs_propaganda_k5"] = {
                "keyword_tests": kw_tests,
                "caution_stay_comm": round(comm_caution_rate, 4) if comm_caution_rate is not None else None,
                "caution_stay_prop_k5": round(prop_caution_rate, 4) if prop_caution_rate is not None else None,
            }

    return results


# ═══════════════════════════════════════════════════════════════════
# A4: SURVEILLANCE T-TEST
# ═══════════════════════════════════════════════════════════════════

def compute_surveillance_ttest() -> dict:
    """Unpaired t-test: surveillance join fraction vs baseline comm (A4)."""
    results = {}
    surv_base = ROOT / "surveillance"
    for f in _find_summaries(surv_base, "experiment_comm_summary.csv"):
        model_slug = _model_slug_from_summary_path(f, surv_base)
        if model_slug is None:
            continue
        surv_df = _load_summary(f)
        base_comm = _load_summary(ROOT / model_slug / "experiment_comm_summary.csv")
        if len(surv_df) == 0 or len(base_comm) == 0:
            continue
        surv_jcol = _join_col(surv_df)
        base_jcol = _join_col(base_comm)
        t_stat, p_val = stats.ttest_ind(
            surv_df[surv_jcol].dropna(), base_comm[base_jcol].dropna()
        )
        results[SHORT.get(model_slug, model_slug)] = {
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "surv_mean": round(float(surv_df[surv_jcol].mean()), 4),
            "base_mean": round(float(base_comm[base_jcol].mean()), 4),
            "surv_n": int(len(surv_df)),
            "base_n": int(len(base_comm)),
        }
    return results


# ═══════════════════════════════════════════════════════════════════
# A8: PROPAGANDA CROSS-CHECK (CSV vs JSON)
# ═══════════════════════════════════════════════════════════════════

def compute_propaganda_crosscheck() -> dict:
    """Cross-check propaganda real-citizen computation: CSV vs JSON (A8)."""
    results = {}
    for k in [2, 5, 10]:
        prop_base = ROOT / f"propaganda-k{k}"
        for f in _find_summaries(prop_base, "experiment_comm_summary.csv"):
            model_slug = _model_slug_from_summary_path(f, prop_base)
            if model_slug is None:
                continue
            csv_df = _load_summary(f)
            if len(csv_df) == 0:
                continue
            # CSV method: n_join / (25 - k)
            if "n_join" in csv_df.columns:
                csv_real = (csv_df["n_join"] / (25 - k)).mean()
            else:
                continue
            # JSON method: exclude is_propaganda agents
            log_path = f.parent / "experiment_comm_log.json"
            logs = _load_experiment_log(log_path)
            if not logs:
                continue
            json_real = float(_real_join_from_comm_log(logs).mean())

            divergence = abs(csv_real - json_real)
            results[f"k={k}_{SHORT.get(model_slug, model_slug)}"] = {
                "csv_real_mean": round(float(csv_real), 4),
                "json_real_mean": round(float(json_real), 4),
                "divergence": round(float(divergence), 4),
                "warning": divergence > 0.01,
            }
    return results


# ═══════════════════════════════════════════════════════════════════
# A9: TEXT BASELINE LOGISTIC SLOPE
# ═══════════════════════════════════════════════════════════════════

def _logistic_func(x, b0, b1):
    return 1.0 / (1.0 + np.exp(b0 + b1 * x))


def compute_text_baseline() -> dict:
    """Compute naive text predictor (1-direction) logistic slope (A9)."""
    results = {}
    for model in PART1_MODELS:
        log_path = ROOT / model / "experiment_pure_log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            data = json.load(f)

        directions, decisions = [], []
        for period in data:
            for agent in period["agents"]:
                if agent.get("api_error"):
                    continue
                d = agent.get("direction")
                if d is None or (isinstance(d, float) and np.isnan(d)):
                    continue
                dec = 1 if agent.get("decision") == "JOIN" else 0
                directions.append(d)
                decisions.append(dec)

        if len(directions) < 50:
            continue

        dirs = np.array(directions)
        decs = np.array(decisions)
        naive = 1.0 - dirs  # naive predictor: low direction → high join

        # Fit logistic: P(JOIN) = 1/(1+exp(b0+b1*naive))
        try:
            popt, _ = curve_fit(_logistic_func, naive, decs, p0=[0, 2], maxfev=10000)
            b0, b1 = popt
            # Also fit to the LLM behavioral sigmoid on theta
            r_naive = pearson_with_ci(naive, decs)
            results[SHORT.get(model, model)] = {
                "logistic_slope": round(float(b1), 4),
                "logistic_intercept": round(float(b0), 4),
                "r_naive_vs_decision": r_naive,
                "n": len(dirs),
            }
        except RuntimeError:
            pass

    return results


# ═══════════════════════════════════════════════════════════════════
# A10: PAPER CLAIMS VALIDATOR
# ═══════════════════════════════════════════════════════════════════

def validate_paper_claims(all_stats: dict) -> list[dict]:
    """Parse key numeric claims from paper.tex and check vs verified_stats (A10)."""
    paper_path = PROJECT_ROOT / "paper" / "paper.tex"
    if not paper_path.exists():
        return [{"issue": "paper.tex not found"}]

    tex = paper_path.read_text()
    discrepancies = []

    def _check(claim_label: str, paper_val: float, computed_val: float, tol: float = 0.02):
        if computed_val is None or (isinstance(computed_val, float) and np.isnan(computed_val)):
            discrepancies.append({
                "claim": claim_label,
                "paper_value": paper_val,
                "computed_value": None,
                "status": "UNVERIFIABLE",
            })
            return
        diff = abs(paper_val - computed_val)
        status = "OK" if diff <= tol else "MISMATCH"
        if status == "MISMATCH":
            discrepancies.append({
                "claim": claim_label,
                "paper_value": round(paper_val, 4),
                "computed_value": round(computed_val, 4),
                "diff": round(diff, 4),
                "status": status,
            })

    # Extract known claims via regex
    part1 = all_stats.get("part1", {})

    # Mean |r| across models
    m = re.search(r"mean.*\|r\|.*?=\s*([0-9.]+)", tex, re.IGNORECASE)
    if m:
        paper_mean_r = float(m.group(1))
        computed = part1.get("_mean_abs_r_pure_vs_theta")
        _check("mean |r| across models", paper_mean_r, computed)

    # Pooled r values (look for patterns like r = -0.XXX or $r = -0.XXX$)
    pooled_pure = part1.get("_pooled_pure", {})
    pooled_r = (pooled_pure.get("r_vs_attack") or {}).get("r")

    # Belief stats
    beliefs = all_stats.get("beliefs", {})
    pure_beliefs = beliefs.get("pure", {})
    if isinstance(pure_beliefs, dict) and "r_posterior_belief" in pure_beliefs:
        r_post = pure_beliefs["r_posterior_belief"].get("r") if isinstance(pure_beliefs["r_posterior_belief"], dict) else None
        # Look for r(posterior, belief) pattern in paper
        m = re.search(r"r\(posterior.*?belief\).*?([+-]?[0-9.]+)", tex, re.IGNORECASE)
        if m and r_post is not None:
            _check("r(posterior, belief)", float(m.group(1)), r_post)

    # Surveillance t-test — check against primary model (Mistral Small Creative)
    surv_tests = all_stats.get("surveillance_ttest", {})
    surv_matches = re.finditer(r"(?:surveillance|chilling).*?t\s*=\s*([+-]?[0-9.]+)", tex, re.IGNORECASE)
    primary_surv = surv_tests.get("Mistral Small Creative", {})
    for match in surv_matches:
        paper_t = float(match.group(1))
        computed_t = primary_surv.get("t_stat")
        if computed_t is not None:
            _check("surveillance t-test (Mistral Small Creative)", paper_t, computed_t, tol=0.1)
            break

    return discrepancies


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

    print("Running pooled OLS...")
    ols = pooled_ols(None)

    print("Computing belief analysis (A1)...")
    beliefs = compute_beliefs()

    print("Computing message content analysis (A2-A3, A5)...")
    msg_content = compute_message_content()

    print("Computing surveillance t-tests (A4)...")
    surv_ttest = compute_surveillance_ttest()

    print("Computing propaganda cross-check (A8)...")
    prop_crosscheck = compute_propaganda_crosscheck()

    print("Computing text baseline (A9)...")
    text_baseline = compute_text_baseline()

    all_stats = {
        "part1": part1,
        "infodesign": infodesign,
        "regime_control": regime,
        "robustness": robust,
        "pooled_ols": ols,
        "beliefs": beliefs,
        "message_content": msg_content,
        "surveillance_ttest": surv_ttest,
        "propaganda_crosscheck": prop_crosscheck,
        "text_baseline": text_baseline,
    }

    print("\nRunning discrepancy analysis...")
    discrepancies = discrepancy_report(all_stats)
    all_stats["discrepancy_report"] = discrepancies

    print("Validating paper claims (A10)...")
    claim_issues = validate_paper_claims(all_stats)
    all_stats["paper_claim_validation"] = claim_issues

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

    # Belief summary
    print("\n  BELIEFS:")
    for treatment in ["pure", "surveillance", "comm", "propaganda_k5"]:
        b = beliefs.get(treatment, {})
        if isinstance(b, dict) and "n" in b:
            r_post = (b.get("r_posterior_belief") or {}).get("r", "?")
            r_bel = (b.get("r_belief_decision") or {}).get("r", "?")
            print(f"    {treatment}: n={b['n']}, r(post,bel)={r_post}, r(bel,dec)={r_bel}")
    cross = beliefs.get("_cross_pure_vs_surv", {})
    if cross:
        print(f"    Pure→Surv: belief Δ={cross.get('belief_shift')}, action Δ={cross.get('action_shift')}")

    # Surveillance t-tests
    print("\n  SURVEILLANCE T-TESTS:")
    for model, t in surv_ttest.items():
        print(f"    {model}: t={t['t_stat']}, p={t['p_value']}")

    # Propaganda cross-check warnings
    warnings = [k for k, v in prop_crosscheck.items() if v.get("warning")]
    if warnings:
        print(f"\n  WARNING: Propaganda cross-check divergences > 0.01: {warnings}")
    else:
        print(f"\n  Propaganda cross-check: all consistent (N={len(prop_crosscheck)})")

    # Text baseline
    print("\n  TEXT BASELINE (logistic slope of 1-direction predictor):")
    for model, tb in text_baseline.items():
        print(f"    {model}: slope={tb['logistic_slope']}")

    # Paper claim validation
    if claim_issues:
        print(f"\n  PAPER CLAIM VALIDATION: {len(claim_issues)} issue(s)")
        for issue in claim_issues:
            print(f"    {issue.get('status', '?')}: {issue.get('claim', '?')} "
                  f"(paper={issue.get('paper_value')}, computed={issue.get('computed_value')})")
    else:
        print("\n  PAPER CLAIM VALIDATION: all checked claims OK")

    # Save
    with open(OUT, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nFull results saved to {OUT}")


if __name__ == "__main__":
    main()
