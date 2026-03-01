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

from models import PART1_SLUGS, DISPLAY_NAMES, PRIMARY_SLUG

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "output"
OUT = Path(__file__).resolve().parent / "verified_stats.json"

# ── Models with full Part I data ──────────────────────────────────────

PART1_MODELS = PART1_SLUGS

SHORT = DISPLAY_NAMES

PRIMARY = PRIMARY_SLUG

# Part I benchmark for A(theta): canonical Morris-Shin parameters used in plots/tables.
# We compute A(theta) directly from theta rather than relying on the logged
# `theoretical_attack` column, since Part I payoffs are not shown to agents.
PART1_BENCHMARK_THETA_STAR = 0.50  # B=C=1
PART1_BENCHMARK_SIGMA = 0.30


def _attack_mass_benchmark(theta: np.ndarray) -> np.ndarray:
    """Benchmark attack mass A(theta) with fixed (theta*, sigma)."""
    theta = np.asarray(theta, dtype=float)
    ts = float(PART1_BENCHMARK_THETA_STAR)
    sigma = float(PART1_BENCHMARK_SIGMA)
    ts = float(np.clip(ts, 1e-8, 1 - 1e-8))
    x_star = ts + sigma * stats.norm.ppf(ts)
    return stats.norm.cdf((x_star - theta) / sigma)


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


def within_country_pearson(df: pd.DataFrame, xcol: str, ycol: str,
                           group_col: str = "country", alpha: float = 0.05):
    """Pearson r on country-demeaned values (removes between-country variation).

    For the scramble treatment the cross-period permutation creates an
    ecological confound: countries with systematically different theta
    distributions produce spurious pooled correlations.  Demeaning by
    country removes this confound and isolates the within-country
    signal-to-outcome link that the falsification test is meant to assess.
    """
    tmp = df[[group_col, xcol, ycol]].dropna().copy()
    if len(tmp) < 3:
        return {"r": float("nan"), "p": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": len(tmp)}
    # Demean within each country
    for col in [xcol, ycol]:
        tmp[col] = tmp.groupby(group_col)[col].transform(lambda s: s - s.mean())
    return pearson_with_ci(tmp[xcol].values, tmp[ycol].values, alpha=alpha)


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
    """Infer model slug from the CSV's parent directory.

    For flat layouts (base/slug/file.csv) and nested layouts created by
    bash slug issues (base/vendor/model/slug/file.csv), the actual slug
    is always the immediate parent of the CSV.
    """
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    # The CSV's immediate parent is the slug (works for both flat and nested)
    slug = parts[-2]
    # Validate: slugs always contain '--' (vendor--model)
    if "--" in slug:
        return slug
    # Fallback for flat layouts where parts[0] is the slug
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

            # Correlation with benchmark attack mass A(theta) under fixed (theta*, sigma).
            attack = _attack_mass_benchmark(theta)
            r_attack = pearson_with_ci(attack, jf)
            mask = np.isfinite(attack) & np.isfinite(jf)
            rmse = float(np.sqrt(np.mean((jf[mask] - attack[mask]) ** 2))) if mask.any() else float("nan")
            mae = float(np.mean(np.abs(jf[mask] - attack[mask]))) if mask.any() else float("nan")

            entry = {
                "r_vs_theta": r_theta,
                "r_vs_attack": r_attack,
                "n_obs": len(df),
                "mean_join": round(_safe_mean(jf), 4),
                "std_join": round(_safe_std(jf), 4),
                "mean_theta": round(_safe_mean(theta), 4),
                "rmse_vs_attack": round(rmse, 4) if np.isfinite(rmse) else None,
                "mae_vs_attack": round(mae, 4) if np.isfinite(mae) else None,
            }

            # For scramble: use within-country (country-demeaned) correlation
            # as the primary r, to remove the ecological confound created by
            # cross-period permutation within countries.
            if treatment == "scramble" and "country" in df.columns:
                entry["r_vs_theta_raw"] = r_theta
                entry["r_vs_attack_raw"] = r_attack
                entry["r_vs_theta"] = within_country_pearson(df, "theta", jcol)
                tmp = df.copy()
                tmp["_attack_benchmark"] = _attack_mass_benchmark(tmp["theta"].astype(float).values)
                entry["r_vs_attack"] = within_country_pearson(tmp, "_attack_benchmark", jcol)

            m[treatment] = entry

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
        theta = pooled["theta"].astype(float).values
        attack = _attack_mass_benchmark(theta)
        jf = pooled[jcol].astype(float).values
        out = {
            "r_vs_theta": pearson_with_ci(theta, jf),
            "r_vs_attack": pearson_with_ci(attack, jf),
            "n_obs": int(len(pooled)),
            "mean_join": round(_safe_mean(jf), 4),
        }
        mask = np.isfinite(attack) & np.isfinite(jf)
        if mask.any():
            out["rmse_vs_attack"] = round(float(np.sqrt(np.mean((jf[mask] - attack[mask]) ** 2))), 4)
            out["mae_vs_attack"] = round(float(np.mean(np.abs(jf[mask] - attack[mask]))), 4)
        if n_agents is not None:
            out["n_agents"] = int(n_agents)
        return out

    results["_pooled_pure"] = _pooled_entry(all_pure) if all_pure else {}
    results["_pooled_comm"] = _pooled_entry(all_comm) if all_comm else {}
    results["_pooled_flip"] = _pooled_entry(all_flip) if all_flip else {}

    # Pooled scramble: use within-country demeaned correlation
    if all_scramble:
        pooled_scr = pd.concat(all_scramble, ignore_index=True)
        jcol_scr = _join_col(pooled_scr)
        raw_entry = _pooled_entry(all_scramble)

        # Create a model-country group key for demeaning across pooled data
        # (each model's country indices are independent)
        if "country" in pooled_scr.columns:
            # Build a unique group id per (source model, country) combination.
            # Each df in all_scramble came from a different model; tag rows.
            parts = []
            for i, sdf in enumerate(all_scramble):
                tmp = sdf.copy()
                tmp["_model_idx"] = i
                parts.append(tmp)
            tagged = pd.concat(parts, ignore_index=True)
            tagged["_group"] = tagged["_model_idx"].astype(str) + "_" + tagged["country"].astype(str)

            raw_entry["r_vs_theta_raw"] = raw_entry["r_vs_theta"]
            raw_entry["r_vs_attack_raw"] = raw_entry.get("r_vs_attack")
            raw_entry["r_vs_theta"] = within_country_pearson(
                tagged, "theta", jcol_scr, group_col="_group"
            )
            tagged["_attack_benchmark"] = _attack_mass_benchmark(
                tagged["theta"].astype(float).values
            )
            raw_entry["r_vs_attack"] = within_country_pearson(
                tagged, "_attack_benchmark", jcol_scr, group_col="_group"
            )

        results["_pooled_scramble"] = raw_entry
    else:
        results["_pooled_scramble"] = {}

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
    model = PRIMARY
    model_dir = ROOT / model
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
        entry = {
            "mean_join": round(sub[jcol].mean(), 4),
            "r_vs_theta": r_theta,
            "r_vs_attack": r_attack,
            "n_obs": len(sub),
        }

        # For scramble designs: demean by rep to remove ecological confound,
        # mirroring the within-country demeaning used for Part I scramble.
        if "scramble" in design and "rep" in sub.columns:
            entry["r_vs_theta_raw"] = r_theta
            entry["r_vs_theta"] = within_country_pearson(
                sub, "theta", jcol, group_col="rep"
            )
            if "theoretical_attack" in sub.columns:
                entry["r_vs_attack_raw"] = r_attack
                entry["r_vs_attack"] = within_country_pearson(
                    sub, "theoretical_attack", jcol, group_col="rep"
                )

        results[design] = entry

    # ── Baseline source of truth (prevents accidental overwrite) ─────
    # The B/C sweep runner repeatedly writes `experiment_infodesign_baseline_summary.csv`
    # as it iterates over θ* targets, so the on-disk "baseline" file may end up
    # corresponding to the *last* θ* in the sweep (typically 0.75), not the
    # primary infodesign baseline (θ* = 0.50 on θ ∈ [0.20, 0.80]).
    #
    # When available, prefer the θ*=0.50 slice from `experiment_bc_sweep_summary.csv`
    # to compute baseline stats and logistic cutoffs reported in Part II and the
    # B/C narrative table.
    bc_sweep_baseline = None
    bc_sweep_path = model_dir / "experiment_bc_sweep_summary.csv"
    if bc_sweep_path.exists():
        try:
            bc = pd.read_csv(bc_sweep_path)
            if "theta_star_target" in bc.columns:
                bc_sweep_baseline = bc[np.isclose(bc["theta_star_target"].astype(float), 0.50)]
                if len(bc_sweep_baseline) == 0:
                    bc_sweep_baseline = None
        except Exception:
            bc_sweep_baseline = None

    # Also load individual per-design CSVs not in all_summary
    import glob
    for csv_path in sorted(model_dir.glob("experiment_infodesign_*_summary.csv")):
        fname = csv_path.name
        # Extract design name: experiment_infodesign_{design}_summary.csv
        prefix = "experiment_infodesign_"
        suffix = "_summary.csv"
        if not fname.startswith(prefix) or not fname.endswith(suffix):
            continue
        design = fname[len(prefix):-len(suffix)]
        if design == "all" or design in results:
            continue
        df_d = pd.read_csv(csv_path)
        if len(df_d) == 0:
            continue
        jc = "join_fraction_valid" if "join_fraction_valid" in df_d.columns else "join_fraction"
        r_theta = pearson_with_ci(df_d["theta"], df_d[jc])
        r_attack = pearson_with_ci(df_d["theoretical_attack"], df_d[jc]) \
            if "theoretical_attack" in df_d.columns else {}
        entry = {
            "mean_join": round(df_d[jc].mean(), 4),
            "r_vs_theta": r_theta,
            "r_vs_attack": r_attack,
            "n_obs": len(df_d),
        }

        # Scramble demeaning for individual per-design CSVs
        if "scramble" in design and "rep" in df_d.columns:
            entry["r_vs_theta_raw"] = r_theta
            entry["r_vs_theta"] = within_country_pearson(
                df_d, "theta", jc, group_col="rep"
            )
            if "theoretical_attack" in df_d.columns:
                entry["r_vs_attack_raw"] = r_attack
                entry["r_vs_attack"] = within_country_pearson(
                    df_d, "theoretical_attack", jc, group_col="rep"
                )

        results[design] = entry

    # Override baseline stats if we have the canonical θ*=0.50 slice from the sweep.
    if bc_sweep_baseline is not None:
        jc = _join_col(bc_sweep_baseline)
        r_theta = pearson_with_ci(bc_sweep_baseline["theta"], bc_sweep_baseline[jc])
        r_attack = pearson_with_ci(bc_sweep_baseline["theoretical_attack"], bc_sweep_baseline[jc]) \
            if "theoretical_attack" in bc_sweep_baseline.columns else {}
        results["baseline"] = {
            "mean_join": round(bc_sweep_baseline[jc].mean(), 4),
            "r_vs_theta": r_theta,
            "r_vs_attack": r_attack,
            "n_obs": int(len(bc_sweep_baseline)),
            "_source": "experiment_bc_sweep_summary.csv[theta_star_target=0.50]",
        }

    # Treatment effects relative to baseline
    if "baseline" in results:
        baseline_mean = results["baseline"]["mean_join"]
        for design in results:
            if design != "baseline" and not design.startswith("_"):
                delta = results[design]["mean_join"] - baseline_mean
                results[design]["delta_vs_baseline"] = round(delta, 4)

    # B/C narrative comparative statics: include logistic cutoff estimates
    # so the paper can report theory-predicted cutoff shifts without
    # manual copy/paste.
    for dname in ["baseline", "bc_high_cost", "bc_low_cost"]:
        if dname not in results:
            continue
        p = model_dir / f"experiment_infodesign_{dname}_summary.csv"
        if dname != "baseline" and not p.exists():
            continue
        if dname == "baseline" and bc_sweep_baseline is not None:
            df = bc_sweep_baseline
        else:
            if not p.exists():
                continue
            df = pd.read_csv(p)
        if len(df) == 0:
            continue
        jc = _join_col(df)
        fit = _fit_logistic(df["theta"].astype(float).values, df[jc].astype(float).values)
        if fit is not None:
            results[dname]["logistic_fit"] = fit

    # Cross-model infodesign replication
    cross = {}
    for m in PART1_MODELS:
        name = SHORT[m]
        cross[name] = {}
        for design in ["baseline", "scramble", "flip"]:
            # Primary-model baseline is sourced from the canonical θ*=0.50 sweep slice when available.
            if m == model and design == "baseline" and bc_sweep_baseline is not None:
                df_d = bc_sweep_baseline
            else:
                df_d = load_infodesign(m, design)
            if len(df_d) == 0:
                continue
            jc = _join_col(df_d)
            r_t = pearson_with_ci(df_d["theta"], df_d[jc])
            cross[name][design] = {
                "mean_join": round(df_d[jc].mean(), 4),
                "r_vs_theta": r_t,
                "n_obs": len(df_d),
            }
    results["_cross_model"] = cross

    # Diagnostics: scramble should collapse r(θ, join) under correct cross-θ permutation.
    # Flag models where it does not, so stale/buggy runs are caught during verification.
    scramble_fail = []
    for model_name, dct in cross.items():
        scr = dct.get("scramble")
        r_scr = ((scr or {}).get("r_vs_theta") or {}).get("r") if isinstance(scr, dict) else None
        if r_scr is not None and not np.isnan(r_scr) and abs(float(r_scr)) > 0.30:
            scramble_fail.append(model_name)
    results["_infodesign_scramble_not_collapsed_models"] = scramble_fail

    # Slider-independence diagnostic for scramble designs: verify that θ is
    # uncorrelated with each slider (direction, clarity, coordination) under
    # scramble, confirming that the permutation severs the θ→slider link.
    for design_key in sorted(k for k in results if isinstance(k, str) and "scramble" in k and not k.startswith("_")):
        log_path = model_dir / f"experiment_infodesign_{design_key}_log.json"
        if not log_path.exists():
            # Fall back to the combined log if individual log is missing
            log_path = model_dir / "experiment_infodesign_all_log.json"
        if not log_path.exists():
            continue
        log_data = _load_experiment_log(log_path)
        if not log_data:
            continue
        # Filter to scramble design entries
        scramble_periods = [p for p in log_data if p.get("design") == design_key]
        if not scramble_periods:
            continue
        slider_rows = []
        for p in scramble_periods:
            theta_val = p["theta"]
            for a in p.get("agents", []):
                if a.get("api_error"):
                    continue
                row = {"theta": theta_val}
                for slider in ["direction", "clarity", "coordination"]:
                    if slider in a:
                        row[slider] = a[slider]
                if len(row) > 1:
                    slider_rows.append(row)
        if slider_rows:
            sdf = pd.DataFrame(slider_rows)
            slider_indep = {}
            for slider in ["direction", "clarity", "coordination"]:
                if slider in sdf.columns:
                    r_info = pearson_with_ci(sdf["theta"].values, sdf[slider].values)
                    slider_indep[slider] = r_info
            if slider_indep:
                results[design_key]["slider_independence"] = slider_indep

    # Sanity: all primary-model designs should share the same θ grid.
    model_dir = ROOT / model

    def _theta_grid_for(design: str) -> list[float] | None:
        if design == "baseline" and bc_sweep_baseline is not None:
            vals = bc_sweep_baseline["theta"].astype(float).round(12).unique().tolist()
            return sorted(float(v) for v in vals)
        p = model_dir / f"experiment_infodesign_{design}_summary.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if "theta" not in df.columns:
            return None
        vals = df["theta"].astype(float).round(12).unique().tolist()
        return sorted(float(v) for v in vals)

    base_grid = _theta_grid_for("baseline")
    # Sanity-check θ-grid consistency.  Designs with different B/C
    # parameters use shifted grids by construction; collect all grids
    # and warn (rather than crash) when they differ from the majority.
    all_grids: dict[str, list[float]] = {}
    for design in sorted(k for k in results.keys() if isinstance(k, str) and not k.startswith("_")):
        g = _theta_grid_for(design)
        if g is not None:
            all_grids[design] = g
    if all_grids:
        from collections import Counter
        grid_counts = Counter(tuple(g) for g in all_grids.values())
        majority_grid = list(grid_counts.most_common(1)[0][0])
        for design, g in all_grids.items():
            if g != majority_grid:
                import warnings
                warnings.warn(
                    f"Infodesign θ-grid for '{design}' differs from majority: "
                    f"{g} vs {majority_grid}"
                )

    return results


# ═══════════════════════════════════════════════════════════════════
# Information design WITH communication (primary-model robustness)
# ═══════════════════════════════════════════════════════════════════

def compute_infodesign_comm():
    """Compute infodesign-grid stats under communication (no surveillance).

    This is used for the surveillance×censorship interaction table so the
    "No Surv." column is on the same (communication) treatment as the "Surv."
    column, rather than mixing pure and comm samples.
    """
    results = {}

    primary = PRIMARY
    base_dir = ROOT / f"{primary}-infodesign-comm" / primary
    df_all = _load_summary(base_dir / "experiment_infodesign_all_summary.csv")
    if len(df_all) == 0:
        print("  WARNING: no infodesign-comm data for primary model")
        return results

    # Sanity: expect comm treatment
    if "treatment" in df_all.columns:
        treatments = sorted(set(df_all["treatment"].dropna().unique().tolist()))
        if treatments and treatments != ["comm"]:
            raise AssertionError(
                f"Expected only treatment='comm' in {base_dir}, got: {treatments}"
            )

    jcol = _join_col(df_all)
    for design in sorted(df_all["design"].dropna().unique().tolist()):
        sub = df_all[df_all["design"] == design]
        results[design] = {
            "mean_join": round(_safe_mean(sub[jcol]), 4),
            "r_vs_theta": pearson_with_ci(sub["theta"], sub[jcol]),
            "n_obs": int(len(sub)),
        }

    # Deltas vs comm baseline
    if "baseline" in results:
        base_mean = results["baseline"]["mean_join"]
        for design, d in results.items():
            if design == "baseline":
                continue
            d["delta_vs_baseline"] = round(d["mean_join"] - base_mean, 4)

    # Sanity: all comm designs should share the same θ grid.
    def _theta_grid_for(design: str) -> list[float] | None:
        p = base_dir / f"experiment_infodesign_{design}_summary.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if "theta" not in df.columns:
            return None
        vals = df["theta"].astype(float).round(12).unique().tolist()
        return sorted(float(v) for v in vals)

    base_grid = _theta_grid_for("baseline")
    if base_grid:
        for design in sorted(results.keys()):
            g = _theta_grid_for(design)
            if g is not None and g != base_grid:
                raise AssertionError(
                    f"Infodesign-comm θ-grid mismatch for '{design}': "
                    f"{g} vs baseline {base_grid}"
                )

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

    # Propaganda saturation test: k=5 vs k=10 real-citizen join rates
    prop = results.get("propaganda", {})
    k5_data = prop.get("k=5", {}).get("Mistral Small Creative", {})
    k10_data = prop.get("k=10", {}).get("Mistral Small Creative", {})
    if k5_data.get("mean_join_real") is not None and k10_data.get("mean_join_real") is not None:
        # Load agent-level data for proper test
        k5_log = _load_experiment_log(
            ROOT / "propaganda-k5" / PRIMARY / "experiment_comm_log.json"
        )
        k10_log = _load_experiment_log(
            ROOT / "propaganda-k10" / PRIMARY / "experiment_comm_log.json"
        )
        k5_decisions, k10_decisions = [], []
        for log in (k5_log or []):
            for a in log.get("agents", []):
                if not a.get("is_propaganda", False) and not a.get("api_error"):
                    k5_decisions.append(1 if a.get("decision") == "JOIN" else 0)
        for log in (k10_log or []):
            for a in log.get("agents", []):
                if not a.get("is_propaganda", False) and not a.get("api_error"):
                    k10_decisions.append(1 if a.get("decision") == "JOIN" else 0)
        if k5_decisions and k10_decisions:
            k5_arr = np.array(k5_decisions)
            k10_arr = np.array(k10_decisions)
            t_stat, t_p = stats.ttest_ind(k5_arr, k10_arr)
            results["_propaganda_saturation_k5_k10"] = {
                "k5_mean_real": round(float(k5_arr.mean()), 4),
                "k10_mean_real": round(float(k10_arr.mean()), 4),
                "delta_pp": round(float((k10_arr.mean() - k5_arr.mean()) * 100), 2),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(t_p), 6),
                "n_k5": len(k5_decisions),
                "n_k10": len(k10_decisions),
            }

    return results


# ═══════════════════════════════════════════════════════════════════
# ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════

def compute_robustness():
    """Statistics for robustness checks."""
    results = {}

    # Agent count variations
    for n in [5, 10, 50, 100]:
        d = ROOT / f"{PRIMARY}-n{n}"
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


# ═══════════════════════════════════════════════════════════════════
# LOGISTIC FITS: cutoff + slope per model × treatment
# ═══════════════════════════════════════════════════════════════════

def _logistic(x, b0, b1):
    return 1.0 / (1.0 + np.exp(b0 + b1 * x))


def _fit_logistic(x, y):
    """Fit logistic and return params dict, or None on failure."""
    from scipy.optimize import curve_fit
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return None
    try:
        popt, pcov = curve_fit(_logistic, x, y, p0=[0, 2], maxfev=10000)
        b0, b1 = popt
        se_b0 = np.sqrt(pcov[0, 0]) if pcov[0, 0] >= 0 else float("nan")
        se_b1 = np.sqrt(pcov[1, 1]) if pcov[1, 1] >= 0 else float("nan")
        cutoff = -b0 / b1 if abs(b1) > 1e-8 else float("nan")
        # Delta method SE for cutoff: Var(-b0/b1) ≈ (1/b1)^2 Var(b0) + (b0/b1^2)^2 Var(b1)
        # - 2*(b0/b1^3) Cov(b0,b1)
        if abs(b1) > 1e-8:
            grad = np.array([-1.0 / b1, b0 / b1**2])
            se_cutoff = float(np.sqrt(grad @ pcov @ grad))
        else:
            se_cutoff = float("nan")
        return {
            "b0": round(float(b0), 4),
            "b1": round(float(b1), 4),
            "se_b0": round(float(se_b0), 4),
            "se_b1": round(float(se_b1), 4),
            "cutoff": round(float(cutoff), 4),
            "se_cutoff": round(float(se_cutoff), 4),
            "n": int(len(x)),
        }
    except RuntimeError:
        return None


def compute_logistic_fits():
    """Fit logistic per model × treatment, compute cutoff and slope."""
    results = {}
    for model in PART1_MODELS:
        name = SHORT[model]
        m = {}
        for treatment in ["pure", "comm", "scramble", "flip"]:
            df = load(model, treatment)
            if len(df) == 0:
                continue
            jcol = _join_col(df)
            x = df["theta"].astype(float).values
            y = df[jcol].astype(float).values
            fit = _fit_logistic(x, y)
            if fit is not None:
                m[treatment] = fit
        if m:
            results[name] = m
    return results


# ═══════════════════════════════════════════════════════════════════
# CLUSTERED STANDARD ERRORS for pooled OLS
# ═══════════════════════════════════════════════════════════════════

def compute_clustered_ses():
    """Clustered SEs for pooled OLS: join = b0 + b1 * A(theta), clustered by country and model."""
    all_dfs = []
    for model in PART1_MODELS:
        df = load(model, "pure")
        if len(df) > 0:
            df = df.copy()
            df["model"] = model
            all_dfs.append(df)
    if not all_dfs:
        return {}
    pooled = pd.concat(all_dfs, ignore_index=True)
    jcol = "join_fraction_valid" if "join_fraction_valid" in pooled.columns else "join_fraction"
    y = pooled[jcol].values
    x_var = pooled["theoretical_attack"].values if "theoretical_attack" in pooled.columns else None
    if x_var is None:
        return {}

    X = np.column_stack([np.ones_like(x_var), x_var])
    n = len(y)
    k = X.shape[1]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    # Bread: (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)

    results = {}

    # Homoskedastic SE for reference
    s2 = np.sum(resid**2) / (n - k)
    se_homo = np.sqrt(np.diag(s2 * XtX_inv))
    results["homoskedastic"] = {
        "se_intercept": round(float(se_homo[0]), 6),
        "se_slope": round(float(se_homo[1]), 6),
    }

    # HC1 (heteroskedasticity-robust)
    meat_hc1 = sum(resid[i]**2 * np.outer(X[i], X[i]) for i in range(n))
    V_hc1 = (n / (n - k)) * XtX_inv @ meat_hc1 @ XtX_inv
    se_hc1 = np.sqrt(np.diag(V_hc1))
    results["hc1"] = {
        "se_intercept": round(float(se_hc1[0]), 6),
        "se_slope": round(float(se_hc1[1]), 6),
    }

    # Cluster by country
    if "country" in pooled.columns:
        clusters = pooled.groupby("country").indices
        G = len(clusters)
        meat = np.zeros((k, k))
        for _, idx in clusters.items():
            u_g = X[idx].T @ resid[idx]
            meat += np.outer(u_g, u_g)
        V_cl = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX_inv @ meat @ XtX_inv
        se_cl = np.sqrt(np.diag(V_cl))
        results["clustered_country"] = {
            "se_intercept": round(float(se_cl[0]), 6),
            "se_slope": round(float(se_cl[1]), 6),
            "n_clusters": int(G),
        }

    # Cluster by model
    clusters_m = pooled.groupby("model").indices
    G_m = len(clusters_m)
    meat_m = np.zeros((k, k))
    for _, idx in clusters_m.items():
        u_g = X[idx].T @ resid[idx]
        meat_m += np.outer(u_g, u_g)
    V_clm = (G_m / (G_m - 1)) * ((n - 1) / (n - k)) * XtX_inv @ meat_m @ XtX_inv
    se_clm = np.sqrt(np.diag(V_clm))
    results["clustered_model"] = {
        "se_intercept": round(float(se_clm[0]), 6),
        "se_slope": round(float(se_clm[1]), 6),
        "n_clusters": int(G_m),
    }

    return results


# ═══════════════════════════════════════════════════════════════════
# PLACEBO / ANONYMOUS SURVEILLANCE
# ═══════════════════════════════════════════════════════════════════

def compute_temperature_robustness():
    """Compute statistics for temperature robustness experiments."""
    results = {}
    primary = PRIMARY

    for temp in ["0.3", "0.7", "1.0"]:
        temp_dir = ROOT.parent / f"output/temperature-robustness-T{temp}" / primary
        csv_path = temp_dir / "experiment_pure_summary.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values
        theta = df["theta"].astype(float).values
        r_theta = pearson_with_ci(theta, jf)
        attack = df["theoretical_attack"].astype(float).values if "theoretical_attack" in df.columns else None
        r_attack = pearson_with_ci(attack, jf) if attack is not None else {}
        fit = _fit_logistic(theta, jf)
        results[f"T={temp}"] = {
            "n_obs": len(df),
            "mean_join": round(float(np.nanmean(jf)), 4),
            "r_vs_theta": r_theta,
            "r_vs_attack": r_attack,
            "logistic_fit": fit,
        }

    return results


def compute_surveillance_variants():
    """Compute statistics for placebo and anonymous surveillance variants."""
    results = {}
    primary = PRIMARY

    variants = {
        "placebo": ROOT / primary / "_surveillance_placebo_v2" / primary / "experiment_comm_summary.csv",
        "anonymous": ROOT / primary / "_surveillance_anonymous_v2" / primary / "experiment_comm_summary.csv",
    }

    # Load comm baseline for comparison
    comm_df = load(primary, "comm")
    comm_jcol = _join_col(comm_df) if len(comm_df) > 0 else "join_fraction"
    comm_mean = float(comm_df[comm_jcol].mean()) if len(comm_df) > 0 else float("nan")

    for variant_name, path in variants.items():
        if not path.exists():
            results[variant_name] = {"status": "missing", "path": str(path)}
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            results[variant_name] = {"status": "empty"}
            continue
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values
        theta = df["theta"].astype(float).values

        r_theta = pearson_with_ci(theta, jf)
        mean_join = round(float(np.nanmean(jf)), 4)
        delta_vs_comm = round((mean_join - comm_mean) * 100, 2) if np.isfinite(comm_mean) else None

        # t-test vs comm baseline
        if len(comm_df) > 0:
            comm_jf = comm_df[comm_jcol].astype(float).values
            t_stat, t_p = stats.ttest_ind(jf, comm_jf)
            t_test = {"t_stat": round(float(t_stat), 4), "p_value": round(float(t_p), 6)}
        else:
            t_test = None

        results[variant_name] = {
            "n_obs": len(df),
            "mean_join": mean_join,
            "r_vs_theta": r_theta,
            "delta_vs_comm_pp": delta_vs_comm,
            "t_test_vs_comm": t_test,
        }

    return results


def compute_uncalibrated():
    """Statistics for uncalibrated robustness runs (Section D)."""
    results = {}
    uncal_base = ROOT / "uncalibrated-robustness"
    uncal_models = [
        PRIMARY,
        "meta-llama--llama-3.3-70b-instruct",
        "qwen--qwen3-235b-a22b-2507",
    ]
    for model_slug in uncal_models:
        p = uncal_base / model_slug / "experiment_pure_summary.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values
        theta = df["theta"].astype(float).values
        r_theta = pearson_with_ci(theta, jf)
        # Regime fall rate: coup_success column if present, else theta < theta_star
        if "coup_success" in df.columns:
            fall_rate = round(float(df["coup_success"].mean()), 4)
        elif "theta_star" in df.columns:
            fall_rate = round(float((df["theta"] < df["theta_star"]).mean()), 4)
        else:
            fall_rate = None
        name = SHORT.get(model_slug, model_slug)
        results[name] = {
            "n_obs": int(len(df)),
            "mean_join": round(_safe_mean(jf), 4),
            "std_join": round(_safe_std(jf), 4),
            "r_vs_theta": r_theta,
            "regime_fall_rate": fall_rate,
        }
    return results


# ═══════════════════════════════════════════════════════════════════
# BELIEFS V2 (second-order beliefs with fixed prompts)
# ═══════════════════════════════════════════════════════════════════

_MISTRAL_DIR = ROOT / PRIMARY

_BELIEF_V2_SOURCES = {
    "pure": _MISTRAL_DIR / "experiment_pure_log.json",
    "comm": _MISTRAL_DIR / "experiment_comm_log.json",
    "surveillance": _MISTRAL_DIR / "experiment_comm_log.json",
}


def _load_belief_v2_agents(treatment: str) -> list[dict]:
    """Load agent rows from v2 belief data (identified by second_order_belief_raw field).

    For 'comm' and 'surveillance', both come from the comm log.  The comm log
    has 1000 entries total: 600 old + 200 comm-v2 + 200 surveillance-v2.
    - comm: entries with second_order_belief_raw that are NOT in the last 200
    - surveillance: only the last 200 entries that have second_order_belief_raw
    """
    path = _BELIEF_V2_SOURCES.get(treatment)
    if path is None or not path.exists():
        return []
    periods = _load_experiment_log(path)
    if not periods:
        return []

    sigma = 0.3

    # Filter to entries with second_order_belief_raw
    if treatment == "surveillance":
        # Last 200 entries that have second_order_belief_raw
        candidates = periods[-200:]
    elif treatment == "comm":
        # Entries with second_order_belief_raw that are NOT in the last 200
        candidates = periods[:-200] if len(periods) > 200 else periods
    else:
        # pure: all entries with second_order_belief_raw
        candidates = periods

    rows = []
    for p in candidates:
        theta = p["theta"]
        theta_star = p["theta_star"]
        agents = p.get("agents") or []
        # Only include entries where agents have second_order_belief_raw
        if not agents or "second_order_belief_raw" not in agents[0]:
            continue
        # Compute period-level join fraction for calibration
        real_agents = [a for a in agents if not a.get("is_propaganda", False)]
        if not real_agents:
            continue
        period_join = sum(1 for a in real_agents if a.get("decision") == "JOIN") / len(real_agents)
        for a in agents:
            if a.get("api_error"):
                continue
            if a.get("is_propaganda", False):
                continue
            belief = a.get("belief")
            sob = a.get("second_order_belief")
            signal = a.get("signal")
            if belief is None or signal is None:
                continue
            decision = 1 if a.get("decision") == "JOIN" else 0
            posterior = float(stats.norm.cdf((theta_star - signal) / sigma))
            row = {
                "theta": theta,
                "theta_star": theta_star,
                "signal": signal,
                "z_score": a.get("z_score", 0.0),
                "belief": belief / 100.0,
                "decision": decision,
                "posterior": posterior,
                "period_join": period_join,
            }
            if sob is not None:
                row["second_order_belief"] = sob / 100.0
            rows.append(row)
    return rows


def compute_beliefs_v2():
    """Compute belief v2 statistics (with second-order beliefs)."""
    results = {}
    for treatment in ["pure", "comm", "surveillance"]:
        rows = _load_belief_v2_agents(treatment)
        if not rows:
            results[treatment] = {"status": "missing"}
            continue

        beliefs = np.array([r["belief"] for r in rows])
        decisions = np.array([r["decision"] for r in rows])
        thetas = np.array([r["theta"] for r in rows])
        posteriors = np.array([r["posterior"] for r in rows])
        period_joins = np.array([r["period_join"] for r in rows])

        r_post = pearson_with_ci(posteriors, beliefs)
        r_theta = pearson_with_ci(thetas, beliefs)
        r_belief_decision = pearson_with_ci(beliefs, decisions)

        join_beliefs = beliefs[decisions == 1]
        stay_beliefs = beliefs[decisions == 0]

        entry = {
            "n": len(rows),
            "mean_belief": round(float(beliefs.mean()), 4),
            "std_belief": round(float(beliefs.std()), 4),
            "mean_join": round(float(decisions.mean()), 4),
            "r_posterior_belief": r_post,
            "r_theta_belief": r_theta,
            "r_belief_decision": r_belief_decision,
            "mean_belief_join": round(float(join_beliefs.mean()), 4) if len(join_beliefs) else None,
            "mean_belief_stay": round(float(stay_beliefs.mean()), 4) if len(stay_beliefs) else None,
        }

        # Second-order belief stats
        sob_rows = [r for r in rows if "second_order_belief" in r]
        if sob_rows:
            sobs = np.array([r["second_order_belief"] for r in sob_rows])
            sob_thetas = np.array([r["theta"] for r in sob_rows])
            sob_period_joins = np.array([r["period_join"] for r in sob_rows])
            entry["second_order"] = {
                "n": len(sob_rows),
                "mean": round(float(sobs.mean()), 4),
                "std": round(float(sobs.std()), 4),
                "r_vs_theta": pearson_with_ci(sob_thetas, sobs),
                "r_vs_actual_join": pearson_with_ci(sob_period_joins, sobs),
            }

        results[treatment] = entry

    # Cross-treatment: does surveillance shift second-order beliefs?
    comm_rows = [r for r in _load_belief_v2_agents("comm") if "second_order_belief" in r]
    surv_rows = [r for r in _load_belief_v2_agents("surveillance") if "second_order_belief" in r]
    if comm_rows and surv_rows:
        comm_sobs = np.array([r["second_order_belief"] for r in comm_rows])
        surv_sobs = np.array([r["second_order_belief"] for r in surv_rows])
        t_stat, t_p = stats.ttest_ind(comm_sobs, surv_sobs)
        results["_surv_vs_comm_sob"] = {
            "comm_mean": round(float(comm_sobs.mean()), 4),
            "surv_mean": round(float(surv_sobs.mean()), 4),
            "delta_pp": round(float((surv_sobs.mean() - comm_sobs.mean()) * 100), 2),
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(t_p), 6),
        }

    # Preference falsification test: belief shift vs action shift (pure → surveillance)
    pure_rows = _load_belief_v2_agents("pure")
    surv_all_rows = _load_belief_v2_agents("surveillance")
    if pure_rows and surv_all_rows:
        pure_beliefs = np.array([r["belief"] for r in pure_rows])
        surv_beliefs = np.array([r["belief"] for r in surv_all_rows])
        pure_decisions = np.array([r["decision"] for r in pure_rows])
        surv_decisions = np.array([r["decision"] for r in surv_all_rows])

        # Test 1: first-order belief shift (should be NS)
        t_belief, p_belief = stats.ttest_ind(pure_beliefs, surv_beliefs)
        # Test 2: action shift (should be highly significant)
        t_action, p_action = stats.ttest_ind(pure_decisions, surv_decisions)

        results["_pref_falsification"] = {
            "pure_mean_belief": round(float(pure_beliefs.mean()), 4),
            "surv_mean_belief": round(float(surv_beliefs.mean()), 4),
            "belief_delta_pp": round(float((surv_beliefs.mean() - pure_beliefs.mean()) * 100), 2),
            "belief_t_stat": round(float(t_belief), 4),
            "belief_p_value": round(float(p_belief), 6),
            "pure_mean_join": round(float(pure_decisions.mean()), 4),
            "surv_mean_join": round(float(surv_decisions.mean()), 4),
            "action_delta_pp": round(float((surv_decisions.mean() - pure_decisions.mean()) * 100), 2),
            "action_t_stat": round(float(t_action), 4),
            "action_p_value": round(float(p_action), 6),
            "n_pure": len(pure_rows),
            "n_surv": len(surv_all_rows),
        }

    return results


def compute_hypothesis_table(all_stats: dict) -> list[dict]:
    """Build H1-H8 hypothesis table from already-computed stats.

    Each entry: {id, hypothesis, estimand, null, test, stat, p, supported}.
    H1-H4 use pooled Part I data; H5-H8 use primary-model infodesign/regime data.
    """
    part1 = all_stats.get("part1", {})
    infodesign = all_stats.get("infodesign", {})
    regime = all_stats.get("regime_control", {})
    table = []

    # ── H1: Alignment ─ r(J, A(θ)) ≠ 0 ──────────────────────────────
    pooled_pure = part1.get("_pooled_pure", {})
    r_attack = (pooled_pure.get("r_vs_attack") or {})
    table.append({
        "id": "H1",
        "hypothesis": "Sigmoid Response",
        "estimand": r"$r(J, A(\theta))$",
        "null": "$r = 0$",
        "test": "Pearson",
        "stat": r_attack.get("r"),
        "p": r_attack.get("p"),
        "n": r_attack.get("n"),
        "supported": _hypothesis_supported(r_attack.get("p"), alpha=0.05, reject_null=True),
    })

    # ── H2: Scramble ─ r should collapse to ~0 ───────────────────────
    pooled_scr = part1.get("_pooled_scramble", {})
    r_scr = (pooled_scr.get("r_vs_attack") or pooled_scr.get("r_vs_theta") or {})
    table.append({
        "id": "H2",
        "hypothesis": "Scramble Falsification",
        "estimand": r"$r(\text{scramble})$",
        "null": r"$r \neq 0$",
        "test": "Pearson",
        "stat": r_scr.get("r"),
        "p": r_scr.get("p"),
        "n": r_scr.get("n"),
        "supported": _hypothesis_supported(r_scr.get("p"), alpha=0.05, reject_null=False),
    })

    # ── H3: Flip ─ r should be negative ──────────────────────────────
    pooled_flip = part1.get("_pooled_flip", {})
    r_flip = (pooled_flip.get("r_vs_attack") or pooled_flip.get("r_vs_theta") or {})
    # One-sided p: test r < 0
    r_flip_val = r_flip.get("r")
    p_flip_two = r_flip.get("p")
    n_flip = r_flip.get("n")
    if r_flip_val is not None and p_flip_two is not None and not np.isnan(r_flip_val):
        p_flip_one = p_flip_two / 2.0 if r_flip_val < 0 else 1.0 - p_flip_two / 2.0
    else:
        p_flip_one = float("nan")
    table.append({
        "id": "H3",
        "hypothesis": "Directional Sensitivity",
        "estimand": r"$r(\text{flip})$",
        "null": r"$r \geq 0$",
        "test": "Pearson (1-sided)",
        "stat": r_flip_val,
        "p": round(p_flip_one, 6) if np.isfinite(p_flip_one) else None,
        "n": n_flip,
        "supported": _hypothesis_supported(p_flip_one, alpha=0.05, reject_null=True),
    })

    # ── H4: Communication ─ delta_pp ≠ 0 ─────────────────────────────
    comm_eff = part1.get("_pooled_comm_effect", {}).get("unpaired", {})
    table.append({
        "id": "H4",
        "hypothesis": "Communication Channel",
        "estimand": r"$\Delta_{\text{pp}}$",
        "null": "$= 0$",
        "test": "$t$-test",
        "stat": comm_eff.get("t_stat"),
        "p": comm_eff.get("p_value"),
        "n": None,
        "supported": _hypothesis_supported(comm_eff.get("p_value"), alpha=0.05, reject_null=True),
    })

    # ── H5-H8: Infodesign / regime treatments (primary model) ────────
    # H5: Stability ─ infodesign stability vs baseline
    h5 = _hypothesis_from_infodesign(infodesign, "stability", "Ambiguity Pooling")
    table.append(h5)

    # H6: Censorship ─ infodesign censor_upper vs baseline
    h6 = _hypothesis_from_infodesign(infodesign, "censor_upper", "Censorship Distortion")
    table.append(h6)

    # H7: Surveillance ─ regime surveillance effect
    surv = (regime.get("surveillance") or {}).get("Mistral Small Creative", {})
    # t-test: surveilled comm vs baseline comm (load raw data)
    surv_p = None
    surv_t = None
    surv_delta = surv.get("delta_vs_baseline_pp")
    primary = PRIMARY
    surv_df = _load_summary(ROOT / "surveillance" / primary / "experiment_comm_summary.csv")
    base_comm_df = _load_summary(ROOT / primary / "experiment_comm_summary.csv")
    if len(surv_df) > 0 and len(base_comm_df) > 0:
        sjcol = _join_col(surv_df)
        bjcol = _join_col(base_comm_df)
        t_s, p_s = stats.ttest_ind(
            surv_df[sjcol].astype(float).dropna(),
            base_comm_df[bjcol].astype(float).dropna(),
        )
        surv_t = round(float(t_s), 4)
        surv_p = round(float(p_s), 6)
    table.append({
        "id": "H7",
        "hypothesis": "Surveillance Chilling",
        "estimand": r"$\Delta_{\text{pp}}$",
        "null": "$= 0$",
        "test": "$t$-test",
        "stat": surv_t,
        "p": surv_p,
        "n": None,
        "supported": _hypothesis_supported(surv_p, alpha=0.05, reject_null=True),
    })

    # H8: Propaganda ─ regime propaganda k=5 effect (real agents vs baseline)
    prop = (regime.get("propaganda") or {}).get("k=5", {}).get("Mistral Small Creative", {})
    prop_delta = prop.get("delta_real_vs_baseline_pp")
    # t-test on period-level join fractions (propaganda real vs baseline comm)
    prop_t = None
    prop_p = None
    prop_log = _load_experiment_log(
        ROOT / "propaganda-k5" / primary / "experiment_comm_log.json"
    )
    if prop_log and len(base_comm_df) > 0:
        prop_real_jf = _real_join_from_comm_log(prop_log)
        bjcol = _join_col(base_comm_df)
        base_jf = base_comm_df[bjcol].astype(float).dropna()
        prop_real_jf = prop_real_jf.dropna()
        if len(prop_real_jf) > 0 and len(base_jf) > 0:
            t_p_stat, p_p_val = stats.ttest_ind(prop_real_jf, base_jf)
            prop_t = round(float(t_p_stat), 4)
            prop_p = round(float(p_p_val), 6)
    table.append({
        "id": "H8",
        "hypothesis": "Propaganda Dose-Response",
        "estimand": r"$\Delta_{\text{pp}}$",
        "null": "$= 0$",
        "test": "$t$-test",
        "stat": prop_t,
        "p": prop_p,
        "n": None,
        "supported": _hypothesis_supported(prop_p, alpha=0.05, reject_null=True),
    })

    return table


def _hypothesis_supported(p, alpha: float = 0.05, reject_null: bool = True) -> str:
    """Return 'Yes', 'No', or 'No (ambiguous)' based on p-value and direction."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "---"
    if reject_null:
        # Hypothesis is supported when we reject the null
        return "Yes" if p < alpha else "No (ambiguous)"
    else:
        # Hypothesis is supported when we fail to reject the null
        return "Yes" if p >= alpha else "No"


def _hypothesis_from_infodesign(infodesign: dict, design_key: str, label: str) -> dict:
    """Build a hypothesis table row from infodesign data (t-test vs baseline)."""
    primary = PRIMARY
    model_dir = ROOT / primary
    # Load design and baseline data to run a t-test
    df_design = _load_summary(model_dir / f"experiment_infodesign_{design_key}_summary.csv")
    # For baseline, prefer the bc_sweep canonical source
    bc_sweep_path = model_dir / "experiment_bc_sweep_summary.csv"
    df_baseline = pd.DataFrame()
    if bc_sweep_path.exists():
        try:
            bc = pd.read_csv(bc_sweep_path)
            if "theta_star_target" in bc.columns:
                df_baseline = bc[np.isclose(bc["theta_star_target"].astype(float), 0.50)]
        except Exception:
            pass
    if len(df_baseline) == 0:
        df_baseline = _load_summary(model_dir / "experiment_infodesign_baseline_summary.csv")
    # Also try the all summary
    if len(df_baseline) == 0:
        df_all = _load_summary(model_dir / "experiment_infodesign_all_summary.csv")
        if len(df_all) > 0 and "design" in df_all.columns:
            df_baseline = df_all[df_all["design"] == "baseline"]

    t_stat = None
    p_val = None
    if len(df_design) > 0 and len(df_baseline) > 0:
        jc_d = _join_col(df_design)
        jc_b = _join_col(df_baseline)
        t_s, p_v = stats.ttest_ind(
            df_design[jc_d].astype(float).dropna(),
            df_baseline[jc_b].astype(float).dropna(),
        )
        t_stat = round(float(t_s), 4)
        p_val = round(float(p_v), 6)

    h_id = {"Ambiguity Pooling": "H5", "Censorship Distortion": "H6"}.get(label, "H?")
    return {
        "id": h_id,
        "hypothesis": label,
        "estimand": r"$\Delta_{\text{pp}}$",
        "null": "$= 0$",
        "test": "$t$-test",
        "stat": t_stat,
        "p": p_val,
        "n": None,
        "supported": _hypothesis_supported(p_val, alpha=0.05, reject_null=True),
    }


def compute_ck_interaction():
    """2x2 interaction test: CK framing x coordination intensity."""
    primary = PRIMARY
    model_dir = ROOT / primary

    designs = {
        "ck_high_coord": (1, 1),    # (ck, high_coord)
        "ck_low_coord": (1, 0),
        "priv_high_coord": (0, 1),
        "priv_low_coord": (0, 0),
    }

    dfs = []
    cell_means = {}
    for dname, (ck, high) in designs.items():
        p = model_dir / f"experiment_infodesign_{dname}_summary.csv"
        if not p.exists():
            print(f"  WARNING: missing {p}")
            continue
        df = pd.read_csv(p)
        jcol = _join_col(df)
        df = df.copy()
        df["ck"] = ck
        df["high_coord"] = high
        df["_jf"] = df[jcol].astype(float)
        dfs.append(df)
        cell_means[dname] = round(float(df["_jf"].mean()), 4)

    if len(dfs) < 4:
        return {"status": "incomplete", "cell_means": cell_means}

    pooled = pd.concat(dfs, ignore_index=True)

    # OLS: join ~ ck + high_coord + ck*high_coord
    y = pooled["_jf"].values
    ck = pooled["ck"].values.astype(float)
    high = pooled["high_coord"].values.astype(float)
    interact = ck * high
    X = np.column_stack([np.ones_like(y), ck, high, interact])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape
    s2 = np.sum(resid**2) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    t_stats = beta / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

    labels = ["intercept", "ck", "high_coord", "interaction"]
    result = {
        "cell_means": cell_means,
        "n_obs": int(n),
    }
    for i, lbl in enumerate(labels):
        result[lbl] = {
            "beta": round(float(beta[i]), 4),
            "se": round(float(se[i]), 4),
            "t": round(float(t_stats[i]), 4),
            "p": round(float(p_values[i]), 6),
        }

    return result


def compute_fixed_messages_test():
    """Compare baseline comm vs fixed-messages surveillance test."""
    primary = PRIMARY
    baseline_path = ROOT / primary / "experiment_comm_summary.csv"
    surv_path = ROOT / "fixed-messages-surv" / primary / "experiment_comm_summary.csv"

    if not baseline_path.exists() or not surv_path.exists():
        return {"status": "missing"}

    df_base = pd.read_csv(baseline_path)
    df_surv = pd.read_csv(surv_path)

    base_mean = df_base["join_fraction_valid"].mean()
    surv_mean = df_surv["join_fraction_valid"].mean()
    delta_pp = (surv_mean - base_mean) * 100

    # Correlation with theta
    base_r, base_p = stats.pearsonr(df_base["theta"], df_base["join_fraction_valid"])
    surv_r, surv_p = stats.pearsonr(df_surv["theta"], df_surv["join_fraction_valid"])

    # Two-sample t-test
    t_stat, t_pval = stats.ttest_ind(
        df_surv["join_fraction_valid"], df_base["join_fraction_valid"]
    )

    return {
        "baseline_mean_join": round(base_mean, 4),
        "surv_mean_join": round(surv_mean, 4),
        "delta_pp": round(delta_pp, 1),
        "baseline_r_theta": round(base_r, 4),
        "surv_r_theta": round(surv_r, 4),
        "baseline_n": len(df_base),
        "surv_n": len(df_surv),
        "ttest_t": round(t_stat, 3),
        "ttest_p": round(t_pval, 4),
    }


def compute_classifier_baselines():
    """Load classifier baseline results from classifier_results.json."""
    path = Path(__file__).resolve().parent / "classifier_results.json"
    if not path.exists():
        return {"status": "missing"}
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# CROSS-GENERATOR robustness (cable, journalistic variants)
# ═══════════════════════════════════════════════════════════════════

def compute_cross_generator():
    """Compute r-values for cross-generator language variants."""
    results = {}
    cross_gen_base = ROOT / "cross-generator"
    if not cross_gen_base.exists():
        return results

    # Map of (model_display_name, variant) → csv path
    variant_map = {
        ("Mistral Small Creative", "baseline"): "mistralai/mistral-small-creative_baseline",
        ("Mistral Small Creative", "cable"): "mistralai/mistral-small-creative_cable",
        ("Mistral Small Creative", "journalistic"): "mistralai/mistral-small-creative_journalistic",
        ("Llama 3.3 70B", "baseline"): "meta-llama/llama-3.3-70b-instruct_baseline",
        ("Llama 3.3 70B", "cable"): "meta-llama/llama-3.3-70b-instruct_cable",
        ("Llama 3.3 70B", "journalistic"): "meta-llama/llama-3.3-70b-instruct_journalistic",
    }

    for (model_name, variant), rel_path in variant_map.items():
        # Find the CSV (nested slug dir due to bash slug issue)
        csvs = list((cross_gen_base / rel_path).rglob("experiment_pure_summary.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        if len(df) == 0:
            continue
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values
        theta = df["theta"].astype(float).values
        r_theta = pearson_with_ci(theta, jf)
        fit = _fit_logistic(theta, jf)
        entry = {
            "n_obs": int(len(df)),
            "mean_join": round(_safe_mean(jf), 4),
            "r_vs_theta": r_theta,
        }
        if fit is not None:
            entry["logistic_fit"] = fit
        if "theoretical_attack" in df.columns:
            entry["r_vs_attack"] = pearson_with_ci(
                df["theoretical_attack"].astype(float).values, jf
            )
        results.setdefault(model_name, {})[variant] = entry

    return results


# ═══════════════════════════════════════════════════════════════════
# PLACEBO CALIBRATION (wrong center ±0.3)
# ═══════════════════════════════════════════════════════════════════

def compute_placebo_calibration():
    """Compute r-values for placebo calibration experiments."""
    results = {}
    placebo_base = ROOT / "placebo-calibration"
    if not placebo_base.exists():
        return results

    variant_map = {
        ("Mistral Small Creative", "+0.3"): "mistralai/mistral-small-creative_shift_0p3",
        ("Mistral Small Creative", "-0.3"): "mistralai/mistral-small-creative_shift_neg0p3",
        ("Llama 3.3 70B", "+0.3"): "meta-llama/llama-3.3-70b-instruct_shift_0p3",
        ("Llama 3.3 70B", "-0.3"): "meta-llama/llama-3.3-70b-instruct_shift_neg0p3",
    }

    for (model_name, shift), rel_path in variant_map.items():
        csvs = list((placebo_base / rel_path).rglob("experiment_pure_summary.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        if len(df) == 0:
            continue
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values
        theta = df["theta"].astype(float).values
        r_theta = pearson_with_ci(theta, jf)
        fit = _fit_logistic(theta, jf)
        entry = {
            "n_obs": int(len(df)),
            "mean_join": round(_safe_mean(jf), 4),
            "r_vs_theta": r_theta,
        }
        if fit is not None:
            entry["logistic_fit"] = fit
        if "theoretical_attack" in df.columns:
            entry["r_vs_attack"] = pearson_with_ci(
                df["theoretical_attack"].astype(float).values, jf
            )
        results.setdefault(model_name, {})[shift] = entry

    return results


# ═══════════════════════════════════════════════════════════════════
# EXPANDED TEMPERATURE robustness (multi-model)
# ═══════════════════════════════════════════════════════════════════

def compute_temperature_expanded():
    """Temperature robustness for Llama 3.3 70B and Qwen3 235B."""
    results = {}
    temp_base = ROOT / "temperature-robustness"
    if not temp_base.exists():
        return results

    variant_map = {
        ("Llama 3.3 70B", "0.3"): "meta-llama/llama-3.3-70b-instruct_t03",
        ("Llama 3.3 70B", "0.5"): "meta-llama/llama-3.3-70b-instruct_t05",
        ("Llama 3.3 70B", "0.7"): "meta-llama/llama-3.3-70b-instruct_t07",
        ("Llama 3.3 70B", "1.0"): "meta-llama/llama-3.3-70b-instruct_t10",
        ("Llama 3.3 70B", "1.2"): "meta-llama/llama-3.3-70b-instruct_t12",
        ("Qwen3 235B", "0.3"): "qwen/qwen3-235b-a22b-2507_t03",
        ("Qwen3 235B", "0.5"): "qwen/qwen3-235b-a22b-2507_t05",
        ("Qwen3 235B", "0.7"): "qwen/qwen3-235b-a22b-2507_t07",
        ("Qwen3 235B", "1.0"): "qwen/qwen3-235b-a22b-2507_t10",
        ("Qwen3 235B", "1.2"): "qwen/qwen3-235b-a22b-2507_t12",
    }

    for (model_name, temp), rel_path in variant_map.items():
        csvs = list((temp_base / rel_path).rglob("experiment_pure_summary.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        if len(df) == 0:
            continue
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values
        theta = df["theta"].astype(float).values
        r_theta = pearson_with_ci(theta, jf)
        fit = _fit_logistic(theta, jf)
        entry = {
            "n_obs": int(len(df)),
            "mean_join": round(_safe_mean(jf), 4),
            "r_vs_theta": r_theta,
        }
        if fit is not None:
            entry["logistic_fit"] = fit
        results.setdefault(model_name, {})[f"T={temp}"] = entry

    return results


# ═══════════════════════════════════════════════════════════════════
# EXPANDED UNCALIBRATED robustness (all 7 models)
# ═══════════════════════════════════════════════════════════════════

def compute_uncalibrated_expanded():
    """Uncalibrated robustness for all models with data."""
    results = {}
    uncal_base = ROOT / "uncalibrated-robustness"
    if not uncal_base.exists():
        return results

    # Direct slug dirs (older runs)
    direct_slugs = [
        PRIMARY,
        "meta-llama--llama-3.3-70b-instruct",
        "qwen--qwen3-235b-a22b-2507",
        "minimax--minimax-m2-her",
    ]
    for slug in direct_slugs:
        p = uncal_base / slug / "experiment_pure_summary.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values
        theta = df["theta"].astype(float).values
        # Skip if all NaN (e.g. Trinity with 100% API errors)
        if np.all(np.isnan(jf)):
            continue
        r_theta = pearson_with_ci(theta, jf)
        name = SHORT.get(slug, slug)
        results[name] = {
            "n_obs": int(len(df)),
            "mean_join": round(_safe_mean(jf), 4),
            "r_vs_theta": r_theta,
        }

    # Nested slug dirs (newer runs with bash slug issue)
    nested_map = {
        "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3 30B",
        "openai/gpt-oss-120b": "GPT-OSS 120B",
        "arcee-ai/trinity-large-preview_free": "Trinity Large",
    }
    for rel_path, name in nested_map.items():
        csvs = list((uncal_base / rel_path).rglob("experiment_pure_summary.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values
        theta = df["theta"].astype(float).values
        if np.all(np.isnan(jf)):
            continue
        r_theta = pearson_with_ci(theta, jf)
        results[name] = {
            "n_obs": int(len(df)),
            "mean_join": round(_safe_mean(jf), 4),
            "r_vs_theta": r_theta,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# PUNISHMENT RISK ELICITATION
# ═══════════════════════════════════════════════════════════════════

def compute_punishment_risk():
    """Compute punishment risk elicitation statistics."""
    results = {}
    pr_base = ROOT / "punishment-risk"
    if not pr_base.exists():
        return results

    conditions = {
        ("Mistral Small Creative", "pure"): "mistralai/mistral-small-creative",
        ("Mistral Small Creative", "comm"): "mistralai/mistral-small-creative",
        ("Mistral Small Creative", "surveillance"): "mistralai/mistral-small-creative_surv",
        ("Llama 3.3 70B", "pure"): "meta-llama/llama-3.3-70b-instruct",
        ("Llama 3.3 70B", "comm"): "meta-llama/llama-3.3-70b-instruct",
        ("Llama 3.3 70B", "surveillance"): "meta-llama/llama-3.3-70b-instruct_surv",
    }

    for (model_name, condition), rel_path in conditions.items():
        if condition == "surveillance":
            treatment_file = "experiment_comm_summary.csv"
        elif condition == "comm":
            treatment_file = "experiment_comm_summary.csv"
        else:
            treatment_file = "experiment_pure_summary.csv"

        csvs = list((pr_base / rel_path).rglob(treatment_file))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        if len(df) == 0 or "punishment_risk" not in df.columns:
            # Try loading from log JSON if summary doesn't have the column
            continue
        pr = df["punishment_risk"].dropna()
        if len(pr) == 0:
            continue
        jcol = _join_col(df)
        jf = df[jcol].astype(float).values

        entry = {
            "n_obs": int(len(df)),
            "n_pr_valid": int(len(pr)),
            "mean_pr": round(float(pr.mean()), 2),
            "std_pr": round(float(pr.std()), 2),
            "mean_join": round(_safe_mean(jf), 4),
        }

        # Correlation between punishment risk and join rate
        if len(pr) > 3:
            # Align: only rows with both valid
            valid = df[[jcol, "punishment_risk"]].dropna()
            if len(valid) > 3:
                r_pr_join = pearson_with_ci(
                    valid["punishment_risk"].values, valid[jcol].values
                )
                entry["r_pr_join"] = r_pr_join

        results.setdefault(model_name, {})[condition] = entry

    # Check log files for agent-level punishment risk data
    for (model_name, condition), rel_path in conditions.items():
        if condition == "surveillance":
            log_file = "experiment_comm_log.json"
        elif condition == "comm":
            log_file = "experiment_comm_log.json"
        else:
            log_file = "experiment_pure_log.json"

        log_csvs = list((pr_base / rel_path).rglob(log_file))
        if not log_csvs:
            continue
        log_data = _load_experiment_log(log_csvs[0])
        if not log_data:
            continue

        pr_values = []
        join_decisions = []
        for period in log_data:
            for a in period.get("agents", []):
                if a.get("api_error") or a.get("is_propaganda", False):
                    continue
                pr_val = a.get("punishment_risk")
                if pr_val is not None:
                    pr_values.append(pr_val)
                    join_decisions.append(1 if a.get("decision") == "JOIN" else 0)

        if pr_values:
            pr_arr = np.array(pr_values)
            dec_arr = np.array(join_decisions)
            agent_entry = {
                "n_agents": len(pr_arr),
                "mean_pr": round(float(pr_arr.mean()), 2),
                "std_pr": round(float(pr_arr.std()), 2),
                "mean_pr_join": round(float(pr_arr[dec_arr == 1].mean()), 2) if (dec_arr == 1).any() else None,
                "mean_pr_stay": round(float(pr_arr[dec_arr == 0].mean()), 2) if (dec_arr == 0).any() else None,
            }
            if len(pr_arr) > 3:
                agent_entry["r_pr_decision"] = pearson_with_ci(pr_arr, dec_arr)
            results.setdefault(model_name, {}).setdefault(condition, {})["agent_level"] = agent_entry

    return results


# ═══════════════════════════════════════════════════════════════════
# PARSE ERROR / REFUSAL RATES
# ═══════════════════════════════════════════════════════════════════

def compute_parse_error_rates():
    """Aggregate api_error_rate and unparseable_rate by treatment × model."""
    results = {}
    treatments = ["pure", "comm", "scramble", "flip"]
    for model in PART1_MODELS:
        name = SHORT[model]
        model_results = {}
        for treatment in treatments:
            df = load(model, treatment)
            if len(df) == 0:
                continue
            entry = {"n_periods": int(len(df))}
            if "api_error_rate" in df.columns:
                entry["mean_api_error_rate"] = round(float(df["api_error_rate"].mean()), 4)
            if "unparseable_rate" in df.columns:
                entry["mean_unparseable_rate"] = round(float(df["unparseable_rate"].mean()), 4)
            if "n_api_error" in df.columns:
                entry["total_api_errors"] = int(df["n_api_error"].sum())
            if "n_unparseable" in df.columns:
                entry["total_unparseable"] = int(df["n_unparseable"].sum())
            if "n_valid" in df.columns and "n_join" in df.columns:
                n_agents_total = df["n_valid"].sum() + df.get("n_api_error", pd.Series([0]*len(df))).sum() + df.get("n_unparseable", pd.Series([0]*len(df))).sum()
                entry["total_decisions"] = int(n_agents_total)
            model_results[treatment] = entry
        if model_results:
            results[name] = model_results

    # Also check infodesign treatments for primary model
    primary = PRIMARY
    primary_dir = ROOT / primary
    info_csv = primary_dir / "experiment_infodesign_all_summary.csv"
    if info_csv.exists():
        df = pd.read_csv(info_csv)
        if len(df) > 0 and "design" in df.columns:
            info_results = {}
            for design in sorted(df["design"].unique()):
                sub = df[df["design"] == design]
                entry = {"n_periods": int(len(sub))}
                if "api_error_rate" in sub.columns:
                    entry["mean_api_error_rate"] = round(float(sub["api_error_rate"].mean()), 4)
                if "unparseable_rate" in sub.columns:
                    entry["mean_unparseable_rate"] = round(float(sub["unparseable_rate"].mean()), 4)
                info_results[design] = entry
            results["_infodesign"] = info_results

    return results


def main():
    print("Computing Part I statistics...")
    part1 = compute_part1()

    print("Computing information design statistics...")
    infodesign = compute_infodesign()
    infodesign_comm = compute_infodesign_comm()

    print("Computing regime control statistics...")
    regime = compute_regime_control()

    print("Computing robustness statistics...")
    robust = compute_robustness()

    print("Running pooled OLS...")
    ols = pooled_ols(None)

    print("Computing logistic fits (cutoffs + slopes)...")
    logistic_fits = compute_logistic_fits()

    print("Computing clustered standard errors...")
    clustered_ses = compute_clustered_ses()

    print("Computing temperature robustness statistics...")
    temp_robust = compute_temperature_robustness()

    print("Computing surveillance variant statistics...")
    surv_variants = compute_surveillance_variants()

    print("Computing uncalibrated robustness statistics...")
    uncalibrated = compute_uncalibrated()

    print("Computing beliefs v2 statistics...")
    beliefs_v2 = compute_beliefs_v2()

    print("Computing CK interaction test...")
    ck_interaction = compute_ck_interaction()

    print("Computing classifier baselines...")
    classifier_baselines = compute_classifier_baselines()

    print("Computing fixed-messages test...")
    fixed_messages_test = compute_fixed_messages_test()

    print("Computing cross-generator robustness...")
    cross_generator = compute_cross_generator()

    print("Computing placebo calibration...")
    placebo_calibration = compute_placebo_calibration()

    print("Computing expanded temperature robustness...")
    temperature_expanded = compute_temperature_expanded()

    print("Computing expanded uncalibrated robustness...")
    uncalibrated_expanded = compute_uncalibrated_expanded()

    print("Computing punishment risk elicitation...")
    punishment_risk = compute_punishment_risk()

    print("Computing parse error rates...")
    parse_errors = compute_parse_error_rates()

    all_stats = {
        "part1": part1,
        "infodesign": infodesign,
        "infodesign_comm": infodesign_comm,
        "regime_control": regime,
        "robustness": robust,
        "pooled_ols": ols,
        "logistic_fits": logistic_fits,
        "clustered_ses": clustered_ses,
        "surveillance_variants": surv_variants,
        "temperature_robustness": temp_robust,
        "uncalibrated": uncalibrated,
        "beliefs_v2": beliefs_v2,
        "ck_interaction": ck_interaction,
        "classifier_baselines": classifier_baselines,
        "fixed_messages_test": fixed_messages_test,
        "cross_generator": cross_generator,
        "placebo_calibration": placebo_calibration,
        "temperature_expanded": temperature_expanded,
        "uncalibrated_expanded": uncalibrated_expanded,
        "punishment_risk": punishment_risk,
        "parse_errors": parse_errors,
    }

    print("Computing hypothesis table...")
    hypothesis_table = compute_hypothesis_table(all_stats)
    all_stats["hypothesis_table"] = hypothesis_table

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
