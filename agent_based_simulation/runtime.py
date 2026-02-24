"""Shared helpers used across run scripts, figure generation, and model discovery."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def model_slug(model_name: str) -> str:
    """Sanitize model names for filesystem paths."""
    return model_name.replace("/", "--").replace(":", "_").replace(" ", "_")


def parse_float_list(value):
    """Parse comma-separated float strings or pass through iterables."""
    if value is None:
        return None
    if isinstance(value, str):
        return [float(v.strip()) for v in value.split(",") if v.strip()] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (int, float, bytes)):
        return [float(v) for v in value]
    return [float(value)]


def ensure_agg_backend() -> None:
    """Force a non-interactive matplotlib backend, safe for worker threads."""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)


def resolve_model_output_dir(output_dir: str | Path, model_name: str) -> Path:
    """Keep model outputs in `<output_root>/<model_slug>` unless already model-specific."""
    out = Path(output_dir)
    slug = model_slug(model_name)
    return out if out.name == slug else out / slug


# ── Shared CLI argument definitions ──────────────────────────────────


def add_common_args(parser) -> None:
    """Add model, API, and briefing generator args shared across CLIs."""
    parser.add_argument("--model", type=str, default="google/gemini-2.0-flash-001")
    parser.add_argument("--api-base-url", type=str, default="https://openrouter.ai/api/v1",
                        help="API base URL (use http://localhost:1234/v1 for LM Studio)")
    parser.add_argument("--n-agents", type=int, default=25)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--benefit", type=float, default=1.0)
    parser.add_argument("--cost", type=float, default=1.0,
                        help="Cost of joining a failed uprising")
    parser.add_argument("--seed", type=int, default=5150)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--llm-max-retries", type=int, default=5,
                        help="Retries per call for API/network errors")
    parser.add_argument("--llm-empty-retries", type=int, default=12,
                        help="Retries per call for empty/near-empty model responses")
    # Briefing generator tuning
    parser.add_argument("--cutoff-center", type=float, default=0.0)
    parser.add_argument("--clarity-width", type=float, default=1.0)
    parser.add_argument("--direction-slope", type=float, default=0.8,
                        help="Steepness of direction logistic (lower = more gradual)")
    parser.add_argument("--coordination-slope", type=float, default=0.6,
                        help="Steepness of coordination logistic")
    parser.add_argument("--dissent-floor", type=float, default=0.25,
                        help="Min fraction of contrary evidence")
    parser.add_argument("--mixed-cue-clarity", type=float, default=0.5,
                        help="Clarity threshold below which ambiguous cues appear")
    parser.add_argument("--bottomline-cuts", type=str,
                        default="0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85",
                        help="8 cutpoints for bottom-line tier mapping")
    parser.add_argument("--unclear-cuts", type=str,
                        default="0.18,0.33,0.48,0.62,0.77",
                        help="5 cutpoints for uncertainty tier mapping")
    parser.add_argument("--coordination-cuts", type=str,
                        default="0.12,0.25,0.42,0.58,0.75",
                        help="5 cutpoints for atmosphere tier mapping")
    parser.add_argument("--coordination-blend-prob", type=float, default=0.6,
                        help="Blend probability in intermediate coordination bands")
    parser.add_argument("--language-variant", type=str, default="baseline",
                        help="Briefing text schema: legacy|baseline_min|baseline|baseline_assess|baseline_full")
    # Per-model calibration loading
    parser.add_argument("--load-calibrated", action="store_true",
                        help="Load calibrated briefing params for --model")
    parser.add_argument("--calibration-dir", type=str, default=None,
                        help="Directory containing calibrated_index.json (defaults to --output-dir)")
    parser.add_argument("--output-dir", type=str,
                        default=str(OUTPUT_DIR))


# ── Model discovery (was model_discovery.py) ─────────────────────────


def discover_model_dirs(
    base_dir: str | Path,
    required_files: tuple[str, ...] = (),
    *,
    require_any: bool = True,
) -> list[Path]:
    """Find candidate model directories under ``base_dir``."""
    base = Path(base_dir)
    if not base.exists():
        return []

    candidates = []
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        if not required_files:
            candidates.append(child)
            continue

        matches = [child / name for name in required_files if (child / name).exists()]
        if require_any and matches:
            candidates.append(child)
        elif not require_any and len(matches) == len(required_files):
            candidates.append(child)

    return candidates


def model_label(model_dir: str | Path) -> str:
    """Read a readable model label from manifest or fallback to directory name."""
    path = Path(model_dir)
    manifest_dir = path / "manifests"
    if manifest_dir.exists():
        manifests = sorted(manifest_dir.glob("manifest_*.json"))
        if manifests:
            with open(manifests[0], encoding="utf-8") as f:
                payload = json.load(f)
            model = payload.get("resolved", {}).get("model")
            if model:
                return model.replace("/", " / ")
    return path.name.replace("--", "/")


# ── Plot style (was plot_style.py) ───────────────────────────────────


def apply_serif_paper_style() -> None:
    """Apply a consistent serif plot style for manuscript figures."""
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ── Game theory math (was core.py) ──────────────────────────────────


def clamp01(x, eps=1e-8):
    """Clamp values to [eps, 1-eps]. Matches Mata clamp01()."""
    import numpy as np

    return np.clip(x, eps, 1 - eps)


def theta_star_baseline(b):
    """Baseline model (sigma -> 0): theta* = b/(1+b)."""
    return b / (1 + b)


def attack_mass(theta_star, theta, sigma):
    """Fraction of population participating in the coup.

    A(theta) = Phi[(x* - theta) / sigma]
    where x* = theta* + sigma * ppf(theta*)
    """
    import numpy as np
    from scipy.stats import norm

    theta_star = np.asarray(theta_star, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    x_star = theta_star + sigma * norm.ppf(clamp01(theta_star))
    return norm.cdf((x_star - theta) / sigma)


# ── Network generation (was network.py) ─────────────────────────────


def build_network(n_agents, n_neighbors=3, rewire_prob=0.3, seed=None):
    """Build a Watts-Strogatz small-world graph.

    Returns (adjacency dict, networkx Graph).
    """
    import networkx as nx

    k = max(n_neighbors, 2)
    if k % 2 != 0:
        k += 1  # networkx requires even k

    G = nx.watts_strogatz_graph(n_agents, k, rewire_prob, seed=seed)
    adjacency = {node: list(G.neighbors(node)) for node in G.nodes()}
    return adjacency, G


# ── Result column helpers (was results_utils.py) ────────────────────


def join_fraction_column(df):
    """Return the preferred join-rate column from an experiment dataframe."""
    if (
        "join_fraction_valid" in df.columns
        and df["join_fraction_valid"].notna().any()
    ):
        return "join_fraction_valid"
    return "join_fraction"


def join_fraction_column_with_label(df):
    """Return (join column name, human-readable label)."""
    col = join_fraction_column(df)
    if col == "join_fraction_valid":
        return col, "LLM agent join fraction (valid responses)"
    return col, "LLM agent join fraction"


def add_join_column(df, target_col: str = "_join", source_col: str | None = None):
    """Return a copy with a standard join-fraction column added."""
    source_col = source_col or join_fraction_column(df)
    df = df.copy()
    df[target_col] = df[source_col]
    return df


def merge_with_join_columns(
    df_left,
    df_right,
    keys: tuple = ("country", "period"),
    left_suffix: str = "_left",
    right_suffix: str = "_right",
    left_join_col: str | None = None,
    right_join_col: str | None = None,
):
    """Merge two treatment frames after adding standardized join columns."""
    left = add_join_column(df_left, source_col=left_join_col)
    right = add_join_column(df_right, source_col=right_join_col)
    return left.merge(right, on=list(keys), suffixes=(left_suffix, right_suffix))


def summarize_binned_stats(
    df,
    x_col: str,
    y_col: str,
    bins,
    method: str = "cut",
    *,
    duplicates: str = "drop",
    bin_col: str = "_bin",
):
    """Summarize y within binned x by mean and standard-error."""
    import pandas as pd

    work = df.copy()
    if method == "qcut":
        work[bin_col] = pd.qcut(work[x_col], q=bins, duplicates=duplicates)
    else:
        work[bin_col] = pd.cut(work[x_col], bins=bins)

    return work.groupby(bin_col, observed=True).agg(
        x=(x_col, "mean"),
        y=(y_col, "mean"),
        se=(y_col, lambda s: s.std(ddof=1) / (len(s) ** 0.5 if len(s) else 1)),
        n=(y_col, "count"),
    ).dropna()


# ── LLM response cache (was llm_cache.py) ───────────────────────────

ENV_LLM_CACHE_DIR = "GGC_LLM_CACHE_DIR"


def _sha256_json(obj: Any) -> str:
    data = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass
class FileLLMCache:
    """Very small disk cache for LLM responses keyed by request hash."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                obj = json.load(f)
            value = obj.get("response")
            return value if isinstance(value, str) else None
        except Exception:
            return None

    def set(self, key: str, request: dict, response: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        payload = {
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": request,
            "response": response,
        }
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)


_CACHE: FileLLMCache | None = None
_CACHE_DIR: str | None = None


def get_cache() -> FileLLMCache | None:
    """Return cache instance if ENV_LLM_CACHE_DIR is set, else None."""
    global _CACHE, _CACHE_DIR
    cache_dir = (os.environ.get(ENV_LLM_CACHE_DIR) or "").strip()
    if not cache_dir:
        return None
    if _CACHE is None or _CACHE_DIR != cache_dir:
        _CACHE = FileLLMCache(Path(cache_dir))
        _CACHE_DIR = cache_dir
    return _CACHE


def build_cache_key_and_request(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict]:
    request = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    return _sha256_json(request), request
