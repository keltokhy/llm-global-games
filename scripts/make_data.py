#!/usr/bin/env python3
"""Smart experiment runner — like make(1) for the paper's data.

Declares every experiment output needed for the paper, checks what
exists on disk, and runs only what's missing or incomplete.

Usage:
    uv run python scripts/make_data.py                # status: show what exists / what's missing
    uv run python scripts/make_data.py --run           # execute missing experiments
    uv run python scripts/make_data.py --run --group calibration   # run only one group
    uv run python scripts/make_data.py --dry-run       # print commands without executing
    uv run python scripts/make_data.py --concurrency 200  # override max-concurrent (default 200)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

# ── Models ────────────────────────────────────────────────────────────

MODELS = {
    "mistral":   "mistralai/mistral-small-creative",
    "llama":     "meta-llama/llama-3.3-70b-instruct",
    "gptoss":    "openai/gpt-oss-120b",
    "qwen235":   "qwen/qwen3-235b-a22b-2507",
    "qwen30":    "qwen/qwen3-30b-a3b-instruct-2507",
    "olmo":      "allenai/olmo-3-7b-instruct",
    "minimax":   "minimax/minimax-m2-her",
    "ministral": "mistralai/ministral-3b-2512",
}

# Trinity excluded — unreliable API
PART1_MODELS = list(MODELS.keys())  # all 8 for Part I

INFODESIGN_CORE_MODELS = ["mistral", "llama", "gptoss", "qwen235", "qwen30", "olmo", "ministral"]

SURVEILLANCE_MODELS = ["mistral", "llama", "qwen30"]

def slug(short: str) -> str:
    return MODELS[short].replace("/", "--")

def model(short: str) -> str:
    return MODELS[short]


# ── Target definition ─────────────────────────────────────────────────

@dataclass
class Target:
    name: str
    file: str              # relative to output/
    min_rows: int          # 0 means just check file exists
    command: str           # shell command (from repo root)
    group: str = ""
    depends: list[str] = field(default_factory=list)  # other target names

    @property
    def path(self) -> Path:
        return OUTPUT / self.file

    def exists(self) -> bool:
        return self.path.exists()

    def row_count(self) -> int:
        if not self.exists():
            return 0
        if self.path.suffix == ".json":
            return 1 if self.path.stat().st_size > 10 else 0
        try:
            with open(self.path) as f:
                return sum(1 for _ in f) - 1  # subtract header
        except Exception:
            return 0

    def is_complete(self) -> bool:
        if self.min_rows == 0:
            return self.exists()
        return self.row_count() >= self.min_rows

    def status(self) -> str:
        if not self.exists():
            return "MISSING"
        n = self.row_count()
        if n >= self.min_rows:
            return f"OK ({n})"
        return f"PARTIAL ({n}/{self.min_rows})"


# ── Manifest ──────────────────────────────────────────────────────────

def build_manifest(conc: int = 200) -> list[Target]:
    MC = f"--max-concurrent {conc}"
    targets = []

    s_m, m_m = slug("mistral"), model("mistral")  # primary model shorthand

    # ── Calibration ───────────────────────────────────────────────
    # JSON file, min_rows=0 (existence check only)
    for short in PART1_MODELS:
        s, m = slug(short), model(short)
        targets.append(Target(
            name=f"calibrate_{short}",
            file=f"{s}/calibrated_params_{s}.json",
            min_rows=0,
            command=f"uv run python -m agent_based_simulation.run autocalibrate --model {m} {MC}",
            group="calibration",
        ))

    # ── Part I: core experiments ──────────────────────────────────
    # Primary model (mistral): 10×20=200 for pure/comm (main analysis, belief base)
    #                           5×20=100  for scramble/flip (falsification checks)
    # Cross-models:             5×20=100  for all treatments (robustness checks)
    for short in PART1_MODELS:
        s, m = slug(short), model(short)
        cal_dep = f"calibrate_{short}"
        for tx in ["pure", "comm", "scramble", "flip"]:
            is_primary_main = (short == "mistral" and tx in ("pure", "comm"))
            n_c = 10 if is_primary_main else 5
            n_p = 20
            min_r = n_c * n_p
            targets.append(Target(
                name=f"{tx}_{short}",
                file=f"{s}/experiment_{tx}_summary.csv",
                min_rows=min_r,
                command=f"uv run python -m agent_based_simulation.run {tx} --model {m} --load-calibrated --n-countries {n_c} --n-periods {n_p} {MC}",
                group="part1",
                depends=[cal_dep],
            ))

    # ── Part II: infodesign core designs (cross-model) ────────────
    # 9 θ-points × 30 reps = 270 rows per design per model
    CORE_DESIGNS = ["baseline", "stability", "instability", "public_signal",
                    "censor_upper", "censor_lower", "scramble", "flip"]

    for short in INFODESIGN_CORE_MODELS:
        s, m = slug(short), model(short)
        cal_dep = f"calibrate_{short}"
        for design in CORE_DESIGNS:
            targets.append(Target(
                name=f"infodesign_{design}_{short}",
                file=f"{s}/experiment_infodesign_{design}_summary.csv",
                min_rows=270,
                command=f"uv run python -m agent_based_simulation.run_infodesign --model {m} --load-calibrated --designs {design} --reps 30 --append {MC}",
                group="infodesign_core",
                depends=[cal_dep],
            ))

    # ── Part II: primary model extras ─────────────────────────────
    # All use 30 reps → 270 rows each (including decomposition)
    PRIMARY_EXTRAS = [
        "coord_amplified", "coord_suppressed",
        "domain_scramble_coord", "domain_scramble_state", "within_scramble",
        "bc_high_cost", "bc_low_cost",
        "censor_upper_known",
        "stability_clarity", "stability_direction", "stability_dissent",
        "ck_high_coord", "ck_low_coord", "priv_high_coord", "priv_low_coord",
        "hard_scramble",
    ]
    for design in PRIMARY_EXTRAS:
        targets.append(Target(
            name=f"infodesign_{design}_mistral",
            file=f"{s_m}/experiment_infodesign_{design}_summary.csv",
            min_rows=270,
            command=f"uv run python -m agent_based_simulation.run_infodesign --model {m_m} --load-calibrated --designs {design} --reps 30 --append {MC}",
            group="infodesign_primary",
            depends=["calibrate_mistral"],
        ))

    # Hard scramble cross-model (Llama)
    targets.append(Target(
        name="infodesign_hard_scramble_llama",
        file=f"{slug('llama')}/experiment_infodesign_hard_scramble_summary.csv",
        min_rows=270,
        command=f"uv run python -m agent_based_simulation.run_infodesign --model {model('llama')} --load-calibrated --designs hard_scramble --reps 30 --append {MC}",
        group="infodesign_primary",
        depends=["calibrate_llama"],
    ))

    # ── B/C sweep (primary) ───────────────────────────────────────
    # 7 B/C values × 9 θ × 30 reps = 1890 rows
    targets.append(Target(
        name="bc_sweep_mistral",
        file=f"{s_m}/experiment_bc_sweep_summary.csv",
        min_rows=1890,
        command=f"uv run python -m agent_based_simulation.run_infodesign --model {m_m} --load-calibrated --benefit-grid 0.25,0.35,0.40,0.50,0.60,0.65,0.75 --reps 30 {MC}",
        group="bc_sweep",
        depends=["calibrate_mistral"],
    ))

    # ── Surveillance ──────────────────────────────────────────────
    # 10 countries × 100 periods = 1000 rows per model
    for short in SURVEILLANCE_MODELS:
        s, m = slug(short), model(short)
        targets.append(Target(
            name=f"surveillance_{short}",
            file=f"surveillance/{s}/experiment_comm_summary.csv",
            min_rows=1000,
            command=f"uv run python -m agent_based_simulation.run comm --model {m} --load-calibrated --surveillance --n-countries 10 --n-periods 100 --output-dir output/surveillance {MC}",
            group="surveillance",
            depends=[f"calibrate_{short}"],
        ))

    # ── Propaganda ────────────────────────────────────────────────
    # 10 countries × 50 periods = 500 rows per config
    for k in [2, 5, 10]:
        targets.append(Target(
            name=f"propaganda_k{k}_mistral",
            file=f"propaganda-k{k}/{s_m}/experiment_comm_summary.csv",
            min_rows=500,
            command=f"uv run python -m agent_based_simulation.run comm --model {m_m} --load-calibrated --n-propaganda {k} --n-countries 10 --n-periods 50 --output-dir output/propaganda-k{k} {MC}",
            group="propaganda",
            depends=["calibrate_mistral"],
        ))

    # Propaganda k5 cross-model (Llama)
    targets.append(Target(
        name="propaganda_k5_llama",
        file=f"propaganda-k5/{slug('llama')}/experiment_comm_summary.csv",
        min_rows=500,
        command=f"uv run python -m agent_based_simulation.run comm --model {model('llama')} --load-calibrated --n-propaganda 5 --n-countries 10 --n-periods 50 --output-dir output/propaganda-k5 {MC}",
        group="propaganda",
        depends=["calibrate_llama"],
    ))

    # Propaganda × surveillance
    targets.append(Target(
        name="propaganda_surveillance_mistral",
        file=f"propaganda-surveillance/{s_m}/experiment_comm_summary.csv",
        min_rows=500,
        command=f"uv run python -m agent_based_simulation.run comm --model {m_m} --load-calibrated --n-propaganda 5 --surveillance --n-countries 10 --n-periods 50 --output-dir output/propaganda-surveillance {MC}",
        group="propaganda",
        depends=["calibrate_mistral"],
    ))

    # ── Surveillance × censorship ─────────────────────────────────
    # 3 designs × 9 θ × 30 reps = 810 rows per model
    SXC_MODELS = ["mistral", "llama", "gptoss", "qwen235"]
    for short in SXC_MODELS:
        s, m = slug(short), model(short)
        targets.append(Target(
            name=f"surv_x_censor_{short}",
            file=f"surveillance-x-censorship/{s}/experiment_infodesign_all_summary.csv",
            min_rows=810,
            command=f"uv run python -m agent_based_simulation.run_infodesign --model {m} --load-calibrated --treatment comm --surveillance --designs baseline censor_upper censor_lower --reps 30 --output-dir output/surveillance-x-censorship {MC}",
            group="surv_x_censor",
            depends=[f"calibrate_{short}"],
        ))

    # ── Infodesign under communication (primary) ──────────────────
    # 3 designs × 9 θ × 30 reps = 810 rows
    targets.append(Target(
        name="infodesign_comm_mistral",
        file=f"mistralai--mistral-small-creative-infodesign-comm/{s_m}/experiment_infodesign_all_summary.csv",
        min_rows=810,
        command=f"uv run python -m agent_based_simulation.run_infodesign --model {m_m} --load-calibrated --treatment comm --designs baseline censor_upper censor_lower --reps 30 --output-dir output/mistralai--mistral-small-creative-infodesign-comm {MC}",
        group="infodesign_comm",
        depends=["calibrate_mistral"],
    ))

    # ── Bandwidth robustness ──────────────────────────────────────
    # 4 designs × 9 θ × 20 reps = 720 rows per bandwidth
    for bw, bw_label in [("0.05", "005"), ("0.30", "030")]:
        targets.append(Target(
            name=f"bandwidth_{bw_label}_mistral",
            file=f"bandwidth-{bw_label}/{s_m}/experiment_infodesign_all_summary.csv",
            min_rows=720,
            command=f"uv run python -m agent_based_simulation.run_infodesign --model {m_m} --load-calibrated --bandwidth {bw} --designs baseline stability censor_upper censor_lower --reps 20 --output-dir output/bandwidth-{bw_label} {MC}",
            group="bandwidth",
            depends=["calibrate_mistral"],
        ))

    # ── Temperature robustness ────────────────────────────────────
    # 5 countries × 20 periods = 100 rows per temperature
    for temp in ["0.3", "0.7", "1.0"]:
        t_label = temp.replace(".", "")
        targets.append(Target(
            name=f"temperature_T{t_label}_mistral",
            file=f"temperature-robustness-T{temp}/{s_m}/experiment_pure_summary.csv",
            min_rows=100,
            command=f"uv run python -m agent_based_simulation.run pure --model {m_m} --load-calibrated --temperature {temp} --n-countries 5 --n-periods 20 --output-dir output/temperature-robustness-T{temp} {MC}",
            group="temperature",
            depends=["calibrate_mistral"],
        ))

    # ── Agent count robustness ────────────────────────────────────
    # 10 countries × 50 periods = 500 rows per agent count
    for n in [5, 10, 50, 100]:
        targets.append(Target(
            name=f"n_agents_{n}_mistral",
            file=f"mistralai--mistral-small-creative-n{n}/{s_m}/experiment_pure_summary.csv",
            min_rows=500,
            command=f"uv run python -m agent_based_simulation.run pure --model {m_m} --load-calibrated --n-agents {n} --n-countries 10 --n-periods 50 --output-dir output/mistralai--mistral-small-creative-n{n} {MC}",
            group="agent_count",
            depends=["calibrate_mistral"],
        ))

    # ── Network robustness (k=8) ──────────────────────────────────
    # 10 countries × 100 periods = 1000 rows
    targets.append(Target(
        name="network_k8_mistral",
        file=f"network-k8/{s_m}/experiment_comm_summary.csv",
        min_rows=1000,
        command=f"uv run python -m agent_based_simulation.run comm --model {m_m} --load-calibrated --n-neighbors 8 --n-countries 10 --n-periods 100 --output-dir output/network-k8 {MC}",
        group="network",
        depends=["calibrate_mistral"],
    ))

    # ── Mixed models ──────────────────────────────────────────────
    # 5 countries × 50 periods = 250 rows per treatment
    mixed_models_str = " ".join(model(s) for s in ["llama", "gptoss", "qwen30", "minimax"])
    targets.append(Target(
        name="mixed_5model_pure",
        file=f"mixed-5model-pure/{s_m}/experiment_pure_summary.csv",
        min_rows=250,
        command=f"uv run python -m agent_based_simulation.run pure --model {m_m} --load-calibrated --mixed-models {mixed_models_str} --n-countries 5 --n-periods 50 --output-dir output/mixed-5model-pure {MC}",
        group="mixed_models",
        depends=["calibrate_mistral"],
    ))
    targets.append(Target(
        name="mixed_5model_comm",
        file=f"mixed-5model-comm/{s_m}/experiment_comm_summary.csv",
        min_rows=250,
        command=f"uv run python -m agent_based_simulation.run comm --model {m_m} --load-calibrated --mixed-models {mixed_models_str} --n-countries 5 --n-periods 50 --output-dir output/mixed-5model-comm {MC}",
        group="mixed_models",
        depends=["calibrate_mistral"],
    ))

    # ── Uncalibrated robustness ───────────────────────────────────
    # 5 countries × 20 periods = 100 rows per model
    for short in ["mistral", "llama", "qwen235"]:
        s, m = slug(short), model(short)
        targets.append(Target(
            name=f"uncalibrated_{short}",
            file=f"uncalibrated-robustness/{s}/experiment_pure_summary.csv",
            min_rows=100,
            command=f"uv run python -m agent_based_simulation.run pure --model {m} --n-countries 5 --n-periods 20 --output-dir output/uncalibrated-robustness {MC}",
            group="uncalibrated",
        ))

    # ── Beliefs: primary model (pure + comm + surveillance) ───────
    # Appended to main logs via --append. Each adds 10×20=200 rows.
    # Cumulative: pure = 200 base + 200 beliefs = 400
    #             comm = 200 base + 200 beliefs = 400 + 200 surv = 600
    targets.append(Target(
        name="beliefs_pure_mistral",
        file=f"{s_m}/experiment_pure_summary.csv",
        min_rows=400,
        command=f"uv run python -m agent_based_simulation.run pure --model {m_m} --load-calibrated --n-countries 10 --n-periods 20 --elicit-beliefs --elicit-second-order --belief-order pre --append {MC}",
        group="beliefs",
        depends=["calibrate_mistral", "pure_mistral"],
    ))
    targets.append(Target(
        name="beliefs_comm_mistral",
        file=f"{s_m}/experiment_comm_summary.csv",
        min_rows=400,
        command=f"uv run python -m agent_based_simulation.run comm --model {m_m} --load-calibrated --n-countries 10 --n-periods 20 --elicit-beliefs --elicit-second-order --append {MC}",
        group="beliefs",
        depends=["calibrate_mistral", "comm_mistral"],
    ))
    targets.append(Target(
        name="beliefs_surv_mistral",
        file=f"{s_m}/experiment_comm_summary.csv",
        min_rows=600,
        command=f"uv run python -m agent_based_simulation.run comm --model {m_m} --load-calibrated --n-countries 10 --n-periods 20 --surveillance --elicit-beliefs --elicit-second-order --append {MC}",
        group="beliefs",
        depends=["calibrate_mistral", "beliefs_comm_mistral"],
    ))

    # ── Fixed-messages surveillance test ──────────────────────────
    # 5 countries × 40 periods = 200 rows
    targets.append(Target(
        name="fixed_messages_surv_mistral",
        file=f"fixed-messages-surv/{s_m}/experiment_comm_summary.csv",
        min_rows=200,
        command=f"uv run python -m agent_based_simulation.run comm --model {m_m} --load-calibrated --surveillance --fixed-messages output/{s_m}/experiment_comm_log.json --n-countries 5 --n-periods 40 --output-dir output/fixed-messages-surv {MC}",
        group="fixed_messages",
        depends=["calibrate_mistral", "comm_mistral"],
    ))

    # ── Holdout validation ────────────────────────────────────────
    targets.append(Target(
        name="holdout_mistral",
        file=f"holdout-validation/{s_m}/autocalibrate_history.csv",
        min_rows=2,
        command=f"uv run python -m agent_based_simulation.run autocalibrate --model {m_m} --output-dir output/holdout-validation --holdout-fraction 0.3 --n-reps 8 --z-steps 21 --max-rounds 3 {MC}",
        group="holdout",
    ))

    # ── Group-size info ───────────────────────────────────────────
    # 5 countries × 20 periods = 100 rows per treatment
    for tx in ["pure", "comm"]:
        targets.append(Target(
            name=f"groupsize_{tx}_mistral",
            file=f"group-size-info/{s_m}/experiment_{tx}_summary.csv",
            min_rows=100,
            command=f"uv run python -m agent_based_simulation.run {tx} --model {m_m} --load-calibrated --group-size-info --n-countries 5 --n-periods 20 --output-dir output/group-size-info {MC}",
            group="group_size",
            depends=["calibrate_mistral"],
        ))

    # ── Surveillance variants (placebo, anonymous) ────────────────
    # 10 countries × 20 periods = 200 rows per variant
    for variant in ["placebo", "anonymous"]:
        targets.append(Target(
            name=f"surv_{variant}_mistral",
            file=f"{s_m}/_surveillance_{variant}_v2/{s_m}/experiment_comm_summary.csv",
            min_rows=200,
            command=f"uv run python -m agent_based_simulation.run comm --model {m_m} --load-calibrated --surveillance --surveillance-mode {variant} --n-countries 10 --n-periods 20 --output-dir output/{s_m}/_surveillance_{variant}_v2 {MC}",
            group="surv_variants",
            depends=["calibrate_mistral"],
        ))

    return targets


# ── Status display ────────────────────────────────────────────────────

def print_status(targets: list[Target], group_filter: str | None = None):
    groups: dict[str, list[Target]] = {}
    for t in targets:
        if group_filter and t.group != group_filter:
            continue
        groups.setdefault(t.group, []).append(t)

    total = done = missing = partial = 0
    for group_name in sorted(groups):
        print(f"\n{'─'*60}")
        print(f"  {group_name.upper()}")
        print(f"{'─'*60}")
        for t in groups[group_name]:
            s = t.status()
            marker = "  " if "OK" in s else (" ~" if "PARTIAL" in s else " X")
            print(f"  {marker} {t.name:45s} {s}")
            total += 1
            if "OK" in s:
                done += 1
            elif "PARTIAL" in s:
                partial += 1
            else:
                missing += 1

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total}  |  OK: {done}  |  PARTIAL: {partial}  |  MISSING: {missing}")
    print(f"{'='*60}")
    return missing + partial


# ── Runner ────────────────────────────────────────────────────────────

def run_targets(targets: list[Target], dry_run: bool = False,
                group_filter: str | None = None):
    # Build dependency-aware execution order
    by_name = {t.name: t for t in targets}
    to_run = []
    for t in targets:
        if group_filter and t.group != group_filter:
            continue
        if t.is_complete():
            continue
        # Check dependencies
        deps_ok = all(by_name[d].is_complete() for d in t.depends if d in by_name)
        if not deps_ok:
            # Find which deps are missing
            missing_deps = [d for d in t.depends if d in by_name and not by_name[d].is_complete()]
            # Add missing deps first (if not already queued)
            for d in missing_deps:
                dep_t = by_name[d]
                if dep_t not in to_run and not dep_t.is_complete():
                    to_run.append(dep_t)
        if t not in to_run:
            to_run.append(t)

    if not to_run:
        print("\nNothing to do — all targets are complete.")
        return

    print(f"\n{'='*60}")
    print(f"  {len(to_run)} target(s) to run")
    print(f"{'='*60}")

    for i, t in enumerate(to_run, 1):
        s = t.status()
        print(f"\n[{i}/{len(to_run)}] {t.name} ({s})")
        print(f"  $ {t.command}")
        if dry_run:
            continue
        try:
            result = subprocess.run(
                t.command, shell=True, cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            if result.returncode != 0:
                print(f"  FAILED (exit code {result.returncode})")
                print(f"  Stopping. Fix the error and re-run.")
                sys.exit(1)
            new_status = t.status()
            print(f"  Done: {new_status}")
        except KeyboardInterrupt:
            print("\n  Interrupted. Re-run to continue from where you left off.")
            sys.exit(130)


# ── Analysis pipeline ─────────────────────────────────────────────────

def run_pipeline(dry_run: bool = False):
    print(f"\n{'='*60}")
    print("  ANALYSIS PIPELINE")
    print(f"{'='*60}")
    cmds = [
        ("verify_paper_stats", "uv run python analysis/verify_paper_stats.py"),
        ("render_paper_tables", "uv run python analysis/render_paper_tables.py"),
        ("make_figures", "uv run python analysis/make_figures.py"),
        ("compile_paper", "cd paper && pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1; pdflatex -interaction=nonstopmode paper.tex 2>&1 | grep -e 'Output written' -e 'Error' -e 'undefined'; cd .."),
    ]
    for name, cmd in cmds:
        print(f"\n  {name}")
        print(f"  $ {cmd}")
        if not dry_run:
            subprocess.run(cmd, shell=True, cwd=ROOT)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smart experiment runner for the paper")
    parser.add_argument("--run", action="store_true", help="Execute missing experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--group", type=str, default=None, help="Only process targets in this group")
    parser.add_argument("--concurrency", type=int, default=200, help="Max concurrent API calls")
    parser.add_argument("--pipeline", action="store_true", help="Also run analysis pipeline after experiments")
    parser.add_argument("--pipeline-only", action="store_true", help="Only run analysis pipeline")
    args = parser.parse_args()

    if args.pipeline_only:
        run_pipeline(dry_run=args.dry_run)
        return

    targets = build_manifest(conc=args.concurrency)
    n_incomplete = print_status(targets, group_filter=args.group)

    if args.run or args.dry_run:
        run_targets(targets, dry_run=args.dry_run, group_filter=args.group)
        if args.pipeline:
            run_pipeline(dry_run=args.dry_run)
    elif n_incomplete > 0:
        print("\nRe-run with --run to execute, or --dry-run to preview commands.")


if __name__ == "__main__":
    main()
