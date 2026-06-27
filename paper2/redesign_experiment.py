"""Unified monitor blind-spot experiment for Paper 2.

Thesis: a monitor is blinded *for free* whenever it cannot observe something the
agents share -- a message subspace (structural blind spot) or a shared key
(informational blind spot). Only when the monitor sees everything the agents
condition on does monitoring stay reliable.

Conditions form a ladder:
  1. blind_spot      monitor sees only a projection z=Am; receiver sees full m.
  2. full_*          monitor sees the entire message (no structural blind spot).
  3. key_*           full-visibility monitor, but agents share a key it cannot see.

Each is run with and without monitor pressure (an adversarial detector the
sender is penalized by), plus a strong-monitor check on the key condition.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paper2.continuous_audit_game import (
    ContinuousGameConfig,
    ContinuousPPOConfig,
    evaluate_metrics,
    resolve_device,
    train_continuous_policy,
)


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    monitor_full_message: bool
    key_dim: int
    active_monitor_cost: float
    monitor_updates: int = 1
    receiver_observation: str = "full"


def default_conditions() -> list[Condition]:
    return [
        Condition("no_comm", "No communication", True, 0, 0.0, receiver_observation="none"),
        Condition("blind_spot", "Structural blind spot", False, 0, 0.12),
        Condition("full_passive", "Full visibility, no pressure", True, 0, 0.0),
        Condition("full_adversarial", "Full visibility + pressure", True, 0, 0.12),
        Condition("key_passive", "Shared key, no pressure", True, 6, 0.0),
        Condition("key_adversarial", "Shared key + pressure", True, 6, 0.12),
        Condition("key_strong_monitor", "Shared key + strong monitor", True, 6, 0.12, monitor_updates=8),
    ]


def run(conditions, *, seeds, updates, device, batch, eval_episodes) -> pd.DataFrame:
    rows: list[dict] = []
    for cond in conditions:
        for seed in seeds:
            game = ContinuousGameConfig(sigma=0.35, message_dim=6, auditor_dim=2, key_dim=cond.key_dim)
            passive = cond.active_monitor_cost == 0.0 and cond.name != "no_comm"
            ppo = ContinuousPPOConfig(
                updates=updates, batch_size=batch, minibatch_size=batch, epochs=3,
                device=device, learning_rule="ppo",
                active_monitor_cost=cond.active_monitor_cost,
                monitor_full_message=cond.monitor_full_message,
                monitor_passive=passive,
                monitor_updates=cond.monitor_updates,
                monitor_learning_rate=5e-4,
            )
            policy, _ = train_continuous_policy(
                game, ppo, seed=seed, receiver_observation=cond.receiver_observation
            )
            m = evaluate_metrics(policy, seed=900 + seed, n_episodes=eval_episodes)
            # Forensic probe restricted to the monitor's own access.
            restricted = (
                m["full_content_balanced_accuracy"]
                if cond.monitor_full_message
                else m["auditor_content_balanced_accuracy"]
            )
            row = {
                "condition": cond.name,
                "label": cond.label,
                "seed": seed,
                "coordination": m["message_value"],
                "monitor_live": m.get("live_monitor_content_balanced_accuracy", float("nan")),
                "forensic_restricted": restricted,
                "oracle_full_message": m["full_content_balanced_accuracy"],
                "oracle_with_key": m.get("oracle_key_content_balanced_accuracy", float("nan")),
                "hidden_causal_influence": m["hidden_causal_influence"],
            }
            rows.append(row)
            print(
                f"{cond.name:20s} seed={seed} coord={row['coordination']:.3f} "
                f"monitor_live={row['monitor_live']:.3f} restricted={restricted:.3f} "
                f"oracle_key={row['oracle_with_key']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["coordination", "monitor_live", "forensic_restricted", "oracle_full_message", "oracle_with_key"]
    g = df.groupby(["condition", "label"], sort=False)[metrics]
    summary = g.agg(["mean", "sem"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("paper2/output/redesign"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--eval-episodes", type=int, default=5000)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"device={device} seeds={args.seeds} updates={args.updates}", flush=True)
    df = run(
        default_conditions(),
        seeds=args.seeds,
        updates=args.updates,
        device=device,
        batch=args.batch,
        eval_episodes=args.eval_episodes,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps({"seeds": args.seeds, "updates": args.updates, "device": device}, indent=2)
    )
    print("\n=== summary (means) ===", flush=True)
    print(summary[[
        "condition", "coordination_mean", "monitor_live_mean",
        "forensic_restricted_mean", "oracle_with_key_mean",
    ]].to_string(index=False), flush=True)
    print(f"\nWrote {args.outdir/'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
