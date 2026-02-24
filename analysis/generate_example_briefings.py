"""
Generate example briefings at z-scores {-2, 0, +2} for paper inclusion.

Prints rendered briefing text and the surveillance communication prompt.

Usage:
    uv run python analysis/generate_example_briefings.py
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so agent_based_simulation is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_based_simulation.briefing import BriefingGenerator
from agent_based_simulation.experiment import SYSTEM_COMMUNICATE_SURVEILLED


def main():
    gen = BriefingGenerator()

    z_scores = [-2.0, 0.0, 2.0]
    labels = ["weak regime (z = -2.0)", "borderline (z = 0.0)", "strong regime (z = +2.0)"]

    for z, label in zip(z_scores, labels):
        briefing = gen.generate(z_score=z, agent_id=0, period=0)
        print("=" * 72)
        print(f"EXAMPLE BRIEFING: {label}")
        print(f"  direction={briefing.direction:.3f}  clarity={briefing.clarity:.3f}  coordination={briefing.coordination:.3f}")
        print("=" * 72)
        print(briefing.render())
        print()

    print("=" * 72)
    print("SURVEILLANCE COMMUNICATION PROMPT")
    print("=" * 72)
    print(SYSTEM_COMMUNICATE_SURVEILLED)


if __name__ == "__main__":
    main()
