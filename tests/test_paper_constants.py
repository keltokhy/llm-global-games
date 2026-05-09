"""Regression checks for hand-written theory constants in the paper."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_based_simulation.briefing import _compute_sliders


ROOT = Path(__file__).resolve().parent.parent
PAPER_TEX = ROOT / "paper" / "paper.tex"


def _paper_tex() -> str:
    return PAPER_TEX.read_text()


def test_payoff_sweep_mapping_matches_theta_star_formula():
    rows = re.findall(
        r"\$\\theta\^\* = (?P<theta>\d+\.\d+)\$ & "
        r"(?P<benefit>\d+\.\d+) & (?P<cost>\d+) & --- \\\\",
        _paper_tex(),
    )

    assert len(rows) == 7
    for theta_raw, benefit_raw, cost_raw in rows:
        theta_star = float(theta_raw)
        expected_benefit = round(theta_star / (1.0 - theta_star), 2)
        assert float(benefit_raw) == expected_benefit
        assert cost_raw == "1"


def test_slider_table_matches_briefing_generator_defaults():
    rows = re.findall(
        r"\$(?P<z>(?:[-+]\d+\.\d|\\phantom\{-\}0\.0))\$ & "
        r"(?P<direction>\d+\.\d+) & "
        r"(?P<clarity>\d+\.\d+) & "
        r"(?P<coordination>\d+\.\d+) \\\\",
        _paper_tex(),
    )

    assert len(rows) == 5
    for z_raw, direction_raw, clarity_raw, coordination_raw in rows:
        z = 0.0 if "phantom" in z_raw else float(z_raw)
        direction, clarity, coordination = _compute_sliders(z)

        assert float(direction_raw) == round(direction, 2)
        assert float(clarity_raw) == round(clarity, 2)
        assert float(coordination_raw) == round(coordination, 2)
