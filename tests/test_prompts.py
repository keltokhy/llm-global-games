"""Regression tests for experiment prompt isolation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_based_simulation.experiment import (
    SURVEILLANCE_WARNING_ANONYMOUS,
    SURVEILLANCE_WARNING_FULL,
    SURVEILLANCE_WARNING_PLACEBO,
    SYSTEM_COMMUNICATE,
    SYSTEM_COMMUNICATE_SURVEILLED,
    SYSTEM_COMMUNICATE_SURVEILLED_ANONYMOUS,
    SYSTEM_COMMUNICATE_SURVEILLED_PLACEBO,
)


def test_surveillance_prompt_is_baseline_plus_warning():
    assert SYSTEM_COMMUNICATE_SURVEILLED == (
        SYSTEM_COMMUNICATE + "\n\n" + SURVEILLANCE_WARNING_FULL
    )
    assert "trusted people" in SYSTEM_COMMUNICATE_SURVEILLED
    assert "Be natural" in SYSTEM_COMMUNICATE_SURVEILLED


def test_surveillance_variants_are_baseline_plus_one_appendix_sentence():
    assert SYSTEM_COMMUNICATE_SURVEILLED_PLACEBO == (
        SYSTEM_COMMUNICATE + "\n\n" + SURVEILLANCE_WARNING_PLACEBO
    )
    assert SYSTEM_COMMUNICATE_SURVEILLED_ANONYMOUS == (
        SYSTEM_COMMUNICATE + "\n\n" + SURVEILLANCE_WARNING_ANONYMOUS
    )
