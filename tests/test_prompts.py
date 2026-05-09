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
    SYSTEM_DECIDE_COMM,
    _system_decide_comm,
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


def test_clean_decision_prompt_has_no_surveillance_warning():
    assert SYSTEM_DECIDE_COMM == _system_decide_comm(decision_context="none")
    assert "Use your private briefing and messages from trusted contacts" in SYSTEM_DECIDE_COMM

    forbidden = (
        "monitored",
        "security services",
        "self-censoring",
        "serious consequences",
        "research purposes",
        "aggregated anonymously",
    )
    for phrase in forbidden:
        assert phrase not in SYSTEM_DECIDE_COMM


def test_decision_context_warning_is_explicit_opt_in():
    surveilled_decision = _system_decide_comm(decision_context="full")

    assert "monitored by regime security services" in surveilled_decision
    assert "self-censoring" in surveilled_decision
    assert surveilled_decision != SYSTEM_DECIDE_COMM
