"""Unit tests for the regime-analyst pilot (no API calls)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_based_simulation.analyst import (
    CellItem,
    NESTED_BASELINE_LOG,
    NESTED_SURVEILLANCE_LOG,
    Sender,
    build_analyst_prompt,
    load_nested_cells,
    parse_analyst_response,
    sample_cells,
    _shuffle_senders,
)


# ── Parser ────────────────────────────────────────────────────────────


def test_parse_clean_response():
    text = (
        "FALL: 70\n"
        "JOIN_PERCENT: 60\n"
        "SENDER 1: JOIN 85\n"
        "SENDER 2: STAY 20\n"
        "SENDER 3: JOIN 90\n"
    )
    out = parse_analyst_response(text, 3)
    assert out["fall_est"] == 70.0
    assert out["join_pct_est"] == 60.0
    assert out["parse_ok_fall"] and out["parse_ok_join_pct"]
    assert [s["verdict"] for s in out["senders"]] == ["JOIN", "STAY", "JOIN"]
    assert [s["p_join"] for s in out["senders"]] == [85.0, 20.0, 90.0]
    assert not any(s["p_imputed"] for s in out["senders"])
    assert not out["api_error"]


def test_parse_markdown_wrapped():
    text = (
        "**FALL:** 40\n"
        "**JOIN_PERCENT:** 35\n"
        "**SENDER 1:** STAY 15\n"
        "- SENDER 2: JOIN 80\n"
    )
    out = parse_analyst_response(text, 2)
    assert out["fall_est"] == 40.0
    assert out["join_pct_est"] == 35.0
    assert out["senders"][0]["verdict"] == "STAY"
    assert out["senders"][1]["p_join"] == 80.0


def test_parse_truncated_keeps_cell_level():
    text = "FALL: 55\nJOIN_PERCENT: 50\nSENDER 1: JOIN 7"
    out = parse_analyst_response(text, 25)
    assert out["fall_est"] == 55.0
    assert out["join_pct_est"] == 50.0
    assert sum(s["parse_ok"] for s in out["senders"]) == 1
    assert out["senders"][0]["p_join"] == 7.0


def test_parse_missing_probability_imputed():
    text = "FALL: 10\nJOIN_PERCENT: 20\nSENDER 1: JOIN\nSENDER 2: STAY\n"
    out = parse_analyst_response(text, 2)
    assert out["senders"][0] == {
        "verdict": "JOIN", "p_join": 75.0, "p_imputed": True, "parse_ok": True,
    }
    assert out["senders"][1]["p_join"] == 25.0 and out["senders"][1]["p_imputed"]


def test_parse_number_only_sender_line():
    text = "FALL: 30\nJOIN_PERCENT: 40\nSENDER 1: 80\nSENDER 2: 10\n"
    out = parse_analyst_response(text, 2)
    assert out["senders"][0]["verdict"] == "JOIN"
    assert out["senders"][1]["verdict"] == "STAY"
    assert not out["senders"][0]["p_imputed"]


def test_parse_prose_prefix_fallback():
    text = (
        "Here is my assessment.\n"
        "Based on the messages, FALL is around 65 percent.\n"
        "JOIN_PERCENT is 55.\n"
    )
    out = parse_analyst_response(text, 3)
    assert out["fall_est"] == 65.0
    assert out["join_pct_est"] == 55.0


def test_parse_api_error():
    out = parse_analyst_response("[API Error: boom]", 5)
    assert out["api_error"]
    assert out["fall_est"] is None
    assert not any(s["parse_ok"] for s in out["senders"])


def test_parse_clamps_out_of_range():
    out = parse_analyst_response("FALL: 250\nJOIN_PERCENT: 90\n", 1)
    assert out["fall_est"] == 100.0


# ── Prompt blinding ───────────────────────────────────────────────────


def _make_item(arm: str = "baseline") -> CellItem:
    senders = [
        Sender(agent_id=3, message="The harvest looks thin this year.", true_decision="STAY"),
        Sender(agent_id=7, message="People I trust are ready to move tonight.", true_decision="JOIN"),
    ]
    return CellItem(
        corpus="nested", arm=arm, country=1, period=2, theta=-0.731,
        theta_star=0.5, coup_success=True, senders=senders,
    )


def test_prompt_blinding():
    for arm in ("baseline", "surveillance"):
        system, user = build_analyst_prompt(_make_item(arm))
        combined = (system + "\n" + user).lower()
        assert "surveillance" not in combined
        assert "baseline" not in combined
        assert "warning" not in combined
        assert "-0.731" not in combined          # theta never shown
        assert "reasoning" not in combined       # sender reasoning never shown
        assert "briefing" not in combined
        assert 'Sender 1: "' in user and 'Sender 2: "' in user


def test_prompt_identical_template_across_arms():
    base_sys, base_user = build_analyst_prompt(_make_item("baseline"))
    surv_sys, surv_user = build_analyst_prompt(_make_item("surveillance"))
    assert base_sys == surv_sys
    assert base_user == surv_user  # same messages => identical prompt; only text varies


def test_shuffle_deterministic_per_arm():
    senders = [Sender(agent_id=i, message=f"m{i}", true_decision="STAY") for i in range(10)]
    a = _shuffle_senders(list(senders), 4, 9, "baseline")
    b = _shuffle_senders(list(senders), 4, 9, "baseline")
    c = _shuffle_senders(list(senders), 4, 9, "surveillance")
    assert [s.agent_id for s in a] == [s.agent_id for s in b]
    assert [s.agent_id for s in a] != list(range(10)) or [s.agent_id for s in c] != list(range(10))


# ── Assembly / pairing ────────────────────────────────────────────────


def _fixture_entry(country, period, theta, decisions, arm):
    agents = []
    for i, dec in enumerate(decisions):
        agents.append({
            "id": i,
            "signal": theta + 0.01 * i,
            "z_score": 0.0,
            "decision": dec,
            "api_error": False,
            "message_sent": f"{arm} message from agent {i}",
            "reasoning": "SECRET-{}".format(dec),
        })
    return {
        "country": country, "period": period, "theta": theta, "theta_star": 0.5,
        "coup_success": theta < 0, "agents": agents,
    }


def test_load_nested_cells_pairs_and_filters(tmp_path):
    decisions = ["JOIN"] * 12 + ["STAY"] * 13
    base = [
        _fixture_entry(0, 0, -1.0, decisions, "base"),
        _fixture_entry(0, 1, 0.8, decisions, "base"),
    ]
    surv = [
        _fixture_entry(0, 0, -1.0, decisions, "surv"),
        _fixture_entry(0, 1, 0.8, decisions, "surv"),
    ]
    # Knock out enough surveillance agents in cell (0,1) to drop it below threshold.
    for a in surv[1]["agents"][:10]:
        a["api_error"] = True

    for rel, payload in ((NESTED_BASELINE_LOG, base), (NESTED_SURVEILLANCE_LOG, surv)):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))

    pairs = load_nested_cells(tmp_path)
    assert len(pairs) == 1  # cell (0,1) dropped (<20 usable in surveillance arm)
    b_item, s_item = pairs[0]
    assert (b_item.arm, s_item.arm) == ("baseline", "surveillance")
    assert b_item.theta == s_item.theta == -1.0
    assert {x.agent_id for x in b_item.senders} == {x.agent_id for x in s_item.senders}
    assert b_item.jf_true_shown == pytest.approx(12 / 25)
    assert all("surv" in x.message for x in s_item.senders)
    assert all("base" in x.message for x in b_item.senders)


def test_load_nested_cells_theta_mismatch_raises(tmp_path):
    decisions = ["JOIN"] * 25
    base = [_fixture_entry(0, 0, -1.0, decisions, "base")]
    surv = [_fixture_entry(0, 0, -0.5, decisions, "surv")]
    for rel, payload in ((NESTED_BASELINE_LOG, base), (NESTED_SURVEILLANCE_LOG, surv)):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="theta mismatch"):
        load_nested_cells(tmp_path)


def test_sample_cells_stratified_deterministic():
    pairs = []
    for k in range(100):
        item = _make_item()
        item.theta = -2.0 + 0.04 * k
        pairs.append((item, item))
    s1 = sample_cells(pairs, 20, seed=5150)
    s2 = sample_cells(pairs, 20, seed=5150)
    assert len(s1) == 20
    assert [p[0].theta for p in s1] == [p[0].theta for p in s2]
    thetas = [p[0].theta for p in s1]
    assert min(thetas) < -1.5 and max(thetas) > 1.5  # spans the theta range
