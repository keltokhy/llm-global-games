"""Tests for the CLEAN_RUN manifest and schema layer."""

import sys
import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "CLEAN_RUN" / "code"))
sys.path.insert(0, str(ROOT))

from clean_run.config import load_manifest, validate_manifest
from clean_run.message_bank import (
    balance_table,
    build_source_bank_from_manifest,
    combine_bank_batches,
    derive_original_replay_bank,
    manual_audit_sample,
    promote_valid_bank,
    rewrite_direct_coded_bank,
    summary_stats,
    validate_bank,
)
from clean_run.schema import MESSAGE_BANK_COLUMNS
from clean_run.power import build_power_analysis_status
from clean_run.schema import AGENT_COLUMNS, PERIOD_COLUMNS
from clean_run.figures import render_available_figures
from clean_run.artifact_manifest import build_artifact_manifest
from clean_run.analyze import MatchedComparisonError, _paired_effect
from clean_run.tables import INFO_STATE_COLUMNS, arm_sample_size_table, information_state_table, message_bank_qc_table


def test_clean_run_manifests_validate():
    for name in ("pilot.yaml", "main.yaml"):
        manifest = load_manifest(ROOT / "CLEAN_RUN" / "plans" / name)
        assert validate_manifest(manifest) == []


def test_sender_side_surveillance_has_no_receiver_warning():
    manifest = load_manifest(ROOT / "CLEAN_RUN" / "plans" / "main.yaml")
    sender_only = [
        arm for arm in manifest.arms
        if arm["message_stage_context"] == "surveillance_full"
        and arm["claim"] in {"sender_side_surveillance", "pre_decision_belief_mechanism"}
    ]
    assert sender_only
    assert all(arm["decision_context"] == "none" for arm in sender_only)


def test_clean_schema_contains_required_plan_columns():
    for column in (
        "belief_pre_shared_understanding",
        "belief_pre_others_expect_join",
        "prompt_hash",
        "response_hash",
    ):
        assert column in AGENT_COLUMNS
    for column in (
        "message_stage_context",
        "decision_context",
        "decision_task",
        "message_source_arm",
    ):
        assert column in PERIOD_COLUMNS


def test_main_manifest_covers_frozen_roster_for_core_families():
    manifest = load_manifest(ROOT / "CLEAN_RUN" / "plans" / "main.yaml")
    models = {item["model"] for item in manifest.raw["model_roster"]}
    signal_models = {arm["model"] for arm in manifest.arms if arm["claim"] == "language_signal"}
    surveillance_models = {
        arm["model"] for arm in manifest.arms if arm["claim"] == "sender_side_surveillance"
    }
    assert signal_models == models
    assert surveillance_models == models
    assert len([arm for arm in manifest.arms if arm["claim"] == "language_signal"]) == 18
    assert len([arm for arm in manifest.arms if arm["claim"] == "sender_side_surveillance"]) == 12


def test_information_state_table_is_manifest_derived():
    table = information_state_table(ROOT / "CLEAN_RUN" / "plans" / "main.yaml")
    assert list(table.columns) == INFO_STATE_COLUMNS
    assert not table.empty
    sender_only = table[table["arm"].str.contains("surv_sender_only")]
    assert not sender_only.empty
    assert set(sender_only["sender_sees_monitoring_warning"]) == {"yes"}
    assert set(sender_only["receiver_sees_monitoring_warning"]) == {"no"}


def test_rewrite_pairs_refuses_accidental_source_overwrite():
    path = ROOT / "CLEAN_RUN" / "message_banks" / "direct_coded_pairs.parquet"
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        asyncio.run(
            rewrite_direct_coded_bank(
                path,
                path,
                model="mistralai/mistral-small-2603",
            )
        )


def test_promote_bank_rejects_failed_qc(tmp_path):
    candidate = ROOT / "CLEAN_RUN" / "message_banks" / "direct_coded_pairs.parquet"
    target = tmp_path / "direct_coded_pairs.parquet"
    with pytest.raises(RuntimeError, match="failed QC"):
        promote_valid_bank(candidate, target)
    assert not target.exists()


def test_rewrite_pairs_allows_explicit_overwrite_to_reach_dependency_check():
    path = ROOT / "CLEAN_RUN" / "message_banks" / "direct_coded_pairs.parquet"
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        asyncio.run(
            rewrite_direct_coded_bank(
                path,
                path,
                model="mistralai/mistral-small-2603",
                allow_overwrite=True,
            )
        )


def test_message_bank_balance_and_manual_sample_are_real_rows():
    path = ROOT / "CLEAN_RUN" / "message_banks" / "direct_coded_pairs.parquet"
    import pandas as pd

    bank = pd.read_parquet(path)
    balance = balance_table(bank)
    sample = manual_audit_sample(bank, n=50, seed=5150)
    assert len(balance) >= 8
    assert "standardized_difference" in balance.columns
    assert len(sample) == 50
    assert sample["message_id"].notna().all()
    assert "manual_same_regime_strength" in sample.columns


def test_direct_coded_qc_scores_each_side_from_text():
    import pandas as pd

    row = {column: pd.NA for column in MESSAGE_BANK_COLUMNS}
    row.update(
        {
            "message_id": "qc-demo",
            "source_arm_id": "unit",
            "country": 0,
            "period": 0,
            "sender_agent_id": 0,
            "theta": 0.2,
            "sender_signal": 0.1,
            "sender_z_score": -1.0,
            "original_message": "source",
            "direct_message": "The regime is weak; security is divided; citizens may join in the streets now.",
            "coded_message": "The weather is changing; market signals are divided and neighbors may open doors now.",
            "valence": "weakness_indicating",
            "sentiment_score": 0.0,
            "hedge_density": 0.0,
            "specificity_score": 0.5,
            "verbosity_tokens": 13,
            "syntactic_complexity": 0.0,
            "first_order_similarity": 0.95,
            "directness_score": 0.0,
            "codedness_score": 0.0,
            "risk_salience_score": 0.0,
            "factual_equivalence_pass": True,
            "style_balance_pass": True,
            "qc_notes": "unit test",
        }
    )
    assert validate_bank(pd.DataFrame([row]))["pass"] is True

    row["coded_message"] = "The regime is weak; security is divided; citizens may join in the streets now."
    assert validate_bank(pd.DataFrame([row]))["pass"] is False


def test_manifest_source_and_derived_original_banks_validate(tmp_path):
    source = tmp_path / "source.parquet"
    original = tmp_path / "original.parquet"
    df = build_source_bank_from_manifest(
        ROOT / "CLEAN_RUN" / "plans" / "pilot.yaml",
        source,
        arm_id="pilot_direct_replay_mistral",
        max_cells=1,
        max_agents=2,
    )
    assert len(df) == 2
    assert source.exists()
    assert validate_bank(df)["pass"] is True

    derived = derive_original_replay_bank(source, original, text_column="original_message")
    assert original.exists()
    assert len(derived) == 2
    assert validate_bank(derived)["pass"] is True

    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    derived.iloc[:1].to_parquet(batch_dir / "batch_00000.parquet", index=False)
    derived.iloc[1:].to_parquet(batch_dir / "batch_00001.parquet", index=False)
    combined = combine_bank_batches(batch_dir, tmp_path / "combined.parquet")
    assert len(combined) == 2
    assert validate_bank(combined)["pass"] is True


def test_message_bank_summary_stats_include_qc_and_counts():
    import pandas as pd

    path = ROOT / "CLEAN_RUN" / "message_banks" / "baseline_messages.parquet"
    stats = summary_stats(pd.read_parquet(path))
    assert stats["rows"] == 25000
    assert stats["qc"]["pass"] is True
    assert "valence" in stats["counts"]
    assert "theta" in stats["numeric"]


def test_power_analysis_status_is_honest_dependency_artifact():
    status = build_power_analysis_status(ROOT / "CLEAN_RUN" / "plans" / "main.yaml")
    assert status["status"] == "dependency_needed"
    assert "CLEAN_RUN/output/pilot/**/periods.parquet for cross-task pilot arms" in status["dependency_needed"]
    assert status["declared_arm_counts"]["cross_task_decomposition"] == 8


def test_table_render_inputs_cover_manifest_and_message_banks():
    arms = arm_sample_size_table(ROOT / "CLEAN_RUN" / "plans" / "main.yaml")
    banks = message_bank_qc_table(ROOT / "CLEAN_RUN" / "message_banks")
    assert len(arms) == 75
    assert set(["arm_id", "claim", "expected_rows"]).issubset(arms.columns)
    assert set(banks["message_bank"]) >= {
        "baseline_messages.parquet",
        "direct_coded_pairs.parquet",
    }


def test_available_figures_render_without_live_outputs(tmp_path):
    paths = render_available_figures(ROOT / "CLEAN_RUN" / "plans" / "main.yaml", tmp_path)
    assert any(path.name == "fig_manifest_arm_counts.png" for path in paths)
    assert any(path.name == "fig_direct_coded_balance.png" for path in paths)
    assert (tmp_path / "figure_status.json").exists()


def test_artifact_manifest_hashes_clean_run_files():
    manifest = build_artifact_manifest(ROOT / "CLEAN_RUN")
    assert {"path", "bytes", "sha256"}.issubset(manifest.columns)
    assert "plans/main.yaml" in set(manifest["path"])
    assert "artifacts/artifact_manifest.tsv" not in set(manifest["path"])
    assert manifest["sha256"].str.len().eq(64).all()


def test_paired_effect_stops_below_95pct_exact_overlap():
    import pandas as pd

    base = {
        "model": "m",
        "country": 0,
        "z_public": 0.0,
        "benefit": 1.0,
        "cost": 1.0,
        "theta_star": 0.5,
        "join_fraction_valid": 0.5,
    }
    rows = []
    for i in range(20):
        rows.append({"arm_id": "control", "period": i, "theta": float(i), **base})
        theta = float(i) if i < 18 else float(i + 100)
        rows.append({"arm_id": "treat", "period": i, "theta": theta, **base})
    with pytest.raises(MatchedComparisonError, match="below required"):
        _paired_effect(pd.DataFrame(rows), "treat", "control")
