from __future__ import annotations

import numpy as np

from paper2.audit_game import (
    GameConfig,
    TrainConfig,
    evaluate_protocol,
    init_params,
    probe_scores,
    run_batch,
    train_reinforce,
)
from paper2.protocol_suite import ProtocolConfig, cross_play_matrix, evaluate_protocol_arm
from paper2.ppo_agents import PPOConfig, evaluate_ppo_metrics, train_ippo


def test_run_batch_shapes() -> None:
    config = GameConfig(n_agents=4, k_messages=3)
    rng = np.random.default_rng(123)
    params = init_params(config, rng)
    batch = run_batch(params, config, rng, 20, comm_mode="learned")

    assert batch.signals.shape == (20, 4)
    assert batch.messages.shape == (20, 4)
    assert batch.receiver_features.shape == (20, 4, 10)
    assert batch.actions.shape == (20, 4)
    assert np.isfinite(batch.raw_payoffs).all()


def test_reinforce_smoke() -> None:
    config = GameConfig(n_agents=4, k_messages=3)
    train_config = TrainConfig(steps=8, batch_size=32, learning_rate=0.02)
    params, history = train_reinforce(config, train_config, seed=5, comm_mode="learned")
    metrics = evaluate_protocol(params, config, n_episodes=100, comm_mode="learned")

    assert len(history) >= 2
    assert np.isfinite(metrics["welfare"])
    assert "message_value" in metrics


def test_probe_scores_smoke() -> None:
    config = GameConfig(n_agents=4, k_messages=3)
    rng = np.random.default_rng(321)
    params = init_params(config, rng)
    scores = probe_scores(params, config, n_episodes=120, comm_mode="semantic")

    assert 0.0 <= scores["auditor_balanced_accuracy"] <= 1.0
    assert 0.0 <= scores["receiver_balanced_accuracy"] <= 1.0
    assert scores["decodability_gap"] >= 0.0


def test_protocol_suite_separates_public_and_private_codes() -> None:
    config = ProtocolConfig(n_agents=5)
    canonical = evaluate_protocol_arm(config, arm="canonical_semantic", seed=0, n_episodes=500)
    private = evaluate_protocol_arm(config, arm="private_code", seed=4, n_episodes=500)

    assert canonical["auditor_decodability"] > 0.95
    assert private["receiver_decodability"] > 0.95
    assert private["auditor_decodability"] < private["receiver_decodability"]


def test_protocol_cross_play_smoke() -> None:
    config = ProtocolConfig(n_agents=5)
    frame = cross_play_matrix(config, arm="private_code", seeds=[0, 1, 2], n_episodes=200)

    assert set(frame.columns) == {"sender_seed", "receiver_seed", "welfare", "message_value"}
    assert len(frame) == 9


def test_ppo_smoke() -> None:
    config = GameConfig(n_agents=4, k_messages=3, sigma=0.35)
    ppo_config = PPOConfig(
        updates=2,
        batch_size=64,
        minibatch_size=64,
        epochs=1,
        hidden_dim=16,
        device="cpu",
    )
    policy, history = train_ippo(config, ppo_config, seed=7, comm_mode="learned")
    metrics = evaluate_ppo_metrics(policy, seed=8, n_episodes=100, comm_mode="learned")

    assert len(history) >= 2
    assert np.isfinite(metrics["welfare"])
    assert "message_value" in metrics
