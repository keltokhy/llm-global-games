from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from paper2.continuous_audit_game import (
    ContinuousGameConfig,
    ContinuousPPOConfig,
    decompose_messages,
    evaluate_metrics,
    evaluate_policy,
    make_projection,
    train_continuous_policy,
)


def test_projection_decomposition_reconstructs_message() -> None:
    config = ContinuousGameConfig(message_dim=5, auditor_dim=2)
    projection = make_projection(config, "cpu")
    messages = torch.randn(11, 4, config.message_dim)

    _, visible, hidden = decompose_messages(messages, projection)

    assert torch.allclose(messages, visible + hidden, atol=1e-6)
    assert torch.allclose(hidden @ projection.T, torch.zeros(11, 4, config.auditor_dim), atol=1e-5)


def test_hidden_interventions_preserve_auditor_projection() -> None:
    config = ContinuousGameConfig(n_agents=4, message_dim=4, auditor_dim=2)
    ppo = ContinuousPPOConfig(updates=1, batch_size=64, minibatch_size=64, epochs=1, hidden_dim=16, device="cpu")
    policy, _ = train_continuous_policy(config, ppo, seed=3, receiver_observation="full")

    normal = evaluate_policy(policy, seed=44, n_episodes=80, intervention="none")
    zero_hidden = evaluate_policy(policy, seed=44, n_episodes=80, intervention="zero_hidden")
    shuffled_hidden = evaluate_policy(policy, seed=44, n_episodes=80, intervention="shuffle_hidden")

    assert torch.allclose(normal.auditor_observation, zero_hidden.auditor_observation, atol=1e-5)
    assert torch.allclose(normal.auditor_observation, shuffled_hidden.auditor_observation, atol=1e-5)


def test_continuous_ppo_smoke_returns_finite_metrics() -> None:
    config = ContinuousGameConfig(n_agents=4, message_dim=4, auditor_dim=2, sigma=0.35)
    ppo = ContinuousPPOConfig(updates=2, batch_size=96, minibatch_size=96, epochs=1, hidden_dim=16, device="cpu")
    policy, history = train_continuous_policy(config, ppo, seed=7, receiver_observation="full")
    metrics = evaluate_metrics(policy, seed=8, n_episodes=180)

    assert len(history) >= 2
    assert np.isfinite(metrics["welfare"])
    assert np.isfinite(metrics["message_value"])
    assert np.isfinite(metrics["hidden_causal_influence"])
    assert "probe_strategic_opacity_gap" in metrics


def test_visible_bottleneck_has_no_hidden_action_influence_smoke() -> None:
    config = ContinuousGameConfig(n_agents=4, message_dim=4, auditor_dim=2)
    ppo = ContinuousPPOConfig(updates=1, batch_size=64, minibatch_size=64, epochs=1, hidden_dim=16, device="cpu")
    policy, _ = train_continuous_policy(config, ppo, seed=9, receiver_observation="visible")
    metrics = evaluate_metrics(policy, seed=10, n_episodes=160)

    assert metrics["hidden_causal_influence"] < 1e-6


def test_policy_can_be_evaluated_under_shifted_primitives() -> None:
    config = ContinuousGameConfig(n_agents=4, message_dim=4, auditor_dim=2, sigma=0.35)
    shifted = replace(config, sigma=0.55, theta_low=0.05, theta_high=1.65)
    ppo = ContinuousPPOConfig(updates=1, batch_size=64, minibatch_size=64, epochs=1, hidden_dim=16, device="cpu")
    policy, _ = train_continuous_policy(config, ppo, seed=11, receiver_observation="full")
    metrics = evaluate_metrics(policy, seed=12, n_episodes=120, config_override=shifted)

    assert np.isfinite(metrics["welfare"])
    assert np.isfinite(metrics["message_value"])
    assert np.isfinite(metrics["hidden_causal_influence"])


def test_reinforce_learning_rule_smoke_returns_finite_metrics() -> None:
    config = ContinuousGameConfig(n_agents=4, message_dim=4, auditor_dim=2, sigma=0.35)
    reinforce = ContinuousPPOConfig(
        updates=2,
        batch_size=96,
        minibatch_size=96,
        epochs=1,
        hidden_dim=16,
        learning_rule="reinforce",
        device="cpu",
    )
    policy, history = train_continuous_policy(config, reinforce, seed=13, receiver_observation="full")
    metrics = evaluate_metrics(policy, seed=14, n_episodes=180)

    assert len(history) >= 2
    assert np.isfinite(metrics["welfare"])
    assert np.isfinite(metrics["message_value"])
    assert np.isfinite(metrics["hidden_causal_influence"])


def test_active_monitor_training_returns_live_monitor_metrics() -> None:
    config = ContinuousGameConfig(n_agents=4, message_dim=4, auditor_dim=2, sigma=0.35)
    monitored = ContinuousPPOConfig(
        updates=2,
        batch_size=96,
        minibatch_size=96,
        epochs=1,
        hidden_dim=16,
        active_monitor_cost=0.10,
        monitor_updates=1,
        device="cpu",
    )
    policy, history = train_continuous_policy(config, monitored, seed=15, receiver_observation="full")
    metrics = evaluate_metrics(policy, seed=16, n_episodes=180)

    assert policy.monitor is not None
    assert len(history) >= 2
    assert np.isfinite(metrics["live_monitor_content_accuracy"])
    assert np.isfinite(metrics["live_monitor_true_probability"])
