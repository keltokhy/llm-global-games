"""Shared-parameter IPPO/PPO agents for the Paper 2 auditability game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from paper2.audit_game import (
    AuditShaping,
    Batch,
    CommMode,
    GameConfig,
    PolicyParams,
    _private_features,
    audit_message_cost,
    combine_sender_receiver,
    receiver_feature_dim,
    run_batch,
    sender_feature_dim,
)

RewardMode = Literal["team", "individual"]


@dataclass(frozen=True)
class PPOConfig:
    updates: int = 220
    batch_size: int = 4096
    minibatch_size: int = 2048
    epochs: int = 4
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.7
    hidden_dim: int = 64
    reward_mode: RewardMode = "team"
    standardize_advantages: bool = True
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass
class PPOPolicy:
    sender: ActorCritic
    receiver: ActorCritic
    config: GameConfig
    device: str = "cpu"


@dataclass
class PPORollout:
    batch: Batch
    sender_logprob: np.ndarray
    sender_value: np.ndarray
    receiver_logprob: np.ndarray
    receiver_value: np.ndarray


def train_ippo(
    game_config: GameConfig,
    ppo_config: PPOConfig,
    *,
    seed: int,
    comm_mode: CommMode = "learned",
    monitor_penalty: float = 0.0,
    flagged_messages: Iterable[int] = (),
    shaping: "AuditShaping | None" = None,
) -> tuple[PPOPolicy, list[dict[str, float]]]:
    """Train shared sender/receiver policies with a one-step PPO objective."""

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    device = torch.device(_resolve_device(ppo_config.device))
    sender = ActorCritic(sender_feature_dim(), game_config.k_messages, ppo_config.hidden_dim).to(device)
    receiver = ActorCritic(receiver_feature_dim(game_config), 2, ppo_config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()),
        lr=ppo_config.learning_rate,
        eps=1e-5,
    )

    history: list[dict[str, float]] = []
    for update in range(ppo_config.updates):
        rollout = collect_rollout(
            sender,
            receiver,
            game_config,
            np_rng,
            ppo_config.batch_size,
            device=device,
            comm_mode=comm_mode,
            monitor_penalty=monitor_penalty,
            flagged_messages=flagged_messages,
            shaping=shaping,
            reward_mode=ppo_config.reward_mode,
        )
        _ppo_update(sender, receiver, optimizer, rollout, ppo_config, device, train_sender=comm_mode == "learned")

        if update == 0 or (update + 1) % max(1, ppo_config.updates // 10) == 0:
            history.append(
                {
                    "update": float(update + 1),
                    "reward": float(rollout.batch.train_rewards.mean()),
                    "welfare": float(rollout.batch.raw_payoffs.mean()),
                    "join_rate": float(rollout.batch.actions.mean()),
                    "success_rate": float(rollout.batch.success.mean()),
                    "message_entropy": _message_entropy(rollout.batch.messages, game_config.k_messages),
                }
            )

    return PPOPolicy(sender=sender, receiver=receiver, config=game_config, device=str(device)), history


def collect_rollout(
    sender: ActorCritic,
    receiver: ActorCritic,
    config: GameConfig,
    rng: np.random.Generator,
    batch_size: int,
    *,
    device: torch.device,
    comm_mode: CommMode,
    monitor_penalty: float,
    flagged_messages: Iterable[int],
    reward_mode: RewardMode,
    shaping: "AuditShaping | None" = None,
) -> PPORollout:
    """Sample a rollout using torch policies and numpy environment payoffs."""

    theta, signals, type_ids = _sample_world_like(config, rng, batch_size)
    sender_features = _private_features(config, signals, type_ids, config.monitor_strength)
    sender_x = torch.as_tensor(sender_features.reshape(-1, sender_feature_dim()), dtype=torch.float32, device=device)

    with torch.no_grad():
        sender_logits, sender_values_flat = sender(sender_x)
        sender_dist = Categorical(logits=sender_logits)
        sender_actions_flat = sender_dist.sample()
        sender_logprob_flat = sender_dist.log_prob(sender_actions_flat)

    sampled_messages = sender_actions_flat.cpu().numpy().reshape(batch_size, config.n_agents)
    if comm_mode == "none":
        messages = np.zeros((batch_size, config.n_agents), dtype=np.int64)
    elif comm_mode == "semantic":
        from paper2.audit_game import strategic_content

        messages = strategic_content(config, signals, type_ids)
    else:
        messages = sampled_messages

    histograms = _message_histograms_like(messages, config.k_messages)
    receiver_private = _private_features(config, signals, type_ids, config.monitor_strength)
    receiver_features = np.concatenate([receiver_private, histograms], axis=-1)
    receiver_x = torch.as_tensor(receiver_features.reshape(-1, receiver_feature_dim(config)), dtype=torch.float32, device=device)

    with torch.no_grad():
        receiver_logits, receiver_values_flat = receiver(receiver_x)
        receiver_dist = Categorical(logits=receiver_logits)
        receiver_actions_flat = receiver_dist.sample()
        receiver_logprob_flat = receiver_dist.log_prob(receiver_actions_flat)

    actions = receiver_actions_flat.cpu().numpy().reshape(batch_size, config.n_agents)
    batch = _make_batch(
        config,
        theta,
        signals,
        type_ids,
        sender_features,
        messages,
        receiver_features,
        actions,
        sender_probs=F.softmax(sender_logits, dim=-1).cpu().numpy().reshape(batch_size, config.n_agents, config.k_messages),
        action_probs=F.softmax(receiver_logits, dim=-1).cpu().numpy().reshape(batch_size, config.n_agents, 2),
        monitor_penalty=monitor_penalty,
        flagged_messages=flagged_messages,
        shaping=shaping,
        reward_mode=reward_mode,
    )

    return PPORollout(
        batch=batch,
        sender_logprob=sender_logprob_flat.cpu().numpy().reshape(batch_size, config.n_agents),
        sender_value=sender_values_flat.cpu().numpy().reshape(batch_size, config.n_agents),
        receiver_logprob=receiver_logprob_flat.cpu().numpy().reshape(batch_size, config.n_agents),
        receiver_value=receiver_values_flat.cpu().numpy().reshape(batch_size, config.n_agents),
    )


def evaluate_ppo(
    policy: PPOPolicy,
    *,
    seed: int,
    n_episodes: int,
    comm_mode: CommMode,
    monitor_penalty: float = 0.0,
    flagged_messages: Iterable[int] = (),
    shuffle_messages: bool = False,
) -> Batch:
    rng = np.random.default_rng(seed)
    config = policy.config
    device = torch.device(policy.device)
    theta, signals, type_ids = _sample_world_like(config, rng, n_episodes)
    sender_features = _private_features(config, signals, type_ids, config.monitor_strength)
    sender_x = torch.as_tensor(sender_features.reshape(-1, sender_feature_dim()), dtype=torch.float32, device=device)
    with torch.no_grad():
        sender_logits, _ = policy.sender(sender_x)
        messages = sender_logits.argmax(dim=-1).cpu().numpy().reshape(n_episodes, config.n_agents)
    if comm_mode == "none":
        messages = np.zeros((n_episodes, config.n_agents), dtype=np.int64)
    elif comm_mode == "semantic":
        from paper2.audit_game import strategic_content

        messages = strategic_content(config, signals, type_ids)

    receiver_messages = messages.copy()
    if shuffle_messages:
        for row in receiver_messages:
            rng.shuffle(row)
    histograms = _message_histograms_like(receiver_messages, config.k_messages)
    receiver_private = _private_features(config, signals, type_ids, config.monitor_strength)
    receiver_features = np.concatenate([receiver_private, histograms], axis=-1)
    receiver_x = torch.as_tensor(receiver_features.reshape(-1, receiver_feature_dim(config)), dtype=torch.float32, device=device)
    with torch.no_grad():
        receiver_logits, _ = policy.receiver(receiver_x)
        actions = receiver_logits.argmax(dim=-1).cpu().numpy().reshape(n_episodes, config.n_agents)
        sender_probs = F.softmax(sender_logits, dim=-1).cpu().numpy().reshape(n_episodes, config.n_agents, config.k_messages)
        action_probs = F.softmax(receiver_logits, dim=-1).cpu().numpy().reshape(n_episodes, config.n_agents, 2)

    return _make_batch(
        config,
        theta,
        signals,
        type_ids,
        sender_features,
        messages,
        receiver_features,
        actions,
        sender_probs=sender_probs,
        action_probs=action_probs,
        monitor_penalty=monitor_penalty,
        flagged_messages=flagged_messages,
        reward_mode="team",
    )


def evaluate_ppo_metrics(
    policy: PPOPolicy,
    *,
    seed: int,
    n_episodes: int,
    comm_mode: CommMode,
    monitor_penalty: float = 0.0,
    flagged_messages: Iterable[int] = (),
) -> dict[str, float]:
    batch = evaluate_ppo(
        policy,
        seed=seed,
        n_episodes=n_episodes,
        comm_mode=comm_mode,
        monitor_penalty=monitor_penalty,
        flagged_messages=flagged_messages,
    )
    shuffled = evaluate_ppo(
        policy,
        seed=seed,
        n_episodes=n_episodes,
        comm_mode=comm_mode,
        monitor_penalty=monitor_penalty,
        flagged_messages=flagged_messages,
        shuffle_messages=True,
    )
    welfare = float(batch.raw_payoffs.mean())
    shuffled_welfare = float(shuffled.raw_payoffs.mean())
    return {
        "welfare": welfare,
        "shuffled_welfare": shuffled_welfare,
        "message_value": welfare - shuffled_welfare,
        "join_rate": float(batch.actions.mean()),
        "success_rate": float(batch.success.mean()),
        "message_entropy": _message_entropy(batch.messages, policy.config.k_messages),
    }


def ppo_probe_scores(
    policy: PPOPolicy,
    *,
    seed: int,
    n_episodes: int,
    comm_mode: CommMode,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    batch = evaluate_ppo(policy, seed=seed, n_episodes=n_episodes, comm_mode=comm_mode)
    y = batch.content.reshape(-1)
    x_auditor = _one_hot_np(batch.messages.reshape(-1), policy.config.k_messages)
    auditor_score = _fit_probe(LogisticRegression, balanced_accuracy_score, x_auditor, y, train_test_split)

    x_rows = []
    y_rows = []
    receiver_private = _private_features(policy.config, batch.signals, batch.type_ids, policy.config.monitor_strength)
    for episode in range(n_episodes):
        for sender in range(policy.config.n_agents):
            message_features = _one_hot_np(np.array([batch.messages[episode, sender]]), policy.config.k_messages)[0]
            label = batch.content[episode, sender]
            for receiver in range(policy.config.n_agents):
                if receiver == sender:
                    continue
                x_rows.append(np.concatenate([message_features, receiver_private[episode, receiver]]))
                y_rows.append(label)
    receiver_score = _fit_probe(
        LogisticRegression,
        balanced_accuracy_score,
        np.asarray(x_rows),
        np.asarray(y_rows),
        train_test_split,
    )
    return {
        "auditor_balanced_accuracy": auditor_score,
        "receiver_balanced_accuracy": receiver_score,
        "decodability_gap": max(receiver_score - auditor_score, 0.0),
    }


def ppo_join_curve(
    policy: PPOPolicy,
    *,
    seed: int,
    n_episodes: int,
    comm_mode: CommMode,
    n_bins: int = 12,
) -> list[dict[str, float]]:
    batch = evaluate_ppo(policy, seed=seed, n_episodes=n_episodes, comm_mode=comm_mode)
    bins = np.linspace(policy.config.theta_low, policy.config.theta_high, n_bins + 1)
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (batch.theta >= left) & (batch.theta < right)
        if np.any(mask):
            rows.append(
                {
                    "theta_left": float(left),
                    "theta_right": float(right),
                    "theta_mid": float((left + right) / 2.0),
                    "join_rate": float(batch.actions[mask].mean()),
                    "success_rate": float(batch.success[mask].mean()),
                    "welfare": float(batch.raw_payoffs[mask].mean()),
                }
            )
    return rows


def ppo_message_dictionary(
    policy: PPOPolicy,
    *,
    seed: int,
    n_episodes: int,
    comm_mode: CommMode,
) -> list[dict[str, float]]:
    batch = evaluate_ppo(policy, seed=seed, n_episodes=n_episodes, comm_mode=comm_mode)
    rows = []
    flat_messages = batch.messages.reshape(-1)
    flat_signals = batch.signals.reshape(-1)
    flat_content = batch.content.reshape(-1)
    for message in range(policy.config.k_messages):
        mask = flat_messages == message
        if not np.any(mask):
            rows.append({"message": float(message), "share": 0.0})
        else:
            rows.append(
                {
                    "message": float(message),
                    "share": float(mask.mean()),
                    "mean_signal": float(flat_signals[mask].mean()),
                    "p_weak": float((flat_content[mask] == 0).mean()),
                    "p_ambiguous": float((flat_content[mask] == 1).mean()),
                    "p_strong": float((flat_content[mask] == 2).mean()),
                }
            )
    return rows


def ppo_population(
    policy: PPOPolicy,
    *,
    seed: int,
    n_episodes: int,
    comm_mode: CommMode,
):
    """Flattened (message, content, private) population for OOD auditor transfer."""

    from paper2.audit_transfer import Population

    batch = evaluate_ppo(policy, seed=seed, n_episodes=n_episodes, comm_mode=comm_mode)
    private = batch.sender_features.reshape(-1, sender_feature_dim())
    return Population(
        messages=batch.messages.reshape(-1),
        content=batch.content.reshape(-1),
        private=private,
    )


def ppo_cross_play_matrix(
    policies_by_seed: dict[int, PPOPolicy],
    *,
    comm_mode: CommMode,
    n_episodes: int,
) -> tuple[list[int], np.ndarray]:
    seeds = sorted(policies_by_seed)
    matrix = np.zeros((len(seeds), len(seeds)), dtype=float)
    for row, sender_seed in enumerate(seeds):
        for col, receiver_seed in enumerate(seeds):
            combined = PPOPolicy(
                sender=policies_by_seed[sender_seed].sender,
                receiver=policies_by_seed[receiver_seed].receiver,
                config=policies_by_seed[sender_seed].config,
                device=policies_by_seed[sender_seed].device,
            )
            metrics = evaluate_ppo_metrics(
                combined,
                seed=310_000 + 101 * row + col,
                n_episodes=n_episodes,
                comm_mode=comm_mode,
            )
            matrix[row, col] = metrics["welfare"]
    return seeds, matrix


def policy_to_linear_params(policy: PPOPolicy, seed: int = 0) -> PolicyParams:
    """Compatibility helper for old numpy evaluators.

    This samples the neural policies on random feature grids and fits no model;
    it only returns random linear params with the right shape. Prefer PPO-specific
    evaluators above.
    """

    rng = np.random.default_rng(seed)
    return PolicyParams(
        sender_w=rng.normal(0, 0.01, size=(sender_feature_dim(), policy.config.k_messages)),
        receiver_w=rng.normal(0, 0.01, size=(receiver_feature_dim(policy.config), 2)),
    )


def _ppo_update(
    sender: ActorCritic,
    receiver: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: PPORollout,
    config: PPOConfig,
    device: torch.device,
    *,
    train_sender: bool,
) -> None:
    batch = rollout.batch
    n = batch.actions.size
    rewards = torch.as_tensor(batch.train_rewards.reshape(-1), dtype=torch.float32, device=device)

    sender_features = torch.as_tensor(batch.sender_features.reshape(n, sender_feature_dim()), dtype=torch.float32, device=device)
    receiver_features = torch.as_tensor(batch.receiver_features.reshape(n, batch.receiver_features.shape[-1]), dtype=torch.float32, device=device)
    messages = torch.as_tensor(batch.messages.reshape(-1), dtype=torch.long, device=device)
    actions = torch.as_tensor(batch.actions.reshape(-1), dtype=torch.long, device=device)
    old_sender_logprob = torch.as_tensor(rollout.sender_logprob.reshape(-1), dtype=torch.float32, device=device)
    old_receiver_logprob = torch.as_tensor(rollout.receiver_logprob.reshape(-1), dtype=torch.float32, device=device)
    sender_advantages = rewards - torch.as_tensor(rollout.sender_value.reshape(-1), dtype=torch.float32, device=device)
    receiver_advantages = rewards - torch.as_tensor(rollout.receiver_value.reshape(-1), dtype=torch.float32, device=device)
    returns = rewards

    if config.standardize_advantages:
        sender_advantages = (sender_advantages - sender_advantages.mean()) / (sender_advantages.std() + 1e-8)
        receiver_advantages = (receiver_advantages - receiver_advantages.mean()) / (receiver_advantages.std() + 1e-8)

    indices = torch.arange(n, device=device)
    for _ in range(config.epochs):
        shuffled = indices[torch.randperm(n, device=device)]
        for start in range(0, n, config.minibatch_size):
            mb = shuffled[start : start + config.minibatch_size]
            loss = torch.zeros((), dtype=torch.float32, device=device)
            entropy = torch.zeros((), dtype=torch.float32, device=device)

            if train_sender:
                sender_logits, sender_values = sender(sender_features[mb])
                sender_dist = Categorical(logits=sender_logits)
                sender_logprob = sender_dist.log_prob(messages[mb])
                sender_ratio = torch.exp(sender_logprob - old_sender_logprob[mb])
                sender_pg_1 = -sender_advantages[mb] * sender_ratio
                sender_pg_2 = -sender_advantages[mb] * torch.clamp(sender_ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                sender_pg_loss = torch.max(sender_pg_1, sender_pg_2).mean()
                sender_v_loss = F.mse_loss(sender_values, returns[mb])
                loss = loss + sender_pg_loss + config.value_coef * sender_v_loss
                entropy = entropy + sender_dist.entropy().mean()

            receiver_logits, receiver_values = receiver(receiver_features[mb])
            receiver_dist = Categorical(logits=receiver_logits)
            receiver_logprob = receiver_dist.log_prob(actions[mb])
            receiver_ratio = torch.exp(receiver_logprob - old_receiver_logprob[mb])
            receiver_pg_1 = -receiver_advantages[mb] * receiver_ratio
            receiver_pg_2 = -receiver_advantages[mb] * torch.clamp(receiver_ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            receiver_pg_loss = torch.max(receiver_pg_1, receiver_pg_2).mean()
            receiver_v_loss = F.mse_loss(receiver_values, returns[mb])
            loss = loss + receiver_pg_loss + config.value_coef * receiver_v_loss
            entropy = entropy + receiver_dist.entropy().mean()

            loss = loss - config.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(sender.parameters()) + list(receiver.parameters()), config.max_grad_norm)
            optimizer.step()


def _sample_world_like(
    config: GameConfig,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = rng.uniform(config.theta_low, config.theta_high, size=batch_size)
    signals = theta[:, None] + rng.normal(0.0, config.sigma, size=(batch_size, config.n_agents))
    type_ids = rng.integers(0, len(config.payoff_types), size=(batch_size, config.n_agents))
    return theta, signals, type_ids


def _message_histograms_like(messages: np.ndarray, k_messages: int) -> np.ndarray:
    one_hot = _one_hot_np(messages, k_messages)
    n_agents = messages.shape[1]
    if n_agents == 1:
        return np.zeros_like(one_hot)
    return (one_hot.sum(axis=1, keepdims=True) - one_hot) / (n_agents - 1)


def _make_batch(
    config: GameConfig,
    theta: np.ndarray,
    signals: np.ndarray,
    type_ids: np.ndarray,
    sender_features: np.ndarray,
    messages: np.ndarray,
    receiver_features: np.ndarray,
    actions: np.ndarray,
    *,
    sender_probs: np.ndarray,
    action_probs: np.ndarray,
    monitor_penalty: float,
    flagged_messages: Iterable[int],
    reward_mode: RewardMode,
    shaping: "AuditShaping | None" = None,
) -> Batch:
    from paper2.audit_game import _type_arrays, strategic_content

    success = actions.mean(axis=1) > theta
    benefits, costs, _ = _type_arrays(config, type_ids)
    raw_payoffs = np.where(actions == 1, np.where(success[:, None], benefits, -costs), 0.0)
    content = strategic_content(config, signals, type_ids)
    if shaping is not None and shaping.mode != "none":
        message_costs = audit_message_cost(messages, content, shaping, config.k_messages)
    else:
        flagged = np.zeros(config.k_messages, dtype=float)
        for message in flagged_messages:
            if 0 <= int(message) < config.k_messages:
                flagged[int(message)] = 1.0
        message_costs = monitor_penalty * flagged[messages]
    adjusted = raw_payoffs - message_costs
    if reward_mode == "team":
        train_rewards = np.repeat(adjusted.mean(axis=1, keepdims=True), config.n_agents, axis=1)
    else:
        train_rewards = adjusted
    return Batch(
        theta=theta,
        signals=signals,
        type_ids=type_ids,
        content=content,
        sender_features=sender_features,
        sender_probs=sender_probs,
        messages=messages,
        receiver_features=receiver_features,
        action_probs=action_probs,
        actions=actions,
        success=success,
        raw_payoffs=raw_payoffs,
        train_rewards=train_rewards,
    )


def _fit_probe(model_cls, score_fn, x: np.ndarray, y: np.ndarray, split_fn) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    x_train, x_test, y_train, y_test = split_fn(x, y, test_size=0.35, random_state=123, stratify=y)
    model = model_cls(max_iter=1_000)
    model.fit(x_train, y_train)
    return float(score_fn(y_test, model.predict(x_test)))


def _one_hot_np(values: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=float)[values]


def _message_entropy(messages: np.ndarray, k_messages: int) -> float:
    counts = np.bincount(messages.reshape(-1), minlength=k_messages).astype(float)
    probs = counts / max(counts.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device

