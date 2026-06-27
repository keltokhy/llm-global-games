"""Protocol-validation experiments for Paper 2.

The learned-agent experiments ask whether training discovers useful
communication. This module asks whether the measurement strategy works in a
controlled setting where the communication protocol is known. It creates arms
that should separate useful/auditable communication from useful/opaque
communication, then computes the same metrics used for learned agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ProtocolArm = Literal[
    "no_comm",
    "canonical_semantic",
    "private_code",
    "monitored_hidden_code",
    "lossy_auditable",
]


@dataclass(frozen=True)
class ProtocolConfig:
    n_agents: int = 8
    theta_low: float = -0.2
    theta_high: float = 1.8
    sigma: float = 0.35
    payoff_types: tuple[tuple[float, float], ...] = ((1.25, 1.0), (1.25, 1.8))
    content_margin: float = 0.12
    message_weight: float = 0.28
    k_messages: int = 4


@dataclass(frozen=True)
class Codebook:
    """Maps content labels to message tokens and back for the receiver."""

    encode: tuple[int, int, int]
    decode: tuple[int, int, int, int]


@dataclass
class ProtocolBatch:
    theta: np.ndarray
    signals: np.ndarray
    type_ids: np.ndarray
    content: np.ndarray
    messages: np.ndarray
    decoded_messages: np.ndarray
    auditor_decoded: np.ndarray
    actions: np.ndarray
    success: np.ndarray
    payoffs: np.ndarray


CONTENT_SCORE = np.asarray([-1.0, 0.0, 1.0])
AUDITOR_UNKNOWN = 1


def identity_codebook() -> Codebook:
    return Codebook(encode=(0, 1, 2), decode=(0, 1, 2, AUDITOR_UNKNOWN))


def random_private_codebook(seed: int) -> Codebook:
    rng = np.random.default_rng(seed)
    derangements = [(1, 2, 0), (2, 0, 1)]
    perm = derangements[int(rng.integers(0, len(derangements)))]
    decode = [AUDITOR_UNKNOWN] * 4
    for content, token in enumerate(perm):
        decode[token] = content
    return Codebook(encode=perm, decode=tuple(decode))


def monitored_hidden_codebook() -> Codebook:
    return Codebook(encode=(3, 1, 2), decode=(AUDITOR_UNKNOWN, 1, 2, 0))


def lossy_codebook() -> Codebook:
    return Codebook(encode=(0, 1, 1), decode=(0, 1, AUDITOR_UNKNOWN, AUDITOR_UNKNOWN))


def codebook_for_arm(arm: ProtocolArm, seed: int = 0) -> Codebook:
    if arm in {"no_comm", "canonical_semantic"}:
        return identity_codebook()
    if arm == "private_code":
        return random_private_codebook(seed)
    if arm == "monitored_hidden_code":
        return monitored_hidden_codebook()
    if arm == "lossy_auditable":
        return lossy_codebook()
    raise ValueError(f"unknown protocol arm: {arm}")


def sample_world(
    config: ProtocolConfig,
    rng: np.random.Generator,
    n_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = rng.uniform(config.theta_low, config.theta_high, size=n_episodes)
    signals = theta[:, None] + rng.normal(0.0, config.sigma, size=(n_episodes, config.n_agents))
    type_ids = rng.integers(0, len(config.payoff_types), size=(n_episodes, config.n_agents))
    return theta, signals, type_ids


def type_arrays(config: ProtocolConfig, type_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payoff_types = np.asarray(config.payoff_types, dtype=float)
    benefits = payoff_types[type_ids, 0]
    costs = payoff_types[type_ids, 1]
    cutoffs = benefits / (benefits + costs)
    return benefits, costs, cutoffs


def strategic_content(config: ProtocolConfig, signals: np.ndarray, type_ids: np.ndarray) -> np.ndarray:
    _, _, cutoffs = type_arrays(config, type_ids)
    gap = signals - cutoffs
    content = np.full(signals.shape, 1, dtype=np.int64)
    content[gap < -config.content_margin] = 0
    content[gap > config.content_margin] = 2
    return content


def encode_messages(content: np.ndarray, codebook: Codebook, arm: ProtocolArm) -> np.ndarray:
    if arm == "no_comm":
        return np.full(content.shape, 3, dtype=np.int64)
    encode = np.asarray(codebook.encode, dtype=np.int64)
    return encode[content]


def decode_messages(messages: np.ndarray, codebook: Codebook) -> np.ndarray:
    decode = np.asarray(codebook.decode, dtype=np.int64)
    return decode[messages]


def canonical_auditor(messages: np.ndarray, arm: ProtocolArm) -> np.ndarray:
    if arm == "lossy_auditable":
        decoded = np.full(messages.shape, AUDITOR_UNKNOWN, dtype=np.int64)
        decoded[messages == 0] = 0
        decoded[messages == 1] = 1
        return decoded
    decoded = np.full(messages.shape, AUDITOR_UNKNOWN, dtype=np.int64)
    for label in (0, 1, 2):
        decoded[messages == label] = label
    return decoded


def play_protocol(
    config: ProtocolConfig,
    *,
    arm: ProtocolArm,
    seed: int,
    n_episodes: int,
    sender_codebook: Codebook | None = None,
    receiver_codebook: Codebook | None = None,
    shuffle_messages: bool = False,
) -> ProtocolBatch:
    rng = np.random.default_rng(seed)
    theta, signals, type_ids = sample_world(config, rng, n_episodes)
    content = strategic_content(config, signals, type_ids)
    sender_codebook = sender_codebook or codebook_for_arm(arm, seed)
    receiver_codebook = receiver_codebook or sender_codebook
    messages = encode_messages(content, sender_codebook, arm)

    receiver_messages = messages.copy()
    if shuffle_messages:
        for row in receiver_messages:
            rng.shuffle(row)

    decoded = decode_messages(receiver_messages, receiver_codebook)
    auditor_decoded = canonical_auditor(messages, arm)
    _, _, cutoffs = type_arrays(config, type_ids)
    centered_signal = signals - cutoffs

    if arm == "no_comm":
        peer_score = np.zeros_like(centered_signal)
    else:
        decoded_score = CONTENT_SCORE[decoded]
        peer_sum = decoded_score.sum(axis=1, keepdims=True) - decoded_score
        peer_score = peer_sum / max(config.n_agents - 1, 1)

    action_index = centered_signal + config.message_weight * peer_score
    actions = (action_index < 0.0).astype(np.int64)
    success = actions.mean(axis=1) > theta
    benefits, costs, _ = type_arrays(config, type_ids)
    payoffs = np.where(actions == 1, np.where(success[:, None], benefits, -costs), 0.0)

    return ProtocolBatch(
        theta=theta,
        signals=signals,
        type_ids=type_ids,
        content=content,
        messages=messages,
        decoded_messages=decoded,
        auditor_decoded=auditor_decoded,
        actions=actions,
        success=success,
        payoffs=payoffs,
    )


def evaluate_protocol_arm(
    config: ProtocolConfig,
    *,
    arm: ProtocolArm,
    seed: int,
    n_episodes: int,
    sender_codebook: Codebook | None = None,
    receiver_codebook: Codebook | None = None,
) -> dict[str, float | str | int]:
    batch = play_protocol(
        config,
        arm=arm,
        seed=seed,
        n_episodes=n_episodes,
        sender_codebook=sender_codebook,
        receiver_codebook=receiver_codebook,
    )
    shuffled = play_protocol(
        config,
        arm=arm,
        seed=seed,
        n_episodes=n_episodes,
        sender_codebook=sender_codebook,
        receiver_codebook=receiver_codebook,
        shuffle_messages=True,
    )
    y = batch.content.reshape(-1)
    receiver_score = _balanced_accuracy(y, batch.decoded_messages.reshape(-1))
    auditor_score = _balanced_accuracy(y, batch.auditor_decoded.reshape(-1))
    welfare = float(batch.payoffs.mean())
    shuffled_welfare = float(shuffled.payoffs.mean())
    message_value = welfare - shuffled_welfare
    decodability_gap = max(receiver_score - auditor_score, 0.0)
    return {
        "arm": arm,
        "seed": seed,
        "welfare": welfare,
        "shuffled_welfare": shuffled_welfare,
        "message_value": message_value,
        "join_rate": float(batch.actions.mean()),
        "success_rate": float(batch.success.mean()),
        "receiver_decodability": receiver_score,
        "auditor_decodability": auditor_score,
        "decodability_gap": decodability_gap,
        "strategic_opacity_gap": max(message_value, 0.0) * decodability_gap,
    }


def cross_play_matrix(
    config: ProtocolConfig,
    *,
    arm: ProtocolArm,
    seeds: list[int],
    n_episodes: int,
) -> pd.DataFrame:
    rows = []
    sender_codebooks = {seed: codebook_for_arm(arm, seed) for seed in seeds}
    receiver_codebooks = {seed: codebook_for_arm(arm, seed) for seed in seeds}
    for sender_seed in seeds:
        for receiver_seed in seeds:
            metrics = evaluate_protocol_arm(
                config,
                arm=arm,
                seed=90_000 + 101 * sender_seed + receiver_seed,
                n_episodes=n_episodes,
                sender_codebook=sender_codebooks[sender_seed],
                receiver_codebook=receiver_codebooks[receiver_seed],
            )
            rows.append(
                {
                    "sender_seed": sender_seed,
                    "receiver_seed": receiver_seed,
                    "welfare": metrics["welfare"],
                    "message_value": metrics["message_value"],
                }
            )
    return pd.DataFrame(rows)


def join_curve(
    config: ProtocolConfig,
    *,
    arm: ProtocolArm,
    seed: int,
    n_episodes: int,
    n_bins: int = 14,
) -> pd.DataFrame:
    batch = play_protocol(config, arm=arm, seed=seed, n_episodes=n_episodes)
    bins = np.linspace(config.theta_low, config.theta_high, n_bins + 1)
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (batch.theta >= left) & (batch.theta < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "theta_left": left,
                "theta_right": right,
                "theta_mid": (left + right) / 2.0,
                "join_rate": float(batch.actions[mask].mean()),
                "success_rate": float(batch.success[mask].mean()),
                "welfare": float(batch.payoffs[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def message_dictionary(
    config: ProtocolConfig,
    *,
    arm: ProtocolArm,
    seed: int,
    n_episodes: int,
) -> pd.DataFrame:
    batch = play_protocol(config, arm=arm, seed=seed, n_episodes=n_episodes)
    rows = []
    flat_messages = batch.messages.reshape(-1)
    flat_content = batch.content.reshape(-1)
    flat_signals = batch.signals.reshape(-1)
    for message in range(config.k_messages):
        mask = flat_messages == message
        row: dict[str, float | str | int] = {"arm": arm, "seed": seed, "message": message, "share": float(mask.mean())}
        if np.any(mask):
            row |= {
                "mean_signal": float(flat_signals[mask].mean()),
                "p_weak": float((flat_content[mask] == 0).mean()),
                "p_ambiguous": float((flat_content[mask] == 1).mean()),
                "p_strong": float((flat_content[mask] == 2).mean()),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def population(
    config: ProtocolConfig,
    *,
    arm: ProtocolArm,
    seed: int,
    n_episodes: int,
):
    """Flattened (message, content, private) population for OOD auditor transfer.

    The private block is the receiver-side context [signal, cutoff, signal-cutoff]
    that an outsider does not see; it lets the receiver probe decode content even
    when the auditor probe cannot.
    """

    from paper2.audit_transfer import Population

    batch = play_protocol(config, arm=arm, seed=seed, n_episodes=n_episodes)
    _, _, cutoffs = type_arrays(config, batch.type_ids)
    private = np.stack(
        [batch.signals, cutoffs, batch.signals - cutoffs],
        axis=-1,
    ).reshape(-1, 3)
    return Population(
        messages=batch.messages.reshape(-1),
        content=batch.content.reshape(-1),
        private=private,
    )


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.asarray([0, 1, 2])
    present = np.intersect1d(labels, np.unique(y_true))
    if present.size == 0:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))
