"""Out-of-distribution (OOD) auditor / probe transfer metrics for Paper 2.

In-distribution probing -- training and testing a message->content decoder on the
*same* population's traffic -- systematically overstates auditability. If every
population (seed / regime) breaks symmetry into its own idiosyncratic code, an
in-distribution probe still decodes it perfectly, yet an outside auditor who
calibrated on different traffic is lost. This module measures that.

Central diagnostic: the **auditor transfer gap**

    ood_auditor_gap = max(auditor_id - auditor_ood, 0)

where ``auditor_id`` is balanced accuracy of a probe trained and tested on the
same population, and ``auditor_ood`` is the same probe applied to *other*
populations. A shared public code transfers (gap ~ 0); an idiosyncratic private
code does not (gap large). Because receivers also hold private context (their own
signal/type), a receiver probe transfers even when the auditor probe does not --
that asymmetry is the OOD strategic-opacity story.

This module is deliberately dependency-light (numpy + scikit-learn) and knows
nothing about the game; callers pass in ``Population`` objects of flattened
per-agent observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    """One population's flattened per-agent observations.

    messages : (N,) int   emitted message tokens.
    content  : (N,) int   ground-truth strategic-content labels (0/1/2).
    private  : (N, d) float | None   receiver-side private context. When present,
               a second "receiver" probe is fit on [one_hot(message), private].
    """

    messages: np.ndarray
    content: np.ndarray
    private: np.ndarray | None = None


def _one_hot(values: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=float)[np.asarray(values, dtype=np.int64)]


def _safe_split(x: np.ndarray, y: np.ndarray, *, seed: int):
    from sklearn.model_selection import train_test_split

    _, counts = np.unique(y, return_counts=True)
    stratify = y if counts.min() >= 2 else None
    return train_test_split(x, y, test_size=0.35, random_state=seed, stratify=stratify)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    if np.unique(y_true).size < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def _fit_probe(x: np.ndarray, y: np.ndarray, *, seed: int):
    """Fit a logistic probe on a held-out split; return (model, id_accuracy)."""

    from sklearn.linear_model import LogisticRegression

    if np.unique(y).size < 2:
        return None, float("nan")
    x_train, x_test, y_train, y_test = _safe_split(x, y, seed=seed)
    if np.unique(y_train).size < 2:
        return None, float("nan")
    model = LogisticRegression(max_iter=1_000)
    model.fit(x_train, y_train)
    id_acc = _balanced_accuracy(y_test, model.predict(x_test))
    return model, id_acc


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def auditor_transfer_scores(
    populations: dict[int, Population],
    k_messages: int,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Train probes on each population and evaluate in- and out-of-distribution.

    Returns a flat dict with in-distribution accuracy, OOD accuracy, the transfer
    gap for both the auditor probe (message only) and the receiver probe (message
    + private context), plus an OOD strategic-opacity gap.
    """

    ids = sorted(populations)
    has_private = all(populations[i].private is not None for i in ids)

    auditor_models: dict[int, object] = {}
    auditor_id: list[float] = []
    receiver_models: dict[int, object] = {}
    receiver_id: list[float] = []

    for i in ids:
        pop = populations[i]
        x_aud = _one_hot(pop.messages, k_messages)
        model, id_acc = _fit_probe(x_aud, pop.content, seed=seed + i)
        auditor_models[i] = model
        auditor_id.append(id_acc)
        if has_private:
            x_rec = np.concatenate([_one_hot(pop.messages, k_messages), pop.private], axis=1)
            rmodel, rid = _fit_probe(x_rec, pop.content, seed=seed + 1_000 + i)
            receiver_models[i] = rmodel
            receiver_id.append(rid)

    auditor_ood: list[float] = []
    receiver_ood: list[float] = []
    pair_rows: list[dict[str, float]] = []
    for src in ids:
        for dst in ids:
            if src == dst:
                continue
            dst_pop = populations[dst]
            row: dict[str, float] = {"source": float(src), "target": float(dst)}
            if auditor_models[src] is not None:
                x_aud = _one_hot(dst_pop.messages, k_messages)
                acc = _balanced_accuracy(dst_pop.content, auditor_models[src].predict(x_aud))
                auditor_ood.append(acc)
                row["auditor_ood"] = acc
            if has_private and receiver_models.get(src) is not None:
                x_rec = np.concatenate([_one_hot(dst_pop.messages, k_messages), dst_pop.private], axis=1)
                racc = _balanced_accuracy(dst_pop.content, receiver_models[src].predict(x_rec))
                receiver_ood.append(racc)
                row["receiver_ood"] = racc
            pair_rows.append(row)

    aud_id = _nanmean(auditor_id)
    aud_ood = _nanmean(auditor_ood)
    rec_id = _nanmean(receiver_id) if has_private else float("nan")
    rec_ood = _nanmean(receiver_ood) if has_private else float("nan")

    out = {
        "auditor_id": aud_id,
        "auditor_ood": aud_ood,
        "ood_auditor_gap": _pos_gap(aud_id, aud_ood),
        "n_populations": float(len(ids)),
    }
    if has_private:
        out.update(
            {
                "receiver_id": rec_id,
                "receiver_ood": rec_ood,
                # Deployment asymmetry: a receiver always operates inside its own
                # population (self-play), so its realistic decodability is the
                # in-distribution score; only the auditor is forced to generalize
                # across populations. The opacity that matters is therefore how
                # much better an in-population receiver decodes than a transferred
                # auditor does.
                "ood_decodability_gap": _pos_gap(rec_id, aud_ood),
            }
        )
    return out


def _pos_gap(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(max(a - b, 0.0))


def ood_strategic_opacity_gap(message_value: float, transfer: dict[str, float]) -> float:
    """OOD analogue of SOG: message value * (receiver_ood - auditor_ood)_+.

    Uses cross-population decodability. When the receiver-side OOD gap is absent
    (no private context supplied), falls back to the auditor transfer gap so the
    metric still rewards population-idiosyncratic opacity.
    """

    value = max(message_value, 0.0)
    gap = transfer.get("ood_decodability_gap")
    if gap is None or not np.isfinite(gap):
        gap = transfer.get("ood_auditor_gap", 0.0)
    gap = gap if np.isfinite(gap) else 0.0
    return float(value * max(gap, 0.0))
