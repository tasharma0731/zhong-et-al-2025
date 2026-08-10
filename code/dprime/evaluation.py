from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .trials import _positive_integer, trial_responses


def _finite_rows(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    return np.asarray(np.isfinite(values).all(axis=1), dtype=np.bool_)


def _dprime(scores: ArrayLike, labels: ArrayLike, role_a: int, role_b: int) -> dict[str, float]:
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels)
    a = score[label == role_a]
    b = score[label == role_b]
    if min(len(a), len(b)) < 2:
        return {
            "dprime": float("nan"),
            "mean_a": float("nan"),
            "mean_b": float("nan"),
            "sd_a": float("nan"),
            "sd_b": float("nan"),
            "n_a": float(len(a)),
            "n_b": float(len(b)),
        }
    sd_a = float(np.std(a, ddof=1))
    sd_b = float(np.std(b, ddof=1))
    spread = sd_a + sd_b
    value = float("nan") if spread <= 0 else float(2.0 * (np.mean(a) - np.mean(b)) / spread)
    return {
        "dprime": value,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "sd_a": sd_a,
        "sd_b": sd_b,
        "n_a": float(len(a)),
        "n_b": float(len(b)),
    }


def crossvalidated_scores(
    responses: ArrayLike,
    labels: ArrayLike,
    *,
    role_a: int = 2,
    role_b: int = 0,
    n_folds: int = 4,
    min_per_role: int = 4,
) -> dict[str, Any]:
    """Return held-out coding scores and a contrast for every physical-time fold.

    Fold boundaries are assigned *before* trials with other roles or missing
    responses are filtered.  This preserves physical time: an early and late
    fold cannot collapse together simply because one contains unusable trials.
    Every requested fold must contain both roles (at least two test trials per
    role), otherwise the whole block is marked invalid rather than being
    estimated from a convenient subset of time.

    The coding direction and feature standardization are fit only on the other
    contiguous folds.  D-prime is then computed separately inside each held-out
    fold; callers must not pool raw scores from independently fitted folds.

    These local splits make the score held out with respect to trials.  If the
    input features came from the release-wide SVD/area basis, that upstream
    representation remains transductive and this is not a prospective online
    decoder.
    """

    values = np.asarray(responses, dtype=np.float64)
    label = np.asarray(labels)
    if values.ndim != 2 or label.ndim != 1 or len(values) != len(label):
        raise ValueError("responses and labels must align as trials x features")
    if role_a == role_b:
        raise ValueError("role_a and role_b must be distinct")
    n_folds = _positive_integer("n_folds", n_folds, minimum=2)
    min_per_role = _positive_integer("min_per_role", min_per_role, minimum=2)
    if len(values) < n_folds:
        return {
            "scores": np.array([]), "labels": np.array([]),
            "fold": np.array([], dtype=np.int16), "valid_folds": 0,
            "required_folds": int(n_folds), "fold_metrics": [],
            "invalid_reason": "fewer physical trials than requested folds",
        }

    fold = np.empty(len(values), dtype=np.int16)
    for fold_id, indices in enumerate(np.array_split(np.arange(len(values)), n_folds)):
        fold[indices] = fold_id
    selected = np.isin(label, [role_a, role_b]) & _finite_rows(values)

    for fold_id in range(n_folds):
        test = selected & (fold == fold_id)
        train = selected & (fold != fold_id)
        train_labels = label[train]
        test_labels = label[test]
        if min(
            np.count_nonzero(train_labels == role_a),
            np.count_nonzero(train_labels == role_b),
        ) < min_per_role or min(
            np.count_nonzero(test_labels == role_a),
            np.count_nonzero(test_labels == role_b),
        ) < 2:
            return {
                "scores": np.array([]), "labels": np.array([]),
                "fold": np.array([], dtype=np.int16), "valid_folds": 0,
                "required_folds": int(n_folds), "fold_metrics": [],
                "invalid_reason": (
                    f"fold {fold_id} lacks the predeclared train/test class support"
                ),
            }

    scores = np.full(len(values), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, float]] = []
    for fold_id in range(n_folds):
        test = selected & (fold == fold_id)
        train = selected & (fold != fold_id)
        train_labels = label[train]
        mean = np.mean(values[train], axis=0)
        scale = np.std(values[train], axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        train_z = (values[train] - mean) / scale
        direction = (
            np.mean(train_z[train_labels == role_a], axis=0)
            - np.mean(train_z[train_labels == role_b], axis=0)
        )
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0:
            return {
                "scores": np.array([]), "labels": np.array([]),
                "fold": np.array([], dtype=np.int16), "valid_folds": 0,
                "required_folds": int(n_folds), "fold_metrics": [],
                "invalid_reason": f"fold {fold_id} has no trainable coding direction",
            }
        scores[test] = ((values[test] - mean) / scale) @ (direction / norm)
        metrics = _dprime(scores[test], label[test], role_a, role_b)
        if not np.isfinite(metrics["dprime"]):
            return {
                "scores": np.array([]), "labels": np.array([]),
                "fold": np.array([], dtype=np.int16), "valid_folds": 0,
                "required_folds": int(n_folds), "fold_metrics": [],
                "invalid_reason": f"fold {fold_id} has an undefined held-out d-prime",
            }
        fold_metrics.append({"fold": float(fold_id), **metrics})
    keep = np.isfinite(scores)
    return {
        "scores": scores[keep],
        "labels": label[keep],
        "fold": fold[keep],
        "valid_folds": int(n_folds),
        "required_folds": int(n_folds),
        "fold_metrics": fold_metrics,
        "invalid_reason": None,
    }


def _summarize_fold_metrics(local: Mapping[str, Any]) -> dict[str, float]:
    """Combine only within-fold contrasts; never pool fold-specific scores."""

    metrics = list(local.get("fold_metrics", []))
    required = int(local.get("required_folds", 0))
    if len(metrics) != required or int(local.get("valid_folds", 0)) != required:
        return {
            "dprime": float("nan"), "mean_a": float("nan"),
            "mean_b": float("nan"), "sd_a": float("nan"),
            "sd_b": float("nan"), "separation": float("nan"),
            "spread": float("nan"), "n_a": 0.0, "n_b": 0.0,
        }
    return {
        "dprime": float(np.mean([row["dprime"] for row in metrics])),
        "mean_a": float(np.mean([row["mean_a"] for row in metrics])),
        "mean_b": float(np.mean([row["mean_b"] for row in metrics])),
        "sd_a": float(np.mean([row["sd_a"] for row in metrics])),
        "sd_b": float(np.mean([row["sd_b"] for row in metrics])),
        "separation": float(np.mean([
            row["mean_a"] - row["mean_b"] for row in metrics
        ])),
        "spread": float(np.mean([
            (row["sd_a"] + row["sd_b"]) / 2.0 for row in metrics
        ])),
        "n_a": float(np.sum([row["n_a"] for row in metrics])),
        "n_b": float(np.sum([row["n_b"] for row in metrics])),
    }


def blockwise_dprime(
    trial_features: ArrayLike,
    labels: ArrayLike,
    trial_id: ArrayLike,
    *,
    position_mask: ArrayLike | None = None,
    role_a: int = 2,
    role_b: int = 0,
    block_trials: int = 40,
    stride_trials: int | None = None,
    n_folds: int = 4,
    min_per_role: int = 4,
    require_complete_position_coverage: bool = True,
) -> dict[str, NDArray]:
    """Estimate local d-prime in fixed-width trial blocks.

    Non-overlapping blocks are the default.  A smaller stride is allowed for a
    dotted exploratory display, but overlapping points must not be treated as
    independent observations.
    """

    features = np.asarray(trial_features, dtype=np.float64)
    label = np.asarray(labels)
    trials = np.asarray(trial_id, dtype=np.float64)
    if features.ndim != 3 or label.ndim != 1 or trials.ndim != 1:
        raise ValueError("trial_features, labels, and trial_id must be 3D, 1D, and 1D")
    if len(features) != len(label) or len(features) != len(trials):
        raise ValueError("trial_features, labels, and trial_id must align")
    if not np.isfinite(trials).all():
        raise ValueError("trial_id must contain only finite values")
    if role_a == role_b:
        raise ValueError("role_a and role_b must be distinct")
    block_trials = _positive_integer("block_trials", block_trials, minimum=8)
    n_folds = _positive_integer("n_folds", n_folds, minimum=2)
    min_per_role = _positive_integer("min_per_role", min_per_role, minimum=2)
    stride = block_trials if stride_trials is None else _positive_integer(
        "stride_trials", stride_trials
    )
    order = np.argsort(trials)
    features, label, trials = features[order], label[order], trials[order]
    response = trial_responses(
        features,
        position_mask,
        require_complete_position_coverage=require_complete_position_coverage,
    )

    rows: list[dict[str, float]] = []
    starts = range(0, max(len(trials) - block_trials + 1, 0), stride)
    for start in starts:
        stop = start + block_trials
        local = crossvalidated_scores(
            response[start:stop],
            label[start:stop],
            role_a=role_a,
            role_b=role_b,
            n_folds=n_folds,
            min_per_role=min_per_role,
        )
        metrics = _summarize_fold_metrics(local)
        rows.append(
            {
                "start_trial": float(trials[start]),
                "stop_trial": float(trials[stop - 1]),
                "midpoint": float(np.mean(trials[start:stop])),
                "valid_folds": float(local["valid_folds"]),
                "required_folds": float(local["required_folds"]),
                **metrics,
            }
        )
    names = (
        "start_trial", "stop_trial", "midpoint", "dprime", "mean_a",
        "mean_b", "sd_a", "sd_b", "separation", "spread", "n_a", "n_b",
        "valid_folds", "required_folds",
    )
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in names
    }


def position_dprime_surface(
    trial_features: ArrayLike,
    labels: ArrayLike,
    trial_id: ArrayLike,
    *,
    role_a: int = 2,
    role_b: int = 0,
    block_trials: int = 40,
    n_folds: int = 4,
    min_per_role: int = 4,
    require_complete_position_coverage: bool = True,
) -> dict[str, NDArray]:
    """Estimate held-out d-prime for every time block and corridor bin."""

    features = np.asarray(trial_features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError("trial_features must have shape (trials, position, features)")
    values = []
    midpoint = None
    for bin_index in range(features.shape[1]):
        mask = np.zeros(features.shape[1], dtype=bool)
        mask[bin_index] = True
        curve = blockwise_dprime(
            features,
            labels,
            trial_id,
            position_mask=mask,
            role_a=role_a,
            role_b=role_b,
            block_trials=block_trials,
            n_folds=n_folds,
            min_per_role=min_per_role,
            require_complete_position_coverage=require_complete_position_coverage,
        )
        midpoint = curve["midpoint"] if midpoint is None else midpoint
        values.append(curve["dprime"])
    return {
        "midpoint": np.asarray([] if midpoint is None else midpoint),
        "dprime": np.asarray(values, dtype=np.float64).T,
    }


def cross_temporal_dprime(
    trial_features: ArrayLike,
    labels: ArrayLike,
    trial_id: ArrayLike,
    *,
    position_mask: ArrayLike | None = None,
    role_a: int = 2,
    role_b: int = 0,
    block_trials: int = 40,
    min_per_role: int = 4,
    require_complete_position_coverage: bool = True,
) -> dict[str, NDArray]:
    """Train a coding axis in one block and test it in every other block.

    Off-diagonal cells use disjoint train and test trials.  Diagonal cells use
    the contiguous-fold cross-validation from :func:`crossvalidated_scores`.
    """

    features = np.asarray(trial_features, dtype=np.float64)
    label = np.asarray(labels)
    trials = np.asarray(trial_id, dtype=np.float64)
    if features.ndim != 3 or label.ndim != 1 or trials.ndim != 1:
        raise ValueError("trial_features, labels, and trial_id must be 3D, 1D, and 1D")
    if len(features) != len(label) or len(features) != len(trials):
        raise ValueError("trial_features, labels, and trial_id must align")
    if not np.isfinite(trials).all():
        raise ValueError("trial_id must contain only finite values")
    if role_a == role_b:
        raise ValueError("role_a and role_b must be distinct")
    block_trials = _positive_integer("block_trials", block_trials, minimum=8)
    min_per_role = _positive_integer("min_per_role", min_per_role, minimum=2)
    order = np.argsort(trials)
    response = trial_responses(
        features[order],
        position_mask,
        require_complete_position_coverage=require_complete_position_coverage,
    )
    label, trials = label[order], trials[order]
    blocks = [
        np.arange(start, start + block_trials)
        for start in range(0, max(len(trials) - block_trials + 1, 0), block_trials)
    ]
    matrix = np.full((len(blocks), len(blocks)), np.nan, dtype=np.float64)
    for train_id, train_idx in enumerate(blocks):
        train_values, train_labels = response[train_idx], label[train_idx]
        selected_train = np.isin(train_labels, [role_a, role_b]) & _finite_rows(train_values)
        train_values, train_labels = train_values[selected_train], train_labels[selected_train]
        train_support = min(
            np.count_nonzero(train_labels == role_a),
            np.count_nonzero(train_labels == role_b),
        )
        if train_support < min_per_role:
            continue
        mean = np.mean(train_values, axis=0)
        scale = np.std(train_values, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        train_z = (train_values - mean) / scale
        direction = (
            np.mean(train_z[train_labels == role_a], axis=0)
            - np.mean(train_z[train_labels == role_b], axis=0)
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 0 or not np.isfinite(norm):
            continue
        direction /= norm
        for test_id, test_idx in enumerate(blocks):
            if train_id == test_id:
                local = crossvalidated_scores(
                    response[test_idx], label[test_idx], role_a=role_a,
                    role_b=role_b, n_folds=4, min_per_role=min_per_role,
                )
                matrix[train_id, test_id] = _summarize_fold_metrics(local)["dprime"]
                continue
            test_values, test_labels = response[test_idx], label[test_idx]
            selected_test = np.isin(test_labels, [role_a, role_b]) & _finite_rows(test_values)
            score = ((test_values[selected_test] - mean) / scale) @ direction
            metrics = _dprime(score, test_labels[selected_test], role_a, role_b)
            matrix[train_id, test_id] = metrics["dprime"]
    midpoint = np.asarray([float(np.mean(trials[index])) for index in blocks])
    return {"midpoint": midpoint, "dprime": matrix}
