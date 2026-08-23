from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

ArrayLike = Sequence[int] | Sequence[float] | np.ndarray | pd.Series
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5
QUALITY_METRIC_NAMES = ("precision", "recall", "f1", "pr_auc", "roc_auc")


def binary_predictions(scores: ArrayLike, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    return (np.asarray(scores, dtype="float64") >= threshold).astype("int8")


def quality_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype="int8")
    scores = np.asarray(y_score, dtype="float64")
    preds = binary_predictions(scores, threshold)
    metrics = {
        "precision": float(precision_score(truth, preds, zero_division=0)),
        "recall": float(recall_score(truth, preds, zero_division=0)),
        "f1": float(f1_score(truth, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(truth, scores)) if _has_both_classes(truth) else float("nan"),
        "roc_auc": float(roc_auc_score(truth, scores)) if _has_both_classes(truth) else float("nan"),
        "threshold": float(threshold),
        "n": int(len(truth)),
        "n_positive": int(truth.sum()),
        "positive_rate": float(truth.mean()) if len(truth) else float("nan"),
    }
    return metrics


def _has_both_classes(y_true: np.ndarray) -> bool:
    return np.unique(y_true).size == 2


def agreement_metrics(
    v1_pred: ArrayLike,
    v2_pred: ArrayLike,
) -> dict[str, float]:
    left = np.asarray(v1_pred, dtype="int8")
    right = np.asarray(v2_pred, dtype="int8")
    if left.shape != right.shape:
        raise ValueError("prediction arrays must have the same shape")
    n = len(left)
    if n == 0:
        return {
            "agreement": float("nan"),
            "v1_neg_v2_pos": float("nan"),
            "v1_pos_v2_neg": float("nan"),
            "n": 0,
        }
    return {
        "agreement": float((left == right).mean()),
        "v1_neg_v2_pos": float(((left == 0) & (right == 1)).mean()),
        "v1_pos_v2_neg": float(((left == 1) & (right == 0)).mean()),
        "n": int(n),
    }


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    left = np.asarray(p, dtype="float64")
    right = np.asarray(q, dtype="float64")
    left = left / left.sum() if left.sum() else left
    right = right / right.sum() if right.sum() else right
    mid = 0.5 * (left + right)
    return float(0.5 * (_kl(left, mid) + _kl(right, mid)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))


def score_jsd(
    v1_scores: ArrayLike,
    v2_scores: ArrayLike,
    *,
    bins: int = 20,
) -> float:
    left = np.asarray(v1_scores, dtype="float64")
    right = np.asarray(v2_scores, dtype="float64")
    edges = np.linspace(0.0, 1.0, bins + 1)
    p, _ = np.histogram(left, bins=edges)
    q, _ = np.histogram(right, bins=edges)
    return jensen_shannon_divergence(p.astype("float64"), q.astype("float64"))


def population_stability_index(
    expected: Sequence[Any] | np.ndarray | pd.Series,
    actual: Sequence[Any] | np.ndarray | pd.Series,
    *,
    bins: int = 10,
    categorical: bool = False,
    eps: float = 1e-6,
) -> float:
    if categorical:
        expected_counts = pd.Series(expected).astype("string").value_counts(dropna=False)
        actual_counts = pd.Series(actual).astype("string").value_counts(dropna=False)
        index = expected_counts.index.union(actual_counts.index)
        exp = expected_counts.reindex(index, fill_value=0).to_numpy(dtype="float64")
        act = actual_counts.reindex(index, fill_value=0).to_numpy(dtype="float64")
    else:
        expected_num = pd.to_numeric(pd.Series(expected), errors="coerce").dropna()
        actual_num = pd.to_numeric(pd.Series(actual), errors="coerce").dropna()
        if expected_num.empty or actual_num.empty:
            return float("nan")
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(expected_num, quantiles))
        if edges.size < 2:
            return 0.0
        exp, _ = np.histogram(expected_num, bins=edges)
        act, _ = np.histogram(actual_num, bins=edges)
        exp = exp.astype("float64")
        act = act.astype("float64")
    exp_share = (exp + eps) / (exp.sum() + eps * len(exp))
    act_share = (act + eps) / (act.sum() + eps * len(act))
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def feature_psi_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Mapping[str, bool],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column, categorical in columns.items():
        rows.append(
            {
                "feature": column,
                "kind": "categorical" if categorical else "numeric",
                "psi": population_stability_index(
                    reference[column],
                    current[column],
                    categorical=categorical,
                ),
            }
        )
    return pd.DataFrame(rows)


def comparison_frame(
    v1_metrics: Mapping[str, float],
    v2_metrics: Mapping[str, float],
    agreement: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for name in QUALITY_METRIC_NAMES:
        rows.append({"metric": name, "v1": v1_metrics.get(name), "v2": v2_metrics.get(name)})
    rows.extend(
        [
            {"metric": "agreement", "v1": np.nan, "v2": agreement.get("agreement")},
            {"metric": "v1_neg_v2_pos", "v1": np.nan, "v2": agreement.get("v1_neg_v2_pos")},
            {"metric": "v1_pos_v2_neg", "v1": np.nan, "v2": agreement.get("v1_pos_v2_neg")},
        ]
    )
    return pd.DataFrame(rows)
