from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cross_model_drift.features import target_vector
from cross_model_drift.metrics import (
    agreement_metrics,
    binary_predictions,
    comparison_frame,
    feature_psi_table,
    quality_metrics,
    score_jsd,
)
from cross_model_drift.models import TrainedModel

DRIFT_COLUMNS = {
    "payout_country": True,
    "payout_currency": True,
    "amount_usd": False,
    "fee_usd": False,
    "hour_of_day": False,
}


@dataclass(frozen=True)
class ComparisonResult:
    quality: pd.DataFrame
    v1_metrics: dict[str, float]
    v2_metrics: dict[str, float]
    agreement: dict[str, float]
    score_jsd: float
    psi: pd.DataFrame
    predictions: pd.DataFrame

    def promote_v2(self, min_pr_auc_gain: float = 0.0) -> bool:
        return float(self.v2_metrics["pr_auc"]) >= float(self.v1_metrics["pr_auc"]) + min_pr_auc_gain


def compare_models(
    holdout: pd.DataFrame,
    v1: TrainedModel,
    v2: TrainedModel,
    *,
    reference: pd.DataFrame | None = None,
    threshold: float | None = None,
) -> ComparisonResult:
    y = target_vector(holdout)
    cut = threshold if threshold is not None else v1.threshold
    v1_scores = v1.predict_proba(holdout)
    v2_scores = v2.predict_proba(holdout)
    v1_pred = binary_predictions(v1_scores, cut)
    v2_pred = binary_predictions(v2_scores, cut)
    v1_metrics = quality_metrics(y, v1_scores, threshold=cut)
    v2_metrics = quality_metrics(y, v2_scores, threshold=cut)
    agreement = agreement_metrics(v1_pred, v2_pred)
    quality = comparison_frame(v1_metrics, v2_metrics, agreement)
    psi_ref = reference if reference is not None else holdout
    psi = feature_psi_table(psi_ref, holdout, DRIFT_COLUMNS)
    preds = pd.DataFrame(
        {
            "y_true": y.to_numpy(),
            "v1_score": v1_scores,
            "v2_score": v2_scores,
            "v1_pred": v1_pred,
            "v2_pred": v2_pred,
        }
    )
    return ComparisonResult(
        quality=quality,
        v1_metrics=v1_metrics,
        v2_metrics=v2_metrics,
        agreement=agreement,
        score_jsd=score_jsd(v1_scores, v2_scores),
        psi=psi,
        predictions=preds,
    )


def result_payload(result: ComparisonResult) -> dict[str, Any]:
    return {
        "v1": result.v1_metrics,
        "v2": result.v2_metrics,
        "agreement": result.agreement,
        "score_jsd": result.score_jsd,
        "promote_v2": result.promote_v2(),
    }
