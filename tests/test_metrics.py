import numpy as np
import pandas as pd

from cross_model_drift.compare import compare_models
from cross_model_drift.features import engineer_features
from cross_model_drift.metrics import agreement_metrics, population_stability_index, quality_metrics
from cross_model_drift.models import train_logistic


def test_quality_metrics_perfect_ranking() -> None:
    metrics = quality_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_agreement_directional_rates() -> None:
    metrics = agreement_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert metrics["agreement"] == 0.5
    assert metrics["v1_neg_v2_pos"] == 0.25
    assert metrics["v1_pos_v2_neg"] == 0.25


def test_psi_identical_series_is_near_zero() -> None:
    values = np.linspace(1.0, 10.0, 100)
    assert population_stability_index(values, values) < 0.01


def test_compare_models_on_tiny_frame() -> None:
    frame = engineer_features(
        pd.DataFrame(
            {
                "sender_id": ["a", "a", "b", "b", "c", "c"],
                "created": pd.to_datetime(
                    [
                        "2021-06-01 10:00:00",
                        "2021-06-01 11:00:00",
                        "2021-06-02 10:00:00",
                        "2021-06-02 11:00:00",
                        "2021-06-03 10:00:00",
                        "2021-06-03 11:00:00",
                    ]
                ),
                "payout_country": ["US", "US", "DE", "DE", "GB", "GB"],
                "payout_currency": ["USD", "USD", "EUR", "EUR", "GBP", "GBP"],
                "amount_usd": [10.0, 80.0, 12.0, 90.0, 11.0, 85.0],
                "fee_usd": [0.2, 1.6, 0.2, 1.8, 0.2, 1.7],
                "anti_fraud_status": [
                    "negative",
                    "positive",
                    "negative",
                    "positive",
                    "negative",
                    "positive",
                ],
                "compliance_status": ["negative"] * 6,
            }
        )
    )
    y = frame["is_fraud"]
    model = train_logistic(frame, y)
    result = compare_models(frame, model, model)
    assert result.agreement["agreement"] == 1.0
    assert "pr_auc" in result.v1_metrics
