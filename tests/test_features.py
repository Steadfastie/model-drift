import pandas as pd

from cross_model_drift.features import (
    BEHAVIOURAL_FEATURES,
    TARGET_COL,
    add_behavioural_features,
    add_target,
    add_transaction_features,
    engineer_features,
    feature_matrix,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sender_id": ["a", "a", "a", "b"],
            "created": pd.to_datetime(
                [
                    "2021-06-01 10:00:00",
                    "2021-06-01 10:05:00",
                    "2021-06-01 12:00:00",
                    "2021-06-01 10:01:00",
                ]
            ),
            "payout_country": ["US", "DE", "US", "GB"],
            "payout_currency": ["USD", "EUR", "USD", "GBP"],
            "amount_usd": [100.0, 50.0, 25.0, 10.0],
            "fee_usd": [2.0, 1.0, 0.5, 0.2],
            "anti_fraud_status": ["negative", "positive", "negative", "negative"],
            "compliance_status": ["negative", "negative", "negative", "negative"],
        }
    )


def test_target_uses_anti_fraud_only() -> None:
    frame = add_target(_sample_frame())
    assert list(frame[TARGET_COL]) == [0, 1, 0, 0]


def test_transaction_features() -> None:
    frame = add_transaction_features(_sample_frame())
    assert frame.loc[0, "fee_ratio"] == 0.02
    assert frame.loc[0, "hour_of_day"] == 10
    assert frame.loc[0, "is_weekend"] == 0


def test_behavioural_features_use_history_before_current_row() -> None:
    frame = add_target(_sample_frame())
    frame = add_transaction_features(frame)
    frame = add_behavioural_features(frame)
    assert frame.at[0, "sender_tx_count_10m"] == 0
    assert frame.at[1, "sender_tx_count_10m"] == 1
    assert frame.at[1, "sender_avg_amount"] == 100.0
    assert frame.at[1, "sender_previous_fraud_rate"] == 0.0
    assert frame.at[2, "sender_previous_fraud_rate"] == 0.5
    assert frame.at[2, "sender_unique_countries"] == 2


def test_feature_matrix_excludes_leakage_columns() -> None:
    matrix = feature_matrix(engineer_features(_sample_frame()))
    assert "anti_fraud_status" not in matrix.columns
    assert "compliance_status" not in matrix.columns
    assert set(BEHAVIOURAL_FEATURES) <= set(matrix.columns)
