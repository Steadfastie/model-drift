from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

TARGET_COL = "is_fraud"
ID_COLS = ("sender_id", "created")
LEAKAGE_COLS = ("anti_fraud_status", "compliance_status")

TRANSACTION_FEATURES = (
    "amount_usd",
    "fee_usd",
    "fee_ratio",
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
)
BEHAVIOURAL_FEATURES = (
    "sender_tx_count_10m",
    "sender_tx_count_1h",
    "sender_tx_count_24h",
    "sender_amount_1h",
    "sender_avg_amount",
    "sender_amount_stddev",
    "sender_unique_countries",
    "sender_unique_currencies",
    "sender_previous_fraud_rate",
)
CATEGORICAL_FEATURES = ("payout_country", "payout_currency")
NUMERIC_FEATURES = TRANSACTION_FEATURES + BEHAVIOURAL_FEATURES
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_target(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    status = out["anti_fraud_status"].astype("string").str.lower()
    out[TARGET_COL] = (status == "positive").astype("int8")
    return out


def add_transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    created = pd.to_datetime(out["created"])
    amount = pd.to_numeric(out["amount_usd"], errors="coerce")
    fee = pd.to_numeric(out["fee_usd"], errors="coerce")
    out["amount_usd"] = amount
    out["fee_usd"] = fee
    out["fee_ratio"] = np.where(amount.to_numpy() > 0, fee / amount, 0.0)
    out["log_amount"] = np.log1p(amount.clip(lower=0))
    out["hour_of_day"] = created.dt.hour.astype("int16")
    out["day_of_week"] = created.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    return out


def add_behavioural_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Sender history strictly before the current transaction (no leakage)."""
    if frame.empty:
        out = frame.copy()
        for col in BEHAVIOURAL_FEATURES:
            out[col] = pd.Series(dtype="float64")
        return out

    work = frame.copy()
    work["_row_id"] = np.arange(len(work))
    work = work.sort_values(["sender_id", "created", "_row_id"], kind="mergesort")
    created = pd.to_datetime(work["created"])
    amount = pd.to_numeric(work["amount_usd"], errors="coerce").fillna(0.0)
    work["_created"] = created
    work["_amount"] = amount.to_numpy()
    if TARGET_COL in work.columns:
        fraud = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0.0)
    else:
        fraud = (
            work["anti_fraud_status"].astype("string").str.lower().eq("positive").astype("float64")
        )
    work["_fraud"] = fraud.to_numpy()

    grouped = work.groupby("sender_id", sort=False)
    prior_count = grouped.cumcount()
    prior_amount_sum = grouped["_amount"].cumsum() - work["_amount"]
    prior_amount_sq = grouped["_amount"].transform(lambda s: (s**2).cumsum()) - work["_amount"] ** 2
    prior_fraud_sum = grouped["_fraud"].cumsum() - work["_fraud"]

    work["sender_avg_amount"] = np.divide(
        prior_amount_sum,
        prior_count,
        out=np.zeros(len(work), dtype="float64"),
        where=prior_count > 0,
    )
    variance = np.divide(
        prior_amount_sq,
        prior_count,
        out=np.zeros(len(work), dtype="float64"),
        where=prior_count > 0,
    ) - work["sender_avg_amount"] ** 2
    work["sender_amount_stddev"] = np.sqrt(np.clip(variance, 0.0, None))
    work["sender_previous_fraud_rate"] = np.divide(
        prior_fraud_sum,
        prior_count,
        out=np.zeros(len(work), dtype="float64"),
        where=prior_count > 0,
    )
    work["sender_unique_countries"] = _expanding_nunique(work, "sender_id", "payout_country")
    work["sender_unique_currencies"] = _expanding_nunique(work, "sender_id", "payout_currency")

    work["sender_tx_count_10m"] = _rolling_count(work, "10min")
    work["sender_tx_count_1h"] = _rolling_count(work, "1h")
    work["sender_tx_count_24h"] = _rolling_count(work, "24h")
    work["sender_amount_1h"] = _rolling_sum(work, "1h", "_amount")

    work = work.sort_values("_row_id")
    return work.drop(columns=["_row_id", "_created", "_amount", "_fraud"])


def _expanding_nunique(frame: pd.DataFrame, group_col: str, value_col: str) -> np.ndarray:
    counts = np.zeros(len(frame), dtype="int32")
    seen_by_sender: dict[object, set[object]] = {}
    for i, (sender, value) in enumerate(zip(frame[group_col].to_numpy(), frame[value_col].to_numpy())):
        seen = seen_by_sender.setdefault(sender, set())
        counts[i] = len(seen)
        seen.add(value)
    return counts


def _rolling_count(frame: pd.DataFrame, window: str) -> np.ndarray:
    rolled = (
        frame.groupby("sender_id", sort=False)
        .rolling(window, on="_created", closed="left")["_amount"]
        .count()
        .fillna(0)
    )
    return rolled.to_numpy(dtype="int32")


def _rolling_sum(frame: pd.DataFrame, window: str, value_col: str) -> np.ndarray:
    rolled = (
        frame.groupby("sender_id", sort=False)
        .rolling(window, on="_created", closed="left")[value_col]
        .sum()
        .fillna(0.0)
    )
    return rolled.to_numpy(dtype="float64")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_target(frame)
    out = add_transaction_features(out)
    out = add_behavioural_features(out)
    return out


def feature_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str] = MODEL_FEATURES,
) -> pd.DataFrame:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    matrix = frame.loc[:, list(columns)].copy()
    for col in CATEGORICAL_FEATURES:
        if col in matrix.columns:
            matrix[col] = matrix[col].astype("category")
    return matrix


def target_vector(frame: pd.DataFrame) -> pd.Series:
    if TARGET_COL not in frame.columns:
        raise KeyError(f"missing target column: {TARGET_COL}")
    return frame[TARGET_COL].astype("int8")
