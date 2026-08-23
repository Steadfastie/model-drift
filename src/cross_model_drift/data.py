from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from cross_model_drift.config import AppConfig, load_config
from cross_model_drift.features import engineer_features
from cross_model_drift.notebook import create_db_engine
from cross_model_drift.splits import DateWindow, ModelVersion, SplitName, windows_for

TRANSACTION_COLUMNS = (
    "id",
    "sender_id",
    "created",
    "payout_country",
    "payout_currency",
    "amount_usd",
    "fee_usd",
    "status",
    "anti_fraud_status",
    "compliance_status",
)


def load_transactions(
    config: AppConfig | None = None,
    *,
    engine: Engine | None = None,
    columns: Sequence[str] = TRANSACTION_COLUMNS,
    windows: Iterable[DateWindow] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    db = engine or create_db_engine(cfg)
    selected = ", ".join(f"`{col}`" for col in columns)
    sql = f"SELECT {selected} FROM `{cfg.transactions_table}`"
    clauses: list[str] = []
    params: dict[str, datetime] = {}
    if windows:
        window_sql: list[str] = []
        for i, window in enumerate(windows):
            start_key = f"start_{i}"
            end_key = f"end_{i}"
            window_sql.append(f"(created >= :{start_key} AND created < :{end_key})")
            params[start_key] = window.start_ts.to_pydatetime()
            params[end_key] = window.end_exclusive_ts.to_pydatetime()
        clauses.append("(" + " OR ".join(window_sql) + ")")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created, id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    frame = pd.read_sql(text(sql), db, params=params)
    if "created" in frame.columns:
        frame["created"] = pd.to_datetime(frame["created"])
    return frame


def load_split(
    version: ModelVersion,
    split: SplitName,
    config: AppConfig | None = None,
    *,
    engine: Engine | None = None,
    engineer: bool = True,
) -> pd.DataFrame:
    window = windows_for(version)[split]
    frame = load_transactions(config, engine=engine, windows=[window])
    if engineer:
        return engineer_features(frame)
    return frame


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return out


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
