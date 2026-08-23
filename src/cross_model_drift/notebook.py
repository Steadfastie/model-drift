from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from cross_model_drift.config import AppConfig, load_config
from cross_model_drift.splits import SplitPlan, build_split_plan


def create_db_engine(config: AppConfig) -> Engine:
    return create_engine(config.mysql_uri, pool_pre_ping=True)


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 4.8),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def show(fig: Figure | None = None) -> None:
    fig = fig or plt.gcf()
    fig.tight_layout()
    plt.show()


def read_sql(sql: str, config: AppConfig, engine: Engine | None = None) -> pd.DataFrame:
    db = engine or create_db_engine(config)
    return pd.read_sql(text(sql), db)


@dataclass(frozen=True)
class EdaSession:
    config: AppConfig
    table: str
    engine: Engine
    read_sql: Callable[[str], pd.DataFrame]
    show: Callable[..., None]
    min_date: Callable[[str], date]

    def split_plan(self) -> "SplitPlan":
        return build_split_plan(self.min_date("created"))


def setup_eda(name: str = "local") -> EdaSession:
    configure_plotting()
    config = load_config(name)
    engine = create_db_engine(config)

    def query(sql: str) -> pd.DataFrame:
        return pd.read_sql(text(sql), engine)

    def min_date(col: str) -> date:
        value = pd.read_sql(text(f"SELECT MIN(`{col}`) AS d FROM `{config.transactions_table}`"), engine)
        return pd.Timestamp(value.loc[0, "d"]).date()

    return EdaSession(
        config=config,
        table=config.transactions_table,
        engine=engine,
        read_sql=query,
        show=show,
        min_date=min_date,
    )


@dataclass(frozen=True)
class ModelSession(EdaSession):
    artifacts: Path
    threshold: float


def setup_model_session(name: str = "local") -> ModelSession:
    eda = setup_eda(name)
    artifacts = eda.config.artifacts_path()
    artifacts.mkdir(parents=True, exist_ok=True)
    return ModelSession(
        config=eda.config,
        table=eda.table,
        engine=eda.engine,
        read_sql=eda.read_sql,
        show=eda.show,
        artifacts=artifacts,
        threshold=eda.config.classification_threshold,
    )


def artifact_path(*parts: str, config: AppConfig | None = None) -> Path:
    root = (config or load_config()).artifacts_path()
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
