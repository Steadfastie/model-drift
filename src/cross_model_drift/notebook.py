from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from cross_model_drift.config import AppConfig, load_config


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


def show(fig: plt.Figure | None = None) -> None:
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


def setup_eda(name: str = "local") -> EdaSession:
    configure_plotting()
    config = load_config(name)
    engine = create_db_engine(config)

    def query(sql: str) -> pd.DataFrame:
        return pd.read_sql(text(sql), engine)

    return EdaSession(
        config=config,
        table=config.transactions_table,
        engine=engine,
        read_sql=query,
        show=show,
    )
