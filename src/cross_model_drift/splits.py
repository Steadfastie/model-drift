from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd

SplitName = Literal["train", "validation", "test", "holdout"]
ModelVersion = Literal["v1", "v2"]


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_exclusive_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end + timedelta(days=1))

    def contains(self, values: pd.Series) -> pd.Series:
        stamps = pd.to_datetime(values)
        return (stamps >= self.start_ts) & (stamps < self.end_exclusive_ts)

    def label(self) -> str:
        return f"{self.start.isoformat()} → {self.end.isoformat()}"


# Chronological windows from the champion/challenger plan (dataset year 2021).
V1_TRAIN = DateWindow(date(2021, 5, 21), date(2021, 7, 21))
V1_VALIDATION = DateWindow(date(2021, 7, 22), date(2021, 7, 28))
V1_TEST = DateWindow(date(2021, 7, 29), date(2021, 8, 4))

V2_TRAIN = DateWindow(date(2021, 5, 28), date(2021, 7, 28))
V2_VALIDATION = DateWindow(date(2021, 7, 29), date(2021, 8, 4))
V2_TEST = DateWindow(date(2021, 8, 5), date(2021, 8, 21))

FINAL_HOLDOUT = DateWindow(date(2021, 8, 5), date(2021, 8, 21))

V1_WINDOWS: dict[SplitName, DateWindow] = {
    "train": V1_TRAIN,
    "validation": V1_VALIDATION,
    "test": V1_TEST,
    "holdout": FINAL_HOLDOUT,
}
V2_WINDOWS: dict[SplitName, DateWindow] = {
    "train": V2_TRAIN,
    "validation": V2_VALIDATION,
    "test": V2_TEST,
    "holdout": FINAL_HOLDOUT,
}


def windows_for(version: ModelVersion) -> dict[SplitName, DateWindow]:
    if version == "v1":
        return V1_WINDOWS
    return V2_WINDOWS


def mask_window(created: pd.Series, window: DateWindow) -> pd.Series:
    return window.contains(created)


def select_window(frame: pd.DataFrame, window: DateWindow, created_col: str = "created") -> pd.DataFrame:
    return frame.loc[mask_window(frame[created_col], window)].copy()


def split_summary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for version, mapping in (("v1", V1_WINDOWS), ("v2", V2_WINDOWS)):
        for name, window in mapping.items():
            rows.append(
                {
                    "version": version,
                    "split": name,
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                    "window": window.label(),
                }
            )
    return rows


def as_datetime(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(value)
