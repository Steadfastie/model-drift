from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd

SplitName = Literal["train", "validation", "test", "holdout"]
ModelVersion = Literal["v1", "v2"]

TRAIN_MONTHS = 2
VALIDATION_DAYS = 7
TEST_DAYS = 7
V2_SHIFT_DAYS = 7


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


@dataclass(frozen=True)
class SplitPlan:
    origin: date
    v1: dict[SplitName, DateWindow]
    v2: dict[SplitName, DateWindow]

    @property
    def holdout(self) -> DateWindow:
        return self.v2["test"]

    def windows_for(self, version: ModelVersion) -> dict[SplitName, DateWindow]:
        if version == "v1":
            return self.v1
        return self.v2


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _inclusive_days(start: date, days: int) -> DateWindow:
    return DateWindow(start, start + timedelta(days=days - 1))


def _version_windows(origin: date) -> dict[SplitName, DateWindow]:
    train = DateWindow(origin, add_months(origin, TRAIN_MONTHS))
    validation = _inclusive_days(train.end + timedelta(days=1), VALIDATION_DAYS)
    test = _inclusive_days(validation.end + timedelta(days=1), TEST_DAYS)
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "holdout": test,
    }


def build_split_plan(origin: date) -> SplitPlan:
    """v1 starts at origin; v2 starts one week later. Same 2m / 1w / 1w ratios.

    Champion/challenger comparison uses v2's test week (also exposed as holdout).
    """
    v1 = _version_windows(origin)
    v2 = _version_windows(origin + timedelta(days=V2_SHIFT_DAYS))
    v1 = {**v1, "holdout": v2["test"]}
    return SplitPlan(origin=origin, v1=v1, v2=v2)


def windows_for(version: ModelVersion, origin: date) -> dict[SplitName, DateWindow]:
    return build_split_plan(origin).windows_for(version)


def mask_window(created: pd.Series, window: DateWindow) -> pd.Series:
    return window.contains(created)


def select_window(frame: pd.DataFrame, window: DateWindow, created_col: str = "created") -> pd.DataFrame:
    return frame.loc[mask_window(frame[created_col], window)].copy()


def split_summary_rows(plan: SplitPlan) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for version, mapping in (("v1", plan.v1), ("v2", plan.v2)):
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
