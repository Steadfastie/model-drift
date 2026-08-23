from datetime import date

import pandas as pd

from cross_model_drift.splits import (
    build_split_plan,
    select_window,
    split_summary_rows,
)

ORIGIN = date(2026, 5, 21)


def _plan():
    return build_split_plan(ORIGIN)


def test_v2_test_matches_final_holdout() -> None:
    plan = _plan()
    v2_test = plan.v2["test"]
    holdout = plan.holdout
    assert v2_test.start == holdout.start
    assert v2_test.end == holdout.end


def test_select_window_is_inclusive_on_end_date() -> None:
    frame = pd.DataFrame(
        {
            "created": pd.to_datetime(
                [
                    "2026-05-20 23:59:59",
                    "2026-05-21 00:00:00",
                    "2026-07-21 23:59:59",
                    "2026-07-22 00:00:00",
                ]
            )
        }
    )
    selected = select_window(frame, _plan().v1["train"])
    assert list(selected["created"].dt.date) == [date(2026, 5, 21), date(2026, 7, 21)]


def test_split_summary_covers_both_versions() -> None:
    rows = split_summary_rows(_plan())
    versions = {row["version"] for row in rows}
    splits = {row["split"] for row in rows}
    assert versions == {"v1", "v2"}
    assert {"train", "validation", "test", "holdout"} <= splits
